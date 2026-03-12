# Developer Plan: 排球追踪 + 3D 轨迹估计 + 落点预测

> **角色:** Developer subagent | **日期:** 2026-03-12  
> **输入:** Architect 推荐 (Custom Kalman + EKF 9-state + Physics Landing)  
> **目标:** 可落地的代码级实施方案

---

## 0. Architect 建议评审 & 偏差

| Architect 建议 | Developer 评估 | 调整 |
|---|---|---|
| 9-state EKF (x,y,z,vx,vy,vz,ax,ay,az) | **同意但降为 6-state**。加速度只有重力和阻力，不可直接观测，9-state 缺少加速度量测导致可观性差。改为 6-state [x,y,z,vx,vy,vz]，重力进 process model，阻力作为 process noise 补偿。 | **6-state EKF** |
| 全部在 probe callback 单线程 | **同意**。AGX Orin 单核 A78AE@2.2GHz 15ms 内可完成 6x6 矩阵运算 + 前向模拟。 | 无调整 |
| IoU 关联 | 对单球场景 **改为最近邻欧氏距离**。IoU 在小目标快速移动时重叠为 0 导致关联失败。 | 最近邻 + 门控 |
| 阻力模型 quadratic drag | **Phase 1-3 先忽略阻力**，仅重力。排球飞行时间短 (<2s)，阻力对落点影响 <15cm，后续可加。 | 纯重力先行 |

---

## 1. 文件/类结构

```
apps/zed_ds_cpp/
├── include/
│   ├── types.h              ← 修改：新增 TrackState, TrajectoryState, LandingPrediction, 扩展 FrameResult
│   ├── config.h             ← 修改：新增 TrackerConfig, TrajectoryConfig, PredictorConfig
│   ├── ball_tracker.h       ← 新增
│   ├── trajectory_estimator.h ← 新增
│   ├── landing_predictor.h  ← 新增
│   ├── mat6.h               ← 新增：轻量 6x6 矩阵运算（避免 Eigen 依赖）
│   ├── pipeline_builder.h   ← 修改：RuntimeContext 增加新组件指针
│   └── (其余不变)
├── src/
│   ├── ball_tracker.cpp     ← 新增
│   ├── trajectory_estimator.cpp ← 新增
│   ├── landing_predictor.cpp ← 新增
│   ├── pipeline_builder.cpp ← 修改：probe callback 调用新组件
│   ├── main.cpp             ← 修改：初始化新组件
│   ├── config.cpp           ← 修改：解析新 YAML section
│   ├── output_writer.cpp    ← 修改：输出新字段
│   └── (其余不变)
└── CMakeLists.txt           ← 修改：新增源文件
```

---

## 2. 依赖分析：Eigen vs 自定义矩阵

### 结论：**不用 Eigen，自定义 `mat6.h` header-only**

| 对比项 | Eigen | 自定义 mat6.h |
|---|---|---|
| 矩阵大小 | 任意 | 仅 6x6, 6x3, 3x3, 6x1, 3x1 |
| 编译时间 | +3-5s (Eigen 头重) | 0 |
| 运行效率 | 有 SIMD 优化但 6x6 太小无收益 | 手写展开，无分支 |
| 交叉编译依赖 | 需要 `libeigen3-dev` 在 Jetson 上 | 零依赖 |
| 代码量 | 0 (引用) | ~200 行 header-only |
| 调试 | Eigen 表达式模板难读 | 直接数组，易读 |

6x6 矩阵乘法仅 216 次乘加，`mat6.h` 完全够用。若后续需要更大矩阵（如增加风力模型），再引入 Eigen。

---

## 3. 类 API 设计（头文件级）

### 3.1 `mat6.h` — 轻量定长矩阵

```cpp
#pragma once
#include <array>
#include <cmath>
#include <cstring>

namespace zed_ds {

// 定长向量
template <int N> struct Vec {
  float d[N]{};
  float &operator[](int i) { return d[i]; }
  float operator[](int i) const { return d[i]; }
};

// 定长矩阵 (行优先)
template <int R, int C> struct Mat {
  float d[R * C]{};
  float &operator()(int r, int c) { return d[r * C + c]; }
  float operator()(int r, int c) const { return d[r * C + c]; }

  static Mat Identity() {
    Mat m{};
    for (int i = 0; i < (R < C ? R : C); ++i)
      m(i, i) = 1.0f;
    return m;
  }
  static Mat Zero() { return Mat{}; }
};

using Vec6 = Vec<6>;
using Vec3 = Vec<3>;
using Mat66 = Mat<6, 6>;
using Mat63 = Mat<6, 3>;
using Mat36 = Mat<3, 6>;
using Mat33 = Mat<3, 3>;

// 矩阵乘法 A[R,K] * B[K,C] = Out[R,C]
template <int R, int K, int C>
Mat<R, C> MatMul(const Mat<R, K> &a, const Mat<K, C> &b);

// M * v
template <int R, int C>
Vec<R> MatVecMul(const Mat<R, C> &m, const Vec<C> &v);

// 矩阵转置
template <int R, int C>
Mat<C, R> Transpose(const Mat<R, C> &m);

// 矩阵加法 / 减法
template <int R, int C>
Mat<R, C> MatAdd(const Mat<R, C> &a, const Mat<R, C> &b);

template <int R, int C>
Mat<R, C> MatSub(const Mat<R, C> &a, const Mat<R, C> &b);

// 3x3 矩阵求逆 (Cramer's rule, 对 EKF S 矩阵足够)
bool Invert3x3(const Mat33 &m, Mat33 *out);

// 6x6 矩阵的迹 (trace)
float Trace(const Mat66 &m);

} // namespace zed_ds
```

### 3.2 `ball_tracker.h` — 单球 2D+3D 卡尔曼跟踪器

```cpp
#pragma once
#include <cstdint>
#include <optional>
#include "types.h"

namespace zed_ds {

struct TrackerConfig {
  // 最大允许关联距离 (像素), 超过则视为新球
  float max_assoc_dist_px{150.0f};
  // 丢失多少帧后删除 track
  int max_lost_frames{15};  // 120fps 下约 125ms
  // EKF 预测步的过程噪声 (像素²/帧²)
  float process_noise_px{5.0f};
  // 量测噪声 (像素²)
  float measurement_noise_px{4.0f};
};

// 单球追踪器输出
struct TrackState {
  int64_t track_id{-1};
  // 已连续跟踪帧数
  int age{0};
  // 丢失帧数 (=0 表示当前帧有检测)
  int lost_frames{0};
  // 2D 滤波后的中心 (像素)
  float cx_filtered{0.0f};
  float cy_filtered{0.0f};
  // 2D 像素速度 (px/frame)
  float vx_px{0.0f};
  float vy_px{0.0f};
};

class BallTracker {
public:
  explicit BallTracker(TrackerConfig cfg);

  // 每帧调用一次。传入当前帧的检测结果 (可能为空)。
  // 返回跟踪状态 (std::nullopt 表示无有效 track)。
  std::optional<TrackState> Update(const std::vector<DetectionResult> &dets,
                                   uint64_t frame_id);

  // 返回当前帧 (无检测时) 的纯预测位置
  std::optional<TrackState> Predict() const;

  // 重置 tracker (场景切换时)
  void Reset();

private:
  TrackerConfig cfg_;

  // ── 内部 2D Kalman (4-state: cx, cy, vx, vy) ──
  struct KF2D {
    float x[4]{};     // state: [cx, cy, vx_px, vy_px]
    float P[16]{};     // 4x4 covariance (row-major)
    bool active{false};
    int64_t track_id{0};
    int age{0};
    int lost{0};
    uint64_t last_frame{0};
  };

  KF2D kf_;
  int64_t next_id_{1};

  void KFPredict(KF2D &kf, float dt_frames);
  void KFUpdate(KF2D &kf, float cx, float cy);
  float AssocDistance(const KF2D &kf, float cx, float cy) const;
};

} // namespace zed_ds
```

### 3.3 `trajectory_estimator.h` — 6-state EKF 3D 轨迹

```cpp
#pragma once
#include "mat6.h"
#include "types.h"
#include <cstdint>
#include <optional>

namespace zed_ds {

struct TrajectoryConfig {
  // 过程噪声标准差
  float sigma_accel{8.0f};      // m/s², 排球加速度噪声（扣杀可达 25m/s²）
  // 量测噪声标准差
  float sigma_pos_xy{0.03f};    // m, XY 量测噪声 (~30mm)
  float sigma_pos_z{0.08f};     // m, 深度量测噪声 (~80mm)
  // 重力 (相机坐标系 Y 轴朝下为正，Z 轴朝前)
  // 相机坐标系: X=右, Y=下, Z=前
  float gravity_y{9.81f};       // m/s², Y 朝下
  // 最大允许速度, 超过则判定 EKF 发散并 reset
  float max_velocity{40.0f};    // m/s (排球扣杀 ~30m/s)
  // 最少需要多少次量测才认为速度估计有效
  int min_updates_for_velocity{3};
};

// 3D 轨迹估计输出
struct TrajectoryState {
  Vec6 x{};     // [x, y, z, vx, vy, vz] 相机坐标系 (米, 米/秒)
  Mat66 P{};    // 6x6 协方差
  int updates{0};        // 累计量测更新次数
  bool velocity_valid{false};  // 速度估计是否可靠
  uint64_t last_update_ns{0};
};

class TrajectoryEstimator {
public:
  explicit TrajectoryEstimator(TrajectoryConfig cfg);

  // 用 3D 量测更新 EKF。ts_ns = 当前帧时间戳。
  // meas = [x, y, z] 相机坐标系 (米)。
  void Update(const Vec3 &meas, uint64_t ts_ns);

  // 纯预测 (无量测时调用，例如丢检)。
  void PredictOnly(uint64_t ts_ns);

  // 获取当前状态
  std::optional<TrajectoryState> GetState() const;

  // EKF 是否已初始化
  bool IsInitialized() const { return initialized_; }

  // 重置
  void Reset();

private:
  void PredictStep(float dt);
  bool CheckDivergence() const;

  TrajectoryConfig cfg_;
  Vec6 x_{};
  Mat66 P_{};
  bool initialized_{false};
  int update_count_{0};
  uint64_t last_ts_ns_{0};
};

} // namespace zed_ds
```

### 3.4 `landing_predictor.h` — 物理落点预测

```cpp
#pragma once
#include "mat6.h"
#include "types.h"
#include <optional>

namespace zed_ds {

struct PredictorConfig {
  // 球场平面高度, 相机坐标系 Y 值 (相机安装高度决定)
  // 例如相机在 2.5m 高，球场地面 y_court = 2.5 (Y 朝下)
  float y_court{2.5f};
  // 前向模拟步长
  float sim_dt{0.005f};       // 5ms 步长
  // 最大模拟时长
  float max_sim_time{3.0f};   // 最多模拟 3 秒
  // 最小要求的速度分量 (m/s), vy 必须有向下的分量才做预测
  float min_vy_down{0.5f};
  // 是否启用二次阻力
  bool enable_drag{false};
  // 排球阻力系数 (Cd*A*rho / (2*m)), 典型值 ~0.21
  float drag_coeff{0.21f};
};

// 落点预测输出
struct LandingPrediction {
  bool valid{false};
  // 预计落地坐标 (相机坐标系, 米)
  float x_land{0.0f};
  float y_land{0.0f};  // ≈ y_court
  float z_land{0.0f};
  // 预计飞行时间 (秒)
  float time_to_land{0.0f};
  // 落点不确定性 (标准差, 米)
  float sigma_x{0.0f};
  float sigma_z{0.0f};
  // 置信度 [0, 1], 由协方差推导
  float confidence{0.0f};
};

class LandingPredictor {
public:
  explicit LandingPredictor(PredictorConfig cfg);

  // 根据 EKF 状态预测落点
  // 要求: trajectory 有有效速度估计
  std::optional<LandingPrediction> Predict(const Vec6 &state,
                                            const Mat66 &cov) const;

private:
  // 牛顿法求解 landing time (纯重力)
  float SolveLandingTimeAnalytical(float y, float vy) const;
  // 前向模拟 (含阻力)
  std::optional<LandingPrediction> SimulateLanding(const Vec6 &state) const;
  // 协方差传播
  void PropagateCovarianceToLanding(const Mat66 &cov, float t_land,
                                     float *sigma_x, float *sigma_z) const;

  PredictorConfig cfg_;
};

} // namespace zed_ds
```

---

## 4. 集成改动详解

### 4.1 `types.h` — 新增结构体，扩展 FrameResult

```cpp
// 在现有 FrameResult 中追加：
struct FrameResult {
  uint64_t ts_ns{0};
  uint64_t frame_id{0};
  std::vector<DetectionResult> detections;

  // ── 新增 ──
  // 跟踪状态 (nullopt = 无有效 track)
  std::optional<TrackState> track;
  // 轨迹状态 (nullopt = EKF 未初始化)
  std::optional<TrajectoryState> trajectory;
  // 落点预测 (nullopt = 条件不满足)
  std::optional<LandingPrediction> landing;
};
```

需要 `#include <optional>` 和前向声明 / include 新头文件。

### 4.2 `config.h` — 新增配置段

```cpp
struct AppConfig {
  CameraConfig camera;
  InferConfig infer;
  DepthConfig depth;
  PerfConfig perf;
  OutputConfig output;

  // ── 新增 ──
  TrackerConfig tracker;
  TrajectoryConfig trajectory;
  PredictorConfig predictor;
};
```

### 4.3 `pipeline_builder.h` — RuntimeContext 新增成员

```cpp
struct RuntimeContext {
  std::shared_ptr<DepthEstimator> depth;
  std::shared_ptr<OutputWriter> writer;
  std::shared_ptr<PerfMonitor> perf;
  bool enable_osd{true};
  const CameraConfig *camera{nullptr};
  std::atomic<int64_t> last_probe_time_ns{0};
  std::atomic<uint64_t> drift_check_count{0};

  // ── 新增 ──
  std::shared_ptr<BallTracker> tracker;
  std::shared_ptr<TrajectoryEstimator> trajectory;
  std::shared_ptr<LandingPredictor> predictor;
};
```

### 4.4 `pipeline_builder.cpp` — probe callback 改动

在 `OnInferSrcPadBuffer` 中，现有代码构建 `frame_out.detections` 之后，**追加**：

```cpp
// ── [新增] 跟踪 + 轨迹 + 落点 ──
if (ctx.tracker) {
  auto track_opt = ctx.tracker->Update(frame_out.detections, frame_out.frame_id);
  frame_out.track = track_opt;

  if (ctx.trajectory) {
    if (track_opt && !frame_out.detections.empty()) {
      // 找到被跟踪的检测, 取第一个有效深度的检测用于 EKF 更新
      const auto &d = frame_out.detections[0];
      if (d.depth.valid) {
        Vec3 meas{};
        meas[0] = d.depth.x_m;
        meas[1] = d.depth.y_m;
        meas[2] = d.depth.z_m;
        ctx.trajectory->Update(meas, frame_out.ts_ns);
      } else {
        ctx.trajectory->PredictOnly(frame_out.ts_ns);
      }
    } else if (ctx.trajectory->IsInitialized()) {
      // 丢检：纯预测
      ctx.trajectory->PredictOnly(frame_out.ts_ns);
    }

    auto traj = ctx.trajectory->GetState();
    frame_out.trajectory = traj;

    // 落点预测
    if (ctx.predictor && traj && traj->velocity_valid) {
      frame_out.landing = ctx.predictor->Predict(traj->x, traj->P);
    }
  }
}
```

### 4.5 `main.cpp` — 初始化

在 `RunOnce()` 中 `auto ctx = ...` 之后追加：

```cpp
auto tracker = std::make_shared<zed_ds::BallTracker>(cfg.tracker);
auto trajectory = std::make_shared<zed_ds::TrajectoryEstimator>(cfg.trajectory);
auto predictor = std::make_shared<zed_ds::LandingPredictor>(cfg.predictor);
ctx->tracker = tracker;
ctx->trajectory = trajectory;
ctx->predictor = predictor;
```

### 4.6 `config.cpp` — YAML 解析

新增 `tracker:`, `trajectory:`, `predictor:` 节点解析，映射到对应 Config 结构体。所有字段都有默认值，YAML 中缺失时不报错。

### 4.7 `output_writer.cpp` — 新字段输出

在 `Write()` 末尾 `"]}\n"` 之前追加 track/trajectory/landing 的 JSON 字段：

```json
{
  "track": {"id":1, "age":45, "lost":0},
  "trajectory": {"vx":-2.1, "vy":3.5, "vz":8.2, "speed":9.1},
  "landing": {"x":1.2, "z":5.8, "t":0.65, "sigma_x":0.3, "sigma_z":0.4, "conf":0.72}
}
```

### 4.8 `CMakeLists.txt` — 新增源文件

```cmake
add_executable(zed_ds_app
  src/main.cpp
  src/config.cpp
  src/pipeline_builder.cpp
  src/depth_estimator.cpp
  src/output_writer.cpp
  src/perf_monitor.cpp
  src/ball_tracker.cpp          # 新增
  src/trajectory_estimator.cpp  # 新增
  src/landing_predictor.cpp     # 新增
)
```

无需新增库依赖。`mat6.h` 是 header-only。

---

## 5. YAML 配置示例

```yaml
tracker:
  max_assoc_dist_px: 150.0
  max_lost_frames: 15
  process_noise_px: 5.0
  measurement_noise_px: 4.0

trajectory:
  sigma_accel: 8.0
  sigma_pos_xy: 0.03
  sigma_pos_z: 0.08
  gravity_y: 9.81
  max_velocity: 40.0
  min_updates_for_velocity: 3

predictor:
  y_court: 2.5          # 相机安装高度(m), 即地面在相机坐标系的 Y 值
  sim_dt: 0.005
  max_sim_time: 3.0
  min_vy_down: 0.5
  enable_drag: false
  drag_coeff: 0.21
```

---

## 6. EKF 核心数学

### 6.1 状态向量 (6-state)

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$$

### 6.2 过程模型 (匀加速 + 重力)

$$\mathbf{x}_{k+1} = F \mathbf{x}_k + \mathbf{g} \cdot \Delta t$$

其中：

$$F = \begin{bmatrix} 1 & 0 & 0 & \Delta t & 0 & 0 \\ 0 & 1 & 0 & 0 & \Delta t & 0 \\ 0 & 0 & 1 & 0 & 0 & \Delta t \\ 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}, \quad
\mathbf{g} \cdot \Delta t = \begin{bmatrix} 0 \\ \frac{1}{2} g \Delta t^2 \\ 0 \\ 0 \\ g \Delta t \\ 0 \end{bmatrix}$$

协方差预测：

$$P_{k+1} = F P_k F^T + Q$$

其中：

$$Q = \sigma_a^2 \begin{bmatrix} \frac{\Delta t^4}{4} I_3 & \frac{\Delta t^3}{2} I_3 \\ \frac{\Delta t^3}{2} I_3 & \Delta t^2 I_3 \end{bmatrix}$$

### 6.3 量测模型

$$H = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}$$

$$R = \text{diag}(\sigma_{xy}^2, \sigma_{xy}^2, \sigma_z^2)$$

### 6.4 落点解析解 (纯重力)

$$t_{land} = \frac{v_y + \sqrt{v_y^2 + 2g(y_{court} - y)}}{g}$$

$$x_{land} = x + v_x \cdot t_{land}$$
$$z_{land} = z + v_z \cdot t_{land}$$

### 6.5 协方差传播到落点

$$J_{land} = \frac{\partial [x_{land}, z_{land}]}{\partial \mathbf{x}}$$

$$\Sigma_{land} = J_{land} \cdot P \cdot J_{land}^T$$

---

## 7. 开发阶段

### Phase 1: BallTracker (2D Kalman, ~2天)

**目标:** 单球 2D 追踪，输出 track_id + 滤波像素坐标  
**测试:** 用离线视频回放，验证 track_id 稳定性、丢检恢复  
**可交付:** 
- `ball_tracker.h/.cpp`, `mat6.h`
- types.h 增加 `TrackState`, `FrameResult::track`
- probe callback 中调用 tracker
- JSONL 输出 track 字段

**验收标准:**
- 连续检测时 track_id 不跳变
- 丢检 <15 帧自动恢复
- 无额外延迟 (< 0.1ms)

### Phase 2: TrajectoryEstimator (3D EKF, ~3天)

**目标:** 6-state EKF 融合 3D 量测，输出位置 + 速度  
**测试:** 对比原始 3D 坐标 vs EKF 输出，验证噪声平滑效果  
**可交付:**
- `trajectory_estimator.h/.cpp`
- types.h 增加 `TrajectoryState`
- probe callback 中 EKF update/predict
- JSONL 输出 velocity

**验收标准:**
- 速度估计在匀速段与距离差分一致 (误差 <20%)
- EKF 不发散 (无 NaN/Inf)
- 深度跳变时滤波器有平滑效果
- 收到第 3 个量测后 velocity_valid=true

### Phase 3: LandingPredictor (落点预测, ~2天)

**目标:** 根据 EKF 状态预测排球落点坐标和时间  
**测试:** 已知抛物线轨迹验证落点精度  
**可交付:**
- `landing_predictor.h/.cpp`
- types.h 增加 `LandingPrediction`, `FrameResult::landing`
- JSONL 输出 landing

**验收标准:**
- 抛物线轨迹落点误差 <30cm (5m 距离)
- 不会对静止/上升中的球给出预测
- 协方差正确传播，置信度随时间收敛

### Phase 4: OSD 可视化 (~1天)

**目标:** 在 nvosd 上绘制轨迹 + 落点标记  
**方案:** 在 probe callback 中修改 `NvDsDisplayMeta`：
- 绘制最近 N 帧的 2D 轨迹点 (线段连接)
- 落点预测用 circle + text 标注
- 置信度用颜色编码 (绿→黄→红)

**实现要点:**
```cpp
NvDsDisplayMeta *dmeta = nvds_acquire_display_meta_from_pool(batch_meta);
// 轨迹线
dmeta->num_lines = trajectory_history.size() - 1;
for (...) {
  dmeta->line_params[i] = ...;
}
// 落点圆
dmeta->num_circles = 1;
dmeta->circle_params[0] = ...;
nvds_add_display_meta_to_frame(frame_meta, dmeta);
```

**注意:** `NvDsDisplayMeta` 单次最多 16 条线段 / 16 个圆，需要多次 acquire。

---

## 8. 风险评估

| 风险 | 严重度 | 概率 | 缓解策略 |
|---|---|---|---|
| **EKF 发散 (遮挡/丢检)** | 高 | 中 | `PredictOnly()` 最多连续调用 `max_lost_frames` 次后自动 Reset；检测恢复后重新初始化而非硬更新 |
| **深度噪声 >5m** | 高 | 高 | ZED X SVGA 在 5-8m 处 σ_z ≈ 5-15cm，已在 R 矩阵中建模；>10m 数据标记为 `valid=false` 不喂入 EKF |
| **扣杀瞬间速度突变** | 中 | 中 | σ_accel=8 m/s² 已覆盖大多数场景；检测到速度突变 >30m/s 时 Reset 并用新量测重新初始化 |
| **相机坐标系 vs 场地坐标系** | 中 | 低 | Phase 1-3 全部在相机坐标系工作；`y_court` 通过相机安装高度标定；后续可加外参旋转矩阵 |
| **120fps 下 dt 过小 (8.3ms)** | 低 | 低 | EKF 用纳秒时间戳计算 dt，无累积误差；dt=0 时跳过 Predict |
| **NvDsDisplayMeta 元素上限** | 低 | 高 | 每次最多 16 线段，需多次 `nvds_acquire_display_meta_from_pool`；轨迹线限制为最近 15 帧 |
| **nvinfer 延迟导致 3D 位置滞后** | 中 | 确定 | 检测结果滞后 3-6 帧 (~25-50ms)，EKF 的 Predict 步天然补偿此延迟；可选: 用推理帧 PTS 而非当前时间作为量测时间戳 |

### 关键设计决策——量测时间戳

推理结果对应的是 **推理帧** 的时间而非当前时间。当前代码用 `steady_clock::now()` 生成 ts_ns，但推理帧本身有 `GST_BUFFER_PTS`。

**建议:** EKF 量测时间戳用 `GST_BUFFER_PTS(buf)` (即推理帧原始采集时间)。EKF 自动通过 predict 步推算到当前时间，消除检测延迟的位置估计偏差。

---

## 9. 性能预算 (AGX Orin A78AE@2.2GHz)

| 组件 | 操作 | 估算耗时 |
|---|---|---|
| BallTracker | 4x4 KF predict+update + 关联 | **< 0.01ms** |
| TrajectoryEstimator | 6x6 EKF predict + update (含 3x3 逆) | **~0.02ms** |
| LandingPredictor | 解析解 + 协方差传播 (6x2 Jacobian) | **~0.01ms** |
| LandingPredictor (含阻力模拟) | 前向模拟 ~600 步 (3s/5ms) | **~0.1ms** |
| OSD 轨迹绘制 | 写入 DisplayMeta | **< 0.01ms** |
| **合计** | | **~0.15ms** |

**远低于 15ms 预算。** 即使 EKF 升级到 9-state 也仅增加到 ~0.3ms。瓶颈始终在 nvinfer (20-35ms) 和深度估计 (~1ms)，跟踪/预测的开销可以忽略。

---

## 10. 坐标系定义

```
相机坐标系 (ZED X, 左目):
       Y (下)
       |
       |
       +------→ X (右)
      /
     /
    Z (前, 远离相机)

地面: y = y_court (正值, 相机在地面上方)
球网: 需要另外标定 z_net, x_net_left, x_net_right (Phase 4+)
```

相机安装于球场侧面或端线后方，高度 `h_cam` 米。则 `y_court = h_cam`。

---

## 11. 对 Tester subagent 的建议

### 需要的测试矩阵

| 测试项 | 方法 |
|---|---|
| mat6.h 正确性 | 单元测试: 矩阵乘法、逆、trace 对比 numpy |
| BallTracker 关联 | 合成序列: 球从左到右匀速移动，验证 track_id 连续 |
| BallTracker 丢检恢复 | 合成序列: 连续 10 帧丢检后恢复 |
| EKF 收敛 | 合成抛物线轨迹 + 高斯噪声，验证速度估计 |
| EKF 发散检测 | 喂入极端值 (NaN, 1000m)，验证不崩溃 |
| LandingPredictor 精度 | 已知初速度的抛物线，对比解析解 |
| LandingPredictor 边界 | 球向上运动 / 已在地面以下 / 速度为 0 |
| 端到端 JSONL | 验证输出 JSON 格式合法，新字段存在且类型正确 |
| 性能回归 | 比较加入跟踪前后的 P95 帧间隔 |

---

## 12. 总结

| 项 | 决策 |
|---|---|
| 状态维度 | **6-state** (非 Architect 建议的 9-state) — 可观性更好 |
| 矩阵库 | **自定义 mat6.h** (非 Eigen) — 零依赖，编译快 |
| 关联方式 | **最近邻欧氏距离** (非 IoU) — 小目标快速运动更鲁棒 |
| 阻力模型 | **Phase 1-3 纯重力，Phase 4+ 可选阻力** |
| 量测时间戳 | **GST_BUFFER_PTS** (非 steady_clock::now) |
| 新增文件 | 3 个 .h + 3 个 .cpp + 1 个 header-only mat6.h |
| 修改文件 | types.h, config.h/cpp, pipeline_builder.h/cpp, main.cpp, output_writer.cpp, CMakeLists.txt |
| 总新增代码量 | 估计 ~800-1000 行 C++ |
| 性能影响 | **~0.15ms/frame，可忽略** |
