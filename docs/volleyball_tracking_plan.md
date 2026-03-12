# 排球实时追踪 + 3D 轨迹 + 落点预测 — 实施计划

> 三方评审共识文档（Architect × Developer × Tester）
> 日期: 2025-01

---

## 1. 目标

在现有 ZED X + DeepStream + D-FINE 检测管线基础上，实现：

1. **单球实时追踪** — 分配稳定 track_id，处理丢检/重检测
2. **3D 轨迹估计** — 6-state EKF 估计排球的 3D 位置 + 速度
3. **落点预测** — 解析弹道方程预测着陆位置、到达时间、不确定度
4. **OSD 可视化** — 在视频流上叠加轨迹线 + 预测落点

**性能约束:** 全部新增计算 < 1ms/帧，不影响 120fps 实时管线。

---

## 2. 核心架构决策（三方共识）

| 决策项 | 结论 | 理由 |
|--------|------|------|
| EKF 状态 | **6-state** `[x,y,z,vx,vy,vz]` | 加速度无直接量测，9-state 不可观 |
| 关联算法 | **3D 欧氏距离 + 门限** | 排球 20-40px，快速运动时 IoU=0 |
| 时间基准 | **GST_BUFFER_PTS** | 消除 nvinfer 30-50ms 延迟偏差 |
| dt 保护 | **钳位 [0.1ms, 100ms]** | 防 PTS 跳变/非单调导致 EKF 发散 |
| 阻力模型 | **Phase 1 纯重力**，Phase 2 加阻力 | 飞行 <2s 阻力影响 <15cm |
| 矩阵库 | **自建 header-only Mat6** | 零外部依赖，利于 x86 单测 |
| 测试框架 | **GoogleTest + GoogleMock** | aarch64 支持好，mock 自然 |
| EKF 重置 | **5 条触发规则** | 详见 §6 |

---

## 3. 新增文件

```
apps/zed_ds_cpp/
├── include/
│   ├── mat6.h                    # [新] Header-only 6×6 矩阵运算
│   ├── ball_tracker.h            # [新] 单球追踪器
│   ├── trajectory_estimator.h    # [新] 6-state EKF
│   └── landing_predictor.h       # [新] 弹道落点预测
├── src/
│   ├── ball_tracker.cpp          # [新]
│   ├── trajectory_estimator.cpp  # [新]
│   └── landing_predictor.cpp     # [新]
├── tests/                        # [新] 单元 + 集成测试
│   ├── CMakeLists.txt
│   ├── test_mat6.cpp
│   ├── test_ball_tracker.cpp
│   ├── test_trajectory_estimator.cpp
│   ├── test_landing_predictor.cpp
│   └── test_integration.cpp
```

**修改文件:**
- `include/types.h` — 新增 `TrackInfo`, `LandingInfo` 到 `FrameResult`
- `include/pipeline_builder.h` — `RuntimeContext` 增加 tracker/ekf/predictor
- `src/pipeline_builder.cpp` — `OnInferSrcPadBuffer` 调用追踪链
- `src/main.cpp` — 初始化新组件
- `include/config.h` + `src/config.cpp` — 新 YAML 配置节
- `src/output_writer.cpp` — JSONL 输出新字段
- `CMakeLists.txt` — 新源文件 + 可选测试 target

---

## 4. 类接口设计

### 4.1 Mat6 (header-only)

```cpp
// include/mat6.h
#pragma once
#include <array>
#include <cmath>
#include <cstring>

namespace zed_ds {

// 固定大小 N×N 矩阵，用于 EKF（N=6 或 N=3）。
template <int N>
struct MatN {
  std::array<float, N * N> d{};

  float& operator()(int r, int c)       { return d[r * N + c]; }
  float  operator()(int r, int c) const { return d[r * N + c]; }

  static MatN Identity();
  static MatN Zero();

  MatN operator+(const MatN& o) const;
  MatN operator-(const MatN& o) const;
  MatN operator*(const MatN& o) const;
  MatN operator*(float s) const;
  MatN Transpose() const;

  // 高斯消元法求逆（N≤6 时 ~1μs）。
  bool Inverse(MatN* out) const;
  float Trace() const;
};

// 固定长度向量。
template <int N>
struct VecN {
  std::array<float, N> d{};
  float& operator[](int i)       { return d[i]; }
  float  operator[](int i) const { return d[i]; }
};

using Mat6 = MatN<6>;
using Mat3 = MatN<3>;
using Vec6 = VecN<6>;
using Vec3 = VecN<3>;

// Mat6 × Vec6
Vec6 operator*(const Mat6& m, const Vec6& v);
Vec3 operator*(const Mat3& m, const Vec3& v);

} // namespace zed_ds
```

### 4.2 BallTracker

```cpp
// include/ball_tracker.h
#pragma once
#include "types.h"
#include <cstdint>
#include <optional>

namespace zed_ds {

class BallTracker {
public:
  struct Config {
    float gate_distance_m{2.0f};  // 关联门限（3D 欧氏距离，米）
    int max_coast_frames{15};     // 最大连续丢检帧数
    int confirm_hits{3};          // 确认 track 所需连续命中数
  };

  struct TrackState {
    int64_t track_id{-1};
    float x{0}, y{0}, z{0};      // 最近位置（米）
    int coast_count{0};           // 连续丢检帧数
    int hit_count{0};             // 连续命中帧数
    bool confirmed{false};        // 是否已确认
  };

  explicit BallTracker(Config cfg);

  // 每帧调用。传入检测到的 3D 位置列表（可为空）。
  // 返回当前活跃 track（单球场景最多 1 个）。
  std::optional<TrackState> Update(const std::vector<DetectionResult>& dets);

  // 获取当前 track（不触发更新）。
  std::optional<TrackState> GetTrack() const;

  void Reset();

private:
  Config cfg_;
  std::optional<TrackState> track_;
  int64_t next_id_{1};
};

} // namespace zed_ds
```

### 4.3 TrajectoryEstimator (EKF)

```cpp
// include/trajectory_estimator.h
#pragma once
#include "mat6.h"
#include <cstdint>

namespace zed_ds {

class TrajectoryEstimator {
public:
  struct Config {
    float gravity{9.81f};              // 重力加速度 (m/s²)
    float process_noise_pos{0.05f};    // Q 对角 - 位置 (m²)
    float process_noise_vel{2.0f};     // Q 对角 - 速度 (m²/s²)
    float measure_noise_xy{0.05f};     // R 对角 - XY (m²)
    float measure_noise_z{0.01f};      // R 对角 - 深度 Z (m²)
    float dt_clamp_min{0.0001f};       // dt 下限 (秒)
    float dt_clamp_max{0.1f};          // dt 上限 (秒)
    float cov_trace_reset{100.0f};     // 协方差 trace 超限触发重置
    float innovation_gate_sigma{5.0f}; // 新息门限 (σ)
  };

  explicit TrajectoryEstimator(Config cfg);

  // 预测步骤：根据 dt 推进状态。
  void Predict(float dt);

  // 量测更新：3D 位置。返回是否接受量测（可能被新息门限拒绝）。
  bool Update(float mx, float my, float mz);

  // 初始化/重置到指定位置。
  void Initialize(float x, float y, float z);

  // 状态访问。
  Vec6 State() const { return x_; }
  Mat6 Covariance() const { return P_; }
  float CovTrace() const { return P_.Trace(); }
  bool IsInitialized() const { return initialized_; }

  void Reset();

private:
  Config cfg_;
  Vec6 x_{};     // [x, y, z, vx, vy, vz]
  Mat6 P_{};     // 6×6 协方差矩阵
  bool initialized_{false};
};

} // namespace zed_ds
```

### 4.4 LandingPredictor

```cpp
// include/landing_predictor.h
#pragma once
#include "mat6.h"
#include <cstdint>

namespace zed_ds {

struct LandingResult {
  float x_land{0.0f};         // 预测落点 X (米)
  float y_land{0.0f};         // 预测落点 Y (米)
  float time_to_land_s{0.0f}; // 剩余飞行时间 (秒)
  float confidence{0.0f};     // 置信度 0-1
  float sigma_x{0.0f};        // X 方向不确定度 (米)
  float sigma_y{0.0f};        // Y 方向不确定度 (米)
  bool valid{false};           // 是否有有效预测
};

class LandingPredictor {
public:
  struct Config {
    float court_z{0.0f};             // 球场平面高度 (米，相机坐标系)
    float gravity{9.81f};
    float max_flight_time{5.0f};     // 最大合理飞行时间 (秒)
    float min_confidence{0.1f};      // 低于此不输出
  };

  explicit LandingPredictor(Config cfg);

  // 根据 EKF 状态 + 协方差，预测落点。
  LandingResult Predict(const Vec6& state, const Mat6& cov) const;

private:
  Config cfg_;
};

} // namespace zed_ds
```

---

## 5. 管线集成

### 5.1 types.h 新增

```cpp
struct TrackInfo {
  int64_t track_id{-1};
  float vx_m_s{0.0f};
  float vy_m_s{0.0f};
  float vz_m_s{0.0f};
};

struct FrameResult {
  uint64_t ts_ns{0};
  uint64_t frame_id{0};
  std::vector<DetectionResult> detections;
  TrackInfo track;           // [新]
  LandingResult landing;     // [新]
};
```

### 5.2 RuntimeContext 新增

```cpp
struct RuntimeContext {
  // 现有...
  std::shared_ptr<BallTracker> tracker;           // [新]
  std::shared_ptr<TrajectoryEstimator> trajectory; // [新]
  std::shared_ptr<LandingPredictor> predictor;     // [新]
  uint64_t last_pts_ns{0};                         // [新] 上一帧 PTS
};
```

### 5.3 OnInferSrcPadBuffer 修改

```
现有检测提取
    ↓
[新] BallTracker::Update(detections) → track_id, 3D position
    ↓
[新] dt = clamp(pts - last_pts, min, max)
[新] TrajectoryEstimator::Predict(dt)
[新] if (track.confirmed && detection.depth.valid)
       TrajectoryEstimator::Update(x, y, z)
    ↓
[新] LandingPredictor::Predict(state, cov) → landing result
    ↓
写入 FrameResult → OutputWriter
```

### 5.4 YAML 配置

```yaml
tracker:
  gate_distance_m: 2.0
  max_coast_frames: 15
  confirm_hits: 3

trajectory:
  gravity: 9.81
  process_noise_pos: 0.05
  process_noise_vel: 2.0
  measure_noise_xy: 0.05
  measure_noise_z: 0.01
  cov_trace_reset: 100.0
  innovation_gate_sigma: 5.0

landing:
  court_z: 0.0
  max_flight_time: 5.0
  min_confidence: 0.1
```

---

## 6. EKF 重置策略（5 条规则）

| # | 触发条件 | 动作 |
|---|----------|------|
| 1 | `P.Trace() > 100` | Reset + 用当前量测重新初始化 |
| 2 | 新息 > 5σ（马氏距离） | 拒绝该量测，连续 3 次触发 Reset |
| 3 | `dt > 300ms` (PTS 跳变) | Reset |
| 4 | 新 track_id (tracker 重新分配) | Reset + Initialize(新位置) |
| 5 | 连续 20 帧无有效深度量测 | Reset |

---

## 7. 开发阶段

### Phase 1: BallTracker（追踪器）
- 实现 `mat6.h`（矩阵基础设施）
- 实现 `BallTracker` 类
- 集成到 pipeline：track_id 输出到 JSONL
- 单元测试通过
- AGX 编译 + 冒烟测试

### Phase 2: TrajectoryEstimator（EKF）
- 实现 6-state EKF
- 集成到 pipeline：速度输出到 JSONL
- 单元测试（自由落体、匀速、噪声抑制）
- AGX 编译 + 运行

### Phase 3: LandingPredictor（落点预测）
- 实现弹道解析解
- 协方差传播 → 不确定度
- 集成到 pipeline：落点 + 到达时间输出
- AGX 测试

### Phase 4: OSD 可视化
- 轨迹线叠加（最近 N 帧位置）
- 预测落点椭圆
- 到达时间文字

---

## 8. 测试计划摘要

**64 个测试用例**，按优先级：

| 优先级 | 类别 | 数量 | 内容 |
|--------|------|------|------|
| P0 | Mat6 单测 | 12 | 矩阵运算正确性 |
| P0 | EKF 核心 | 9 | 物理模型 + 收敛性 |
| P0 | Landing 解析 | 9 | 弹道方程正确性 |
| P1 | BallTracker | 11 | 关联/coast/drop |
| P1 | 集成测试 | 5 | 发球全链路 |
| P2 | EKF 重置 | 5 | 异常恢复 |
| P2 | 边缘情况 | 11 | 极端输入 |
| P3 | 性能 | 2 | 时延 + 内存 |

**验收标准:**
- Mat6 精度: `1e-9`
- EKF 位置收敛: `< 3mm` (静止), `< 50mm` (运动)
- 落点误差: `< 5cm` (理想), `< 50cm` (噪声)
- 全链路帧耗时: `< 500μs`

---

## 9. 性能预算

| 组件 | 估计耗时 | 说明 |
|------|----------|------|
| BallTracker::Update | ~10μs | 单球关联 + 状态更新 |
| EKF::Predict | ~30μs | 6×6 矩阵乘法 × 2 |
| EKF::Update | ~50μs | 含 3×3 求逆 |
| Landing::Predict | ~20μs | 解析公式 + 协方差传播 |
| **总计** | **~110μs** | << 15ms 预算 |

---

## 10. 依赖

- **零新外部依赖** — Mat6 自建，GoogleTest 仅测试时需要
- AGX 无需安装任何新包
- CMake 可选 `BUILD_TESTS` 开关控制测试编译
