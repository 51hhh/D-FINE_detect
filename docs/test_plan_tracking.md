# 排球追踪 + EKF 轨迹 + 落点预测 — 测试计划

> **系统**: Jetson AGX Orin · DeepStream 6.3 · ZED X · C++17  
> **帧率**: 120 fps (dt ≈ 8.33 ms) · SVGA 960×600  
> **深度范围**: 0.10–20.00 m  
> **摄像头内参**: fx=fy=700, cx=480, cy=300  
> **延迟预算**: 全 probe callback ≤ 35 ms, **新模块总计 ≤ 15 ms**

---

## 0. 测试框架 & 基础设施

### 0.1 框架选择: **GoogleTest + GoogleMock**

| 考量 | GoogleTest | Catch2 |
|------|-----------|--------|
| Mock 支持 | GoogleMock 原生集成 | 需第三方 |
| CMake 集成 | `FetchContent` 一行 | 同 |
| Jetson 兼容 | 官方 CI 覆盖 aarch64 | 可行但社区较小 |
| 参数化测试 | `INSTANTIATE_TEST_SUITE_P` | `GENERATE` |
| **推荐** | **✓** | |

**理由**: BallTracker/EKF 需要 mock DepthEstimator 和时钟, GoogleMock 的 `EXPECT_CALL` / `ON_CALL` 直接可用; 参数化测试适合 Mat6 数值验证。

### 0.2 目录结构

```
apps/zed_ds_cpp/
├── include/
│   ├── mat6.h              ← header-only 6×6 矩阵
│   ├── ball_tracker.h
│   ├── trajectory_estimator.h
│   └── landing_predictor.h
├── src/
│   ├── ball_tracker.cpp
│   ├── trajectory_estimator.cpp
│   └── landing_predictor.cpp
├── tests/
│   ├── CMakeLists.txt       ← FetchContent(googletest)
│   ├── test_mat6.cpp
│   ├── test_ball_tracker.cpp
│   ├── test_trajectory_estimator.cpp
│   ├── test_landing_predictor.cpp
│   ├── test_integration.cpp
│   ├── test_performance.cpp
│   ├── test_helpers.h       ← 公用常量 & 断言宏
│   └── mocks/
│       ├── mock_depth_estimator.h
│       └── mock_clock.h
```

### 0.3 x86 编译隔离策略

| 依赖 | 策略 |
|------|------|
| GStreamer/DeepStream | **不链接**。单元测试仅依赖新模块头文件 + libm |
| ZED SDK | 不链接。DepthEstimator 通过 mock 注入 |
| CUDA | Mat6 纯 CPU；EKF 纯 CPU。无 CUDA 依赖 |
| Eigen | **不使用**。Mat6 自行实现，避免额外依赖 |

```cmake
# tests/CMakeLists.txt skeleton
include(FetchContent)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0)
FetchContent_MakeAvailable(googletest)

add_executable(tracking_tests
  test_mat6.cpp
  test_ball_tracker.cpp
  test_trajectory_estimator.cpp
  test_landing_predictor.cpp
  test_integration.cpp
  test_performance.cpp
  ../src/ball_tracker.cpp
  ../src/trajectory_estimator.cpp
  ../src/landing_predictor.cpp)

target_include_directories(tracking_tests PRIVATE ../include)
target_link_libraries(tracking_tests GTest::gtest_main GTest::gmock)
target_compile_features(tracking_tests PRIVATE cxx_std_17)
```

### 0.4 Mock 策略

```cpp
// mocks/mock_depth_estimator.h
// 将 DepthEstimator::Estimate 依赖抽象为接口, 或通过模板注入
struct IDepthProvider {
  virtual ~IDepthProvider() = default;
  virtual zed_ds::DepthEstimate GetDepth(const zed_ds::BBox& bbox,
                                          int w, int h) const = 0;
};

class MockDepthProvider : public IDepthProvider {
public:
  MOCK_METHOD(zed_ds::DepthEstimate, GetDepth,
              (const zed_ds::BBox&, int, int), (const, override));
};
```

```cpp
// mocks/mock_clock.h
// EKF/Tracker 的 dt 通过外部传入 PTS, 无需 mock 系统时钟.
// 直接在测试中构造 PTS 序列即可.
```

### 0.5 通用测试辅助

```cpp
// test_helpers.h
#pragma once
#include <cmath>

constexpr double kGravity = 9.81;       // m/s²
constexpr double kDt120   = 1.0/120.0;  // 8.333... ms
constexpr double kEpsMat  = 1e-9;       // Mat6 精度
constexpr double kEpsEKF  = 1e-3;       // EKF 允许误差 (m)
constexpr double kEpsLand = 0.05;       // 落点允许误差 (m)

// 相机内参 (SVGA)
constexpr float kFx = 700.0f, kFy = 700.0f;
constexpr float kCx = 480.0f, kCy = 300.0f;

// 从 3D 相机坐标反投影到像素
inline zed_ds::BBox Project3DtoBBox(float X, float Y, float Z,
                                     float box_sz_px = 30.0f) {
  float u = kFx * X / Z + kCx;
  float v = kFy * Y / Z + kCy;
  return {u - box_sz_px/2, v - box_sz_px/2, box_sz_px, box_sz_px};
}
```

---

## 1. 单元测试: Mat6 (header-only 6×6 matrix)

### 1.1 测试用例表

| ID | 名称 | 输入 | 预期输出 | 精度 |
|----|------|------|----------|------|
| M01 | 单位矩阵构造 | `Mat6::Identity()` | diag=1, off-diag=0 | exact |
| M02 | 零矩阵构造 | `Mat6::Zero()` | 全 0 | exact |
| M03 | 单位矩阵乘法 | `I * A` | `A` | `kEpsMat` |
| M04 | 乘法结合律 | `(A*B)*C` vs `A*(B*C)` | 相等 | `kEpsMat` |
| M05 | 转置性质 | `(A^T)^T` | `A` | exact |
| M06 | 转置乘法 | `(AB)^T` vs `B^T * A^T` | 相等 | `kEpsMat` |
| M07 | 对角阵逆 | `diag(2,3,4,5,6,7)^{-1}` | `diag(1/2,1/3,...,1/7)` | `kEpsMat` |
| M08 | 一般逆 AA⁻¹=I | 随机 SPD 矩阵 | `A * A^{-1} ≈ I` | `1e-6` |
| M09 | 奇异矩阵逆 | 全零行 / det=0 | 返回 false 或抛异常 | N/A |
| M10 | 加法交换律 | `A+B` vs `B+A` | 相等 | exact |
| M11 | 标量乘法 | `3*I` | `diag(3,3,3,3,3,3)` | exact |
| M12 | 对称 SPD 保持 | Cholesky 分解后 `L*L^T` | 原矩阵 | `kEpsMat` |

### 1.2 具体数值示例 (M07)

```cpp
TEST(Mat6Test, DiagonalInverse) {
  Mat6 D = Mat6::Zero();
  double diag[] = {2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  for (int i = 0; i < 6; ++i) D(i,i) = diag[i];

  Mat6 Dinv;
  ASSERT_TRUE(D.Inverse(Dinv));

  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(Dinv(i,i), 1.0 / diag[i], kEpsMat);

  Mat6 product = D * Dinv;
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j)
      EXPECT_NEAR(product(i,j), (i==j) ? 1.0 : 0.0, kEpsMat);
}
```

### 1.3 奇异矩阵处理 (M09)

```cpp
TEST(Mat6Test, SingularMatrixReturnsFailure) {
  Mat6 S = Mat6::Zero();
  S(0,0) = 1.0; // rank=1, 奇异
  Mat6 Sinv;
  EXPECT_FALSE(S.Inverse(Sinv)); // 或 EXPECT_THROW 取决于接口设计
}
```

---

## 2. 单元测试: BallTracker

### 2.1 配置参数

```cpp
struct BallTracker::Config {
  float max_assoc_dist_px = 100.0f; // 最近邻关联阈值 (像素)
  int   max_coast_frames  = 10;     // 最大连续缺失帧数
  int   min_hits_confirm  = 3;      // 确认 track 所需最少连续检测
};
```

### 2.2 测试用例表

| ID | 名称 | 场景 | 预期行为 |
|----|------|------|----------|
| T01 | 新建 track | 帧 0 单检测 (400,300) | 分配 track_id=1, 状态=Tentative |
| T02 | track 确认 | 帧 0-2 连续 3 帧同位置, conf>0.8 | 帧 2: status=Confirmed |
| T03 | 关联-位移 | 帧 0→1: (400,300)→(410,305) | 同 track_id, 更新位置 |
| T04 | Coast (滑行) | 帧 5 无检测 | track 存活, coast_count=1 |
| T05 | Coast 恢复 | coast 3帧后再检测 | coast_count=0, track 继续 |
| T06 | track 丢弃 | 连续 11 帧无检测 (>max_coast=10) | track 被删除 |
| T07 | 关联距离超限 | 帧 0: (100,100), 帧 1: (500,500) // dist=566>100 | 旧 track coast, 新建 track_id=2 |
| T08 | 两球区分 | 帧 0: [(100,100),(500,400)], 帧 1: [(105,102),(498,403)] | 两个独立 track |
| T09 | 无检测帧 | 空检测列表 | 所有 track coast+1 |
| T10 | track_id 单调递增 | 创建→丢弃→创建 | 新 track_id > 旧 |

### 2.3 具体测试 (T06: 丢弃)

```cpp
TEST(BallTrackerTest, DropAfterMaxCoast) {
  BallTracker::Config cfg{.max_coast_frames = 10};
  BallTracker tracker(cfg);

  // 帧 0: 创建 track
  auto results = tracker.Update(
    {{.bbox={400,300,30,30}, .conf=0.9f}}, /*frame_id=*/0);
  ASSERT_EQ(results.size(), 1);
  int64_t tid = results[0].track_id;
  EXPECT_GT(tid, 0);

  // 帧 1-10: 无检测 (coast 10 帧)
  for (uint64_t f = 1; f <= 10; ++f) {
    results = tracker.Update({}, f);
    EXPECT_EQ(results.size(), 1); // 仍存在
    EXPECT_EQ(results[0].track_id, tid);
  }

  // 帧 11: 第 11 帧无检测 → 丢弃
  results = tracker.Update({}, 11);
  EXPECT_TRUE(results.empty());
}
```

---

## 3. 单元测试: TrajectoryEstimator (EKF)

### 3.1 状态向量 & 模型

```
状态: x = [x, y, z, vx, vy, vz]^T   (相机坐标系, 米)

过程模型 (dt 来自 PTS 差值):
  x'  = x  + vx*dt
  y'  = y  + vy*dt
  z'  = z  + vz*dt
  vx' = vx
  vy' = vy + g*dt          (y 轴向下时取 +g; 向上取 -g)
  vz' = vz

测量: z_meas = [x_m, y_m, z_m]^T   (DepthEstimate 的 3D 相机坐标)
H = [I_3x3 | 0_3x3]
```

### 3.2 测试用例表

| ID | 名称 | 场景 | 预期 | 精度 |
|----|------|------|------|------|
| E01 | 静止球 | 位置 (1,2,5), v=0, 20 帧 | 状态收敛到 (1,2,5,0,0,0) | `kEpsEKF` |
| E02 | 匀速运动 | v=(2,0,1) m/s, 120 帧 (1 秒) | 位置=(2+2,0,5+1)=(4,0,6) | 0.01m |
| E03 | 自由落体 | 初始 (0,0,5), v=(0,0,0), g=9.81 | 120帧后 y≈0+½*9.81*1²=4.905 | 0.05m |
| E04 | 抛物线 | v=(5,3,0), (0,0,8) | 手动计算各帧位置比对 | 0.05m |
| E05 | 噪声抑制 | 真值 (1,2,5), 测量加 σ=0.1m 高斯噪声 | 均值偏差 < 0.03m (50帧后) | 0.03m |
| E06 | 首帧初始化 | 第一次测量 (3,1,7) | 状态=[3,1,7,0,0,0], P 大 | exact pos |
| E07 | PTS 不均匀 | dt 序列: 8.3, 8.3, 16.6, 8.3 ms | 跳过帧后状态仍连续 | 0.02m |
| E08 | 大 dt 跳跃 | dt = 500ms (PTS 跳变) | 触发 EKF 重置 (参见 §6) | reset |
| E09 | 零 dt | 同 PTS 连续两帧 | 跳过 predict, 仅 update | no crash |

### 3.3 具体测试 (E03: 自由落体)

```cpp
TEST(EKFTest, FreeFall) {
  TrajectoryEstimator::Config cfg;
  cfg.gravity = 9.81; // y-down
  TrajectoryEstimator ekf(cfg);

  // 初始化: 球在 (0, 0, 5), 静止
  double x0=0, y0=0, z0=5;
  uint64_t pts = 0;
  ekf.Init({x0, y0, z0}, pts);

  const int N = 120; // 1 秒
  const double dt = 1.0 / 120.0;

  for (int i = 1; i <= N; ++i) {
    pts += static_cast<uint64_t>(dt * 1e9); // 纳秒
    double t = i * dt;
    // 真值: y = y0 + 0.5*g*t^2
    double y_true = y0 + 0.5 * 9.81 * t * t;
    double meas_y = y_true; // 理想测量
    ekf.Update({x0, meas_y, z0}, pts);
  }

  auto state = ekf.GetState();
  EXPECT_NEAR(state.y,  0.0 + 0.5*9.81*1.0, 0.05);
  EXPECT_NEAR(state.vy, 9.81, 0.1); // 末速度 ≈ g*t = 9.81
  EXPECT_NEAR(state.x,  0.0, kEpsEKF);
  EXPECT_NEAR(state.vx, 0.0, kEpsEKF);
}
```

### 3.4 噪声抑制测试 (E05)

```cpp
TEST(EKFTest, NoiseRejection) {
  TrajectoryEstimator ekf(TrajectoryEstimator::Config{});
  std::mt19937 rng(42);
  std::normal_distribution<double> noise(0.0, 0.1); // σ=10cm

  double true_pos[3] = {1.0, 2.0, 5.0};
  ekf.Init({true_pos[0], true_pos[1], true_pos[2]}, 0);

  double sum_err = 0;
  int count = 0;
  for (int i = 1; i <= 120; ++i) {
    uint64_t pts = static_cast<uint64_t>(i * kDt120 * 1e9);
    double mx = true_pos[0] + noise(rng);
    double my = true_pos[1] + noise(rng);
    double mz = true_pos[2] + noise(rng);
    ekf.Update({mx, my, mz}, pts);
    if (i > 50) { // 收敛后统计
      auto s = ekf.GetState();
      sum_err += std::abs(s.x - true_pos[0])
               + std::abs(s.y - true_pos[1])
               + std::abs(s.z - true_pos[2]);
      ++count;
    }
  }
  double avg_err = sum_err / (3.0 * count);
  EXPECT_LT(avg_err, 0.03); // 平均每轴误差 < 3cm
}
```

---

## 4. 单元测试: LandingPredictor

### 4.1 公式

```
着地时间:  t_land = (vy + sqrt(vy² + 2·g·(y - y_court))) / g
          (y 轴向下, y_court = 地面高度)

着地位置:  x_land = x + vx · t_land
          z_land = z + vz · t_land

置信度:    conf = f(trace(P_pos))   // EKF 位置协方差的迹
```

### 4.2 测试用例表

| ID | 名称 | 输入状态 | y_court | 预期 t_land (s) | 预期 (x,z)_land | 精度 |
|----|------|----------|---------|-----------------|-----------------|------|
| L01 | 垂直自由落体 | (0,0,5), v=(0,0,0) | 2.0m | `sqrt(2·(2.0)/9.81)≈0.6386` | (0, 5) | `kEpsLand` |
| L02 | 斜抛 | (0,0,8), v=(5,-3,2) | 0 (地面) | 解析求根 | 手算 | `kEpsLand` |
| L03 | 球向上运动 | (0,3,5), v=(2,**-5**,1) | 0 | 先上后下, t>0 | 计算 | `kEpsLand` |
| L04 | 球已低于地面 | (0,**-0.5**,5), v=(1,1,0) | 0 | t=0 或特殊标志 | 当前位置 | N/A |
| L05 | 判别式为负 | v=(0,-100,0), y=0.01, y_court=100 | 100 | 无实数解 → 无预测 | invalid | N/A |
| L06 | 零速度在地面 | (3,0,7), v=(0,0,0) | 0 | t=0 | (3, 7) | exact |
| L07 | 大速度发球 | (0,2.5,1), v=(15,-8,20) | 0 | 解析 | 解析 | `kEpsLand` |
| L08 | 协方差低→高置信 | P_pos trace=0.01 | any | conf ≈ 1.0 | — | 0.1 |
| L09 | 协方差高→低置信 | P_pos trace=10.0 | any | conf < 0.3 | — | 0.1 |

### 4.3 具体测试 (L01: 垂直自由落体)

```cpp
TEST(LandingPredictorTest, VerticalFreeFall) {
  LandingPredictor predictor;

  // 球在 y=0, 需下落到 y_court=2.0 (y 向下为正)
  LandingPredictor::State state{
    .x=0, .y=0, .z=5, .vx=0, .vy=0, .vz=0};
  double y_court = 2.0;

  auto result = predictor.Predict(state, y_court);

  // t = sqrt(2*Δy/g) = sqrt(2*2.0/9.81) ≈ 0.6386s
  double expected_t = std::sqrt(2.0 * 2.0 / 9.81);
  ASSERT_TRUE(result.valid);
  EXPECT_NEAR(result.t_land, expected_t, 0.001);
  EXPECT_NEAR(result.x_land, 0.0, kEpsLand);
  EXPECT_NEAR(result.z_land, 5.0, kEpsLand);
}
```

### 4.4 具体测试 (L02: 斜抛完整计算)

```
已知: (x,y,z) = (0, 0, 8), (vx,vy,vz) = (5, -3, 2), y_court = 0, g = 9.81
公式: t = (vy + sqrt(vy² + 2g(y-y_court))) / g
     = (-3 + sqrt(9 + 0)) / 9.81
     = (-3 + 3) / 9.81 = 0   ← 错误, 需检查符号

修正 (y 向下为正, 球在 y=0 向上抛 vy=-3 即向上):
  Δy = y_court - y = 0 - 0 = 0, vy=-3 (向上)
  球先上后落回, t = (–vy + sqrt(vy² + 2gΔy))/g 需要按实际公式

正确解: 二次方程 y + vy·t + ½g·t² = y_court
  0 + (-3)t + 4.905t² = 0
  t(4.905t - 3) = 0
  t = 0 (当前) 或 t = 3/4.905 = 0.6116s

  x_land = 0 + 5 * 0.6116 = 3.058
  z_land = 8 + 2 * 0.6116 = 9.223
```

```cpp
TEST(LandingPredictorTest, AngledThrow) {
  LandingPredictor predictor;

  LandingPredictor::State state{
    .x=0, .y=0, .z=8, .vx=5, .vy=-3, .vz=2};
  double y_court = 0.0;

  auto result = predictor.Predict(state, y_court);
  ASSERT_TRUE(result.valid);

  double expected_t = 3.0 / 4.905; // ≈ 0.6116s
  EXPECT_NEAR(result.t_land, expected_t, 0.005);
  EXPECT_NEAR(result.x_land, 5.0 * expected_t, kEpsLand);
  EXPECT_NEAR(result.z_land, 8.0 + 2.0 * expected_t, kEpsLand);
}
```

---

## 5. 集成测试

### 5.1 完整链路: 检测 → Tracker → EKF → Landing

```
场景: 模拟发球, 120fps, 60 帧 (0.5 秒)
  初始 3D: (0, 2.5, 1.0) m   ← 发球点 (手高 2.5m)
  初速:    (8, -5, 12) m/s    ← 向前方发球
  球场地面: y_court = 0 m

每帧:
  1. 计算真实 3D 位置 (含重力)
  2. 反投影到像素坐标 → 生成 BBox
  3. 构造 DepthEstimate
  4. 传入 BallTracker → track_id
  5. 传入 TrajectoryEstimator → state
  6. 传入 LandingPredictor → landing point
```

| ID | 名称 | 描述 | 验证 |
|----|------|------|------|
| I01 | 完整发球链路 | 60帧模拟发球 | 落点误差 < 0.5m (前 10 帧) → < 0.1m (后 30 帧) |
| I02 | 缺失帧处理 | 帧 10,11,12 无检测 | tracker coast, EKF 纯 predict 3 步, 恢复后落点偏差 < 0.3m |
| I03 | 速度突变 (发球) | 帧 0-20 静止, 帧 21 突然 v=(10,-6,15) | EKF 5 帧内收敛到新速度, 落点 < 0.5m |
| I04 | 全链路 PTS 真实性 | PTS 保证单调递增, dt 在 7.5-9.0ms 间抖动 (±8%) | 无崩溃, 落点偏差 < 0.2m |
| I05 | 先无检测再出现 | 前 5 帧空, 帧 5-60 有检测 | track 正确创建, EKF 正确初始化 |

### 5.2 具体集成测试 (I01)

```cpp
TEST(IntegrationTest, ServeScenario) {
  // --- 配置 ---
  BallTracker tracker({.max_coast_frames=10});
  TrajectoryEstimator ekf({.gravity=9.81});
  LandingPredictor predictor;

  // --- 真值发球弹道 ---
  double x0=0, y0=2.5, z0=1.0;
  double vx=8, vy=-5, vz=12;
  double y_court = 0.0;

  uint64_t pts = 0;
  const double dt = 1.0/120.0;
  bool ekf_initialized = false;

  for (int frame = 0; frame < 60; ++frame) {
    double t = frame * dt;
    double true_x = x0 + vx * t;
    double true_y = y0 + vy * t + 0.5 * 9.81 * t * t;
    double true_z = z0 + vz * t;

    if (true_y < 0 || true_z < 0.1) break; // 球出范围

    // 生成模拟检测
    auto bbox = Project3DtoBBox(true_x, true_y, true_z);
    zed_ds::Detection2D det{.bbox=bbox, .conf=0.95f};
    zed_ds::DepthEstimate depth{
      .depth_m=static_cast<float>(true_z),
      .valid=true,
      .x_m=static_cast<float>(true_x),
      .y_m=static_cast<float>(true_y),
      .z_m=static_cast<float>(true_z)};

    // Tracker
    auto tracks = tracker.Update({det}, frame);
    ASSERT_EQ(tracks.size(), 1);

    // EKF
    pts = static_cast<uint64_t>(t * 1e9);
    if (!ekf_initialized) {
      ekf.Init({depth.x_m, depth.y_m, depth.z_m}, pts);
      ekf_initialized = true;
    } else {
      ekf.Update({depth.x_m, depth.y_m, depth.z_m}, pts);
    }

    // Landing prediction
    auto state = ekf.GetState();
    auto landing = predictor.Predict(
      {state.x, state.y, state.z, state.vx, state.vy, state.vz},
      y_court);

    // 真值落点 (解析)
    // vy_t = vy + g*t, 解 y0+vy*T+0.5g*T² = 0
    // 前 10 帧: 误差 < 0.5m, 后 30 帧: < 0.1m
    if (frame >= 30 && landing.valid) {
      // 解析真值落点
      double a = 0.5 * 9.81;
      double b = vy;
      double c = y0;
      double disc = b*b - 4*a*c;
      double T = (-b + std::sqrt(disc)) / (2*a);
      double true_xland = x0 + vx * T;
      double true_zland = z0 + vz * T;

      EXPECT_NEAR(landing.x_land, true_xland, 0.1);
      EXPECT_NEAR(landing.z_land, true_zland, 0.1);
    }
  }
}
```

---

## 6. EKF 重置 (Auto-Reset) 标准

### 6.1 重置触发条件

| 条件 | 阈值 | 原因 |
|------|------|------|
| **协方差爆炸** | `trace(P) > 100.0` | 长时间无观测导致不确定性过大 |
| **新息过大** | `‖y_innov‖ > 5·sqrt(trace(S))` | 观测与预测严重不符 (如球被替换/重发) |
| **时间间断** | `dt > 300ms` (~36 帧 @ 120fps) | PTS 跳变或长时间丢失 |
| **Track 重新初始化** | BallTracker 分配新 track_id | 球重新检测 = 新实体 |
| **深度无效持续** | 连续 20 帧 depth.valid=false | 深度数据不可靠 |

### 6.2 重置行为

```
重置时:
  1. x = [meas_x, meas_y, meas_z, 0, 0, 0]  ← 位置用观测, 速度归零
  2. P = diag(0.1, 0.1, 0.1, 10, 10, 10)     ← 位置可信, 速度不确定
  3. 清零 prediction_count
  4. log 事件: "EKF reset: reason={reason}, frame={id}"
```

### 6.3 重置测试

| ID | 名称 | 场景 | 验证 |
|----|------|------|------|
| R01 | 协方差爆炸重置 | 30 帧无观测 (纯 predict) | `trace(P) > 100` 后下一次观测触发重置 |
| R02 | 新息重置 | EKF 跟踪 (1,2,5), 突然观测 (8,1,3) | 新息 > 5σ, 重置到 (8,1,3,0,0,0) |
| R03 | PTS 跳变重置 | 正常帧后 dt=500ms | 检测 dt>300ms, 重置 |
| R04 | 新 track 重置 | track_id 变化 | EKF 自动重置 |
| R05 | 重置后收敛 | R02 后继续 20 帧正常观测 | 速度在 10 帧内收敛 |

---

## 7. 边缘情况测试

| ID | 场景 | 输入条件 | 预期行为 |
|----|------|----------|----------|
| EC01 | 30+ 帧无检测 | 连续 35 帧空列表 | tracker 清空, EKF 重置, landing=invalid |
| EC02 | 两球同时出现 | 2 个 Detection2D | tracker 维护 2 个 track, **仅主 track (最近/最大)** 送 EKF |
| EC03 | depth.valid=false | 有 BBox 但深度无效 | tracker 更新 2D, EKF **跳过更新** (纯 predict) |
| EC04 | 极端深度 (19.9m) | z_m≈20.0 (接近 max_depth) | 正常处理, 但 landing confidence 低 |
| EC05 | 极近深度 (0.15m) | z_m≈0.15 (接近 min_depth) | 正常处理, 投影区域可能极大 |
| EC06 | PTS 非单调 | pts_new < pts_prev | **丢弃该帧** 或 clamp dt=0 |
| EC07 | 首帧处理 | frame_id=0, 首次检测 | tracker 创建 track, EKF Init, landing=invalid (无速度) |
| EC08 | NaN/Inf 深度 | depth.x_m = NaN | 视为 depth.valid=false |
| EC09 | BBox 出界 | bbox.left < 0 或超出 960×600 | clamp 或丢弃 |
| EC10 | conf=0 检测 | det.conf < threshold | 不送 tracker |
| EC11 | 单帧同位置多检测 | 3 个 BBox 距离 <5px | tracker 仅关联最高 conf 的 1 个 |

---

## 8. 性能测试

### 8.1 帧内时序预算

| 模块 | 目标 (μs) | 方法 |
|------|-----------|------|
| BallTracker::Update | < 50 | 单球最近邻 = O(N), N≤3 |
| EKF::Predict | < 100 | 6×6 矩阵运算 |
| EKF::Update | < 200 | 包含 S 逆 (6×6) |
| LandingPredictor::Predict | < 20 | 解析公式 |
| **总计** | **< 500** | **远低于 15ms 预算** |

### 8.2 性能测试实现

```cpp
TEST(PerformanceTest, PerFrameBudget) {
  BallTracker tracker({});
  TrajectoryEstimator ekf({});
  LandingPredictor predictor;

  // Warmup
  for (int i = 0; i < 100; ++i) { /* ... */ }

  // 计时 1000 帧
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    zed_ds::Detection2D det{.bbox={400,300,30,30}, .conf=0.9f};
    tracker.Update({det}, i);
    uint64_t pts = static_cast<uint64_t>(i * kDt120 * 1e9);
    ekf.Update({1.0, 2.0, 5.0}, pts);
    predictor.Predict({1,2,5,3,-2,8}, 0.0);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
  double per_frame_us = total_ms * 1000.0 / 1000.0;

  EXPECT_LT(per_frame_us, 500.0);  // < 500μs per frame
  std::cout << "Per-frame: " << per_frame_us << " μs\n";
}
```

### 8.3 内存稳定性

```cpp
TEST(PerformanceTest, MemoryStability) {
  BallTracker tracker({.max_coast_frames=10});
  TrajectoryEstimator ekf({});

  // 模拟 10 分钟 @ 120fps = 72000 帧
  // 周期性: 检测 50 帧 → 消失 15 帧 → 复现
  size_t baseline_mem = GetCurrentRSS(); // 平台相关辅助函数

  for (int cycle = 0; cycle < 1000; ++cycle) {
    for (int f = 0; f < 50; ++f) {
      tracker.Update({{.bbox={400,300,30,30}, .conf=0.9f}},
                      cycle * 65 + f);
    }
    for (int f = 0; f < 15; ++f) {
      tracker.Update({}, cycle * 65 + 50 + f);
    }
  }

  size_t final_mem = GetCurrentRSS();
  // 内存增长应 < 1MB (无持续泄漏)
  EXPECT_LT(final_mem - baseline_mem, 1024 * 1024);
}
```

---

## 9. 测试矩阵总结

| 类别 | 测试数量 | 自动化 | 环境 |
|------|----------|--------|------|
| Mat6 | 12 | ✓ GTest | x86 / aarch64 |
| BallTracker | 11 | ✓ GTest | x86 / aarch64 |
| EKF | 9 | ✓ GTest | x86 / aarch64 |
| LandingPredictor | 9 | ✓ GTest | x86 / aarch64 |
| 集成 | 5 | ✓ GTest | x86 / aarch64 |
| EKF 重置 | 5 | ✓ GTest | x86 / aarch64 |
| 边缘情况 | 11 | ✓ GTest | x86 / aarch64 |
| 性能 | 2 | ✓ GTest | **aarch64 优先** |
| **合计** | **64** | | |

### CI 建议

```yaml
# .github/workflows/tracking-tests.yml (概念)
jobs:
  unit-tests-x86:
    runs-on: ubuntu-22.04
    steps:
      - cmake -B build -DBUILD_TESTS=ON
      - cmake --build build -j$(nproc)
      - ctest --test-dir build --output-on-failure

  perf-tests-orin:  # Self-hosted AGX Orin runner
    runs-on: [self-hosted, jetson-agx-orin]
    steps:
      - cmake -B build -DBUILD_TESTS=ON -DBUILD_PERF_TESTS=ON
      - cmake --build build -j8
      - ctest --test-dir build -R Performance
```

---

## 10. 验收标准 (Definition of Done)

- [ ] 全部 64 个测试在 x86 `Debug` 与 `Release` 下通过
- [ ] 全部测试在 AGX Orin (aarch64, JetPack 5.1.2) 下通过
- [ ] 性能测试: 新模块总计 < 500μs/帧 (远低于 15ms)
- [ ] 72000 帧内存稳定, 无泄漏 (Valgrind/ASan)
- [ ] EKF 自由落体 1 秒误差 < 5cm
- [ ] 发球场景 30 帧后落点误差 < 10cm
- [ ] 所有边缘情况不崩溃 (ASan + UBSan clean)
