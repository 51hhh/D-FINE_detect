#pragma once

#include "mat6.h"
#include <cstdint>

namespace zed_ds {

/// 6-state EKF 排球轨迹估计器。
/// 状态: [x, y, z, vx, vy, vz]
/// 过程模型: 匀速 + 重力 (az = -g)
/// 量测模型: 3D 位置 [x, y, z]
class TrajectoryEstimator {
public:
  struct Config {
    float gravity{9.81f};
    float process_noise_pos{0.05f};    // Q 对角 - 位置噪声 σ² (m²)
    float process_noise_vel{2.0f};     // Q 对角 - 速度噪声 σ² (m²/s²)
    float measure_noise_xy{0.05f};     // R 对角 - XY 量测噪声 σ² (m²)
    float measure_noise_z{0.01f};      // R 对角 - Z 量测噪声 σ² (m²)
    float dt_clamp_min{0.0001f};       // dt 下限 (秒)
    float dt_clamp_max{0.1f};          // dt 上限 (秒)
    float cov_trace_reset{100.0f};     // 协方差 trace 超限触发重置
    float innovation_gate_sigma{5.0f}; // 新息门限 (σ 倍数)
    int max_no_update_frames{20};      // 连续无量测帧数上限
  };

  explicit TrajectoryEstimator(Config cfg);

  /// 预测步：推进 dt 秒。
  void Predict(float dt);

  /// 量测更新：传入 3D 位置 (米)。返回 true 表示接受量测。
  bool Update(float mx, float my, float mz);

  /// 用指定位置初始化状态（速度归零、协方差重置）。
  void Initialize(float x, float y, float z);

  Vec6 State() const { return x_; }
  Mat6 Covariance() const { return P_; }
  float CovTrace() const { return P_.Trace(); }
  bool IsInitialized() const { return initialized_; }

  /// 获取速度分量。
  float Vx() const { return x_[3]; }
  float Vy() const { return x_[4]; }
  float Vz() const { return x_[5]; }

  void Reset();

private:
  void CheckAndResetIfNeeded();

  Config cfg_;
  Vec6 x_{};
  Mat6 P_{};
  bool initialized_{false};
  int no_update_count_{0};
};

} // namespace zed_ds
