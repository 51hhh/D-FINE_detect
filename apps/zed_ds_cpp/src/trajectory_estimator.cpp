#include "trajectory_estimator.h"

#include <algorithm>
#include <cmath>

namespace zed_ds {

TrajectoryEstimator::TrajectoryEstimator(Config cfg) : cfg_(cfg) {}

void TrajectoryEstimator::Initialize(float x, float y, float z) {
  x_ = Vec6{};
  x_[0] = x;
  x_[1] = y;
  x_[2] = z;
  // 速度初始化为 0，靠后续量测收敛。

  // 初始协方差：位置较确定，速度不确定。
  P_ = Mat6::Zero();
  P_(0, 0) = 0.1f;  // σ²_x
  P_(1, 1) = 0.1f;  // σ²_y
  P_(2, 2) = 0.1f;  // σ²_z
  P_(3, 3) = 25.0f; // σ²_vx (5 m/s 不确定度)
  P_(4, 4) = 25.0f; // σ²_vy
  P_(5, 5) = 25.0f; // σ²_vz

  initialized_ = true;
  no_update_count_ = 0;
}

void TrajectoryEstimator::Reset() {
  x_ = Vec6{};
  P_ = Mat6::Zero();
  initialized_ = false;
  no_update_count_ = 0;
}

void TrajectoryEstimator::CheckAndResetIfNeeded() {
  if (!initialized_)
    return;
  // 规则 1: 协方差 trace 超限。
  if (P_.Trace() > cfg_.cov_trace_reset) {
    Reset();
    return;
  }
  // 规则 5: 连续无量测帧数超限。
  if (no_update_count_ > cfg_.max_no_update_frames) {
    Reset();
  }
}

void TrajectoryEstimator::Predict(float dt) {
  if (!initialized_)
    return;

  // 钳位 dt。
  dt = std::clamp(dt, cfg_.dt_clamp_min, cfg_.dt_clamp_max);

  // 状态转移矩阵 F：
  // x'  = x + vx*dt
  // y'  = y + vy*dt + 0.5*g*dt²  （重力使 Y 增加，Y 轴向下）
  // z'  = z + vz*dt
  // vx' = vx
  // vy' = vy + g*dt
  // vz' = vz
  //
  // ZED 相机坐标系: X=右, Y=下, Z=前(深度)
  // 重力 = +Y 方向 (向下 = +g)
  Mat6 F = Mat6::Identity();
  F(0, 3) = dt;
  F(1, 4) = dt;
  F(2, 5) = dt;

  // 状态预测。
  Vec6 x_pred;
  x_pred[0] = x_[0] + x_[3] * dt;
  x_pred[1] = x_[1] + x_[4] * dt + 0.5f * cfg_.gravity * dt * dt;
  x_pred[2] = x_[2] + x_[5] * dt;
  x_pred[3] = x_[3];
  x_pred[4] = x_[4] + cfg_.gravity * dt;
  x_pred[5] = x_[5];

  x_ = x_pred;

  // 过程噪声 Q（简化为对角矩阵，位速解耦）。
  Mat6 Q = Mat6::Zero();
  const float dt2 = dt * dt;
  Q(0, 0) = cfg_.process_noise_pos * dt2;
  Q(1, 1) = cfg_.process_noise_pos * dt2;
  Q(2, 2) = cfg_.process_noise_pos * dt2;
  Q(3, 3) = cfg_.process_noise_vel * dt2;
  Q(4, 4) = cfg_.process_noise_vel * dt2;
  Q(5, 5) = cfg_.process_noise_vel * dt2;

  // 协方差预测: P = F * P * F^T + Q。
  P_ = F * P_ * F.Transpose() + Q;

  no_update_count_++;
  CheckAndResetIfNeeded();
}

bool TrajectoryEstimator::Update(float mx, float my, float mz) {
  if (!initialized_)
    return false;

  // 量测模型 H: z = H * x, H = [I₃ | 0₃]
  // 新息 y = z - H*x = z - [x, y, z]
  Vec3 innovation;
  innovation[0] = mx - x_[0];
  innovation[1] = my - x_[1];
  innovation[2] = mz - x_[2];

  // 量测噪声 R。
  Mat3 R = Mat3::Zero();
  R(0, 0) = cfg_.measure_noise_xy;
  R(1, 1) = cfg_.measure_noise_xy;
  R(2, 2) = cfg_.measure_noise_z;

  // S = H * P * H^T + R = P[0:3, 0:3] + R
  Mat3 S = SubBlock33(P_, 0, 0) + R;

  // 新息门限检查（马氏距离）。
  Mat3 S_inv;
  if (!S.Inverse(&S_inv))
    return false;

  // 马氏距离² = innovation^T * S^-1 * innovation
  Vec3 Si = S_inv * innovation;
  float mahal2 = innovation.Dot(Si);
  float gate2 = cfg_.innovation_gate_sigma * cfg_.innovation_gate_sigma * 3.0f;
  // ×3 是因为 3 维自由度。

  if (mahal2 > gate2) {
    // 规则 2: 新息过大，拒绝量测。
    return false;
  }

  // 卡尔曼增益 K = P * H^T * S^-1
  // H^T 是 6×3 矩阵 [I₃; 0₃]，所以 P * H^T = P 的前 3 列。
  // K (6×3) = P[:, 0:3] * S^-1
  // 手动计算避免通用矩阵乘法的维度不匹配。
  float K[6][3]{};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < 3; ++j) {
      float s = 0.0f;
      for (int k = 0; k < 3; ++k)
        s += P_(i, k) * S_inv(k, j);
      K[i][j] = s;
    }
  }

  // 状态更新: x = x + K * innovation
  for (int i = 0; i < 6; ++i) {
    float correction = 0.0f;
    for (int j = 0; j < 3; ++j)
      correction += K[i][j] * innovation[j];
    x_[i] += correction;
  }

  // 协方差更新: P = (I - K*H) * P
  // K*H 是 6×6 矩阵，其中 (K*H)(i,j) = K(i,j) 当 j<3, 否则 0。
  Mat6 KH = Mat6::Zero();
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 3; ++j)
      KH(i, j) = K[i][j];

  Mat6 I_KH = Mat6::Identity() - KH;
  P_ = I_KH * P_;

  // 对称化（消除数值误差积累）。
  P_ = (P_ + P_.Transpose()) * 0.5f;

  no_update_count_ = 0;
  return true;
}

} // namespace zed_ds
