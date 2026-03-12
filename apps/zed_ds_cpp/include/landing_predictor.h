#pragma once

#include "mat6.h"
#include <cstdint>

namespace zed_ds {

/// 落点预测结果。
struct LandingResult {
  float x_land{0.0f};
  float y_land{0.0f};
  float time_to_land_s{0.0f};
  float confidence{0.0f};   // 0-1
  float sigma_x{0.0f};      // X 方向不确定度 (米)
  float sigma_y{0.0f};      // Y 方向不确定度 (米)
  bool valid{false};
};

/// 弹道落点预测器：解析解 + 协方差传播。
class LandingPredictor {
public:
  struct Config {
    float court_z{0.0f};          // 球场平面高度 (米, 相机坐标系, Z 轴向前)
    float gravity{9.81f};
    float max_flight_time{5.0f};
    float min_confidence{0.1f};
  };

  explicit LandingPredictor(Config cfg);

  /// 根据 EKF 状态和协方差预测落点。
  /// state: [x, y, z, vx, vy, vz]
  /// ZED 相机坐标系: X=右, Y=下, Z=前(深度)
  /// 重力 = +Y 方向（向下加速）
  /// 落点判定: Y >= court_y (球降到球场平面)
  /// 输出: x_land = X方向(右), y_land = Z方向(前方深度)
  /// court_z 配置项实际对应 Y 轴球场高度(向后兼容)。
  LandingResult Predict(const Vec6 &state, const Mat6 &cov) const;

  float CourtY() const { return cfg_.court_z; }

private:
  Config cfg_;
};

} // namespace zed_ds
