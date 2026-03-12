#include "landing_predictor.h"

#include <algorithm>
#include <cmath>

namespace zed_ds {

LandingPredictor::LandingPredictor(Config cfg) : cfg_(cfg) {}

LandingResult LandingPredictor::Predict(const Vec6 &state,
                                        const Mat6 &cov) const {
  LandingResult result;

  // 状态: [x, y, z, vx, vy, vz]
  // ZED 相机坐标系: X=右, Y=下, Z=前(深度)
  // 重力作用于 Y 轴: y'' = +g (向下)
  //
  // 弹道方程 (Y 方向):
  //   y(t) = y0 + vy*t + 0.5*g*t²
  //
  // 落地条件: y(t) = court_y (球场平面的 Y 坐标)
  //   0.5*g*t² + vy*t + (y0 - court_y) = 0
  //   t = (-vy + sqrt(vy² - 2*g*(y0 - court_y))) / g
  //
  // 注意: court_z 配置名保持不变（语义上是"球场高度"），
  //       但实际对应的是 Y 轴坐标(相机坐标系 Y=下)。
  //       court_z > y0 表示球场在球的下方(正常情况)。

  const float y0 = state[1];
  const float vy = state[4]; // dy/dt
  const float g = cfg_.gravity;
  const float court_y = cfg_.court_z; // 配置名保持向后兼容

  // ball 已经在球场平面或下方。
  if (y0 >= court_y) {
    result.x_land = state[0];
    result.y_land = state[2]; // Z 方向(前方)位置
    result.time_to_land_s = 0.0f;
    result.confidence = 1.0f;
    result.valid = true;
    return result;
  }

  // 判别式: vy² + 2*g*(court_y - y0)
  // 因为 court_y > y0 且 g > 0，disc 恒正。
  const float disc = vy * vy + 2.0f * g * (court_y - y0);

  if (disc < 0.0f) {
    return result;
  }

  // 取正根: t = (-vy + sqrt(disc)) / g
  const float t_land = (-vy + std::sqrt(disc)) / g;

  if (t_land < 0.0f || t_land > cfg_.max_flight_time) {
    return result;
  }

  // 预测落点: X 方向(右) 和 Z 方向(前/深度) 匀速外推。
  result.x_land = state[0] + state[3] * t_land;
  result.y_land = state[2] + state[5] * t_land; // Z 方向外推
  result.time_to_land_s = t_land;
  result.valid = true;

  // 不确定度传播（一阶近似）。
  // X 方向: σ²_x_land ≈ σ²_x + t² * σ²_vx
  // Z 方向: σ²_z_land ≈ σ²_z + t² * σ²_vz
  const float t2 = t_land * t_land;
  const float var_x = cov(0, 0) + t2 * cov(3, 3);
  const float var_y = cov(2, 2) + t2 * cov(5, 5); // Z 方向(前方)

  result.sigma_x = std::sqrt(std::max(0.0f, var_x));
  result.sigma_y = std::sqrt(std::max(0.0f, var_y));

  // 置信度: 基于落点不确定度，椭圆面积越小越有信心。
  // confidence = 1 / (1 + σ_x * σ_y)
  const float uncertainty_area = result.sigma_x * result.sigma_y;
  result.confidence = 1.0f / (1.0f + uncertainty_area);

  if (result.confidence < cfg_.min_confidence)
    result.valid = false;

  return result;
}

} // namespace zed_ds
