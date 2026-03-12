#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace zed_ds {

// 2D 检测框（像素坐标）。
struct BBox {
  float left{0.0f};
  float top{0.0f};
  float width{0.0f};
  float height{0.0f};
};

// 单个 2D 检测结果。
struct Detection2D {
  BBox bbox;
  int class_id{-1};
  float conf{0.0f};
  std::string label;
  int64_t track_id{-1};
};

// 深度估计结果，包含相机坐标系下的 3D 坐标。
struct DepthEstimate {
  float depth_m{0.0f};
  float valid_ratio{0.0f};
  bool valid{false};
  // 目标中心点在相机坐标系下的三维坐标（米）。
  float x_m{0.0f};
  float y_m{0.0f};
  float z_m{0.0f};
};

// 聚合单个目标的 2D 检测 + 深度估计。
struct DetectionResult {
  Detection2D det;
  DepthEstimate depth;
};

// 追踪信息。
struct TrackInfo {
  int64_t track_id{-1};
  float vx_m_s{0.0f};
  float vy_m_s{0.0f};
  float vz_m_s{0.0f};
};

// 落点预测信息（前向声明避免循环依赖，用 POD 复制）。
struct LandingInfo {
  float x_land{0.0f};
  float y_land{0.0f};
  float time_to_land_s{0.0f};
  float confidence{0.0f};
  float sigma_x{0.0f};
  float sigma_y{0.0f};
  bool valid{false};
};

// 单帧所有检测结果。
struct FrameResult {
  uint64_t ts_ns{0};
  uint64_t frame_id{0};
  std::vector<DetectionResult> detections;
  TrackInfo track;
  LandingInfo landing;
};

} // namespace zed_ds
