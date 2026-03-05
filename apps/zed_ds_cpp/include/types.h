#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace zed_ds {

struct BBox {
  float left{0.0f};
  float top{0.0f};
  float width{0.0f};
  float height{0.0f};
};

struct Detection2D {
  BBox bbox;
  int class_id{-1};
  float conf{0.0f};
  std::string label;
  int64_t track_id{-1};
};

struct DepthEstimate {
  float depth_m{0.0f};
  float valid_ratio{0.0f};
  bool valid{false};
};

struct DetectionResult {
  Detection2D det;
  DepthEstimate depth;
};

struct FrameResult {
  uint64_t ts_ns{0};
  uint64_t frame_id{0};
  std::vector<DetectionResult> detections;
};

}  // namespace zed_ds
