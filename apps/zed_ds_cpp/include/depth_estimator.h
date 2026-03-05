#pragma once

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <mutex>
#include <vector>

#include "types.h"

namespace zed_ds {

class DepthEstimator {
 public:
  struct Config {
    float roi_ratio{0.10f};
    float min_valid_ratio{0.20f};
    float min_depth{0.10f};
    float max_depth{20.0f};
  };

  explicit DepthEstimator(Config cfg);

  void UpdateFromSample(GstSample* sample);
  DepthEstimate Estimate(const BBox& bbox, int frame_w, int frame_h) const;

 private:
  struct DepthFrame {
    int width{0};
    int height{0};
    int stride{0};
    enum class Format { kUnknown, kU16Mm, kF32M } format{Format::kUnknown};
    std::vector<uint8_t> data;
  };

  float ReadDepthMeters(const DepthFrame& frame, int x, int y) const;

  Config cfg_;
  mutable std::mutex mu_;
  DepthFrame frame_;
};

}  // namespace zed_ds
