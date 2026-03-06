#pragma once

#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "types.h"

namespace zed_ds {

// 深度估计器：从 appsink 接收深度帧，按检测框中心 ROI 计算均值深度。
// 线程安全：深度帧由 GStreamer 线程写入、推理 probe 回调线程读取。
class DepthEstimator {
public:
  struct Config {
    float roi_ratio{0.10f};
    float min_valid_ratio{0.20f};
    float min_depth{0.10f};
    float max_depth{20.0f};
  };

  explicit DepthEstimator(Config cfg);

  // 由 appsink 回调调用，更新内部深度帧缓存。
  void UpdateFromSample(GstSample *sample);

  // 估算指定检测框的深度。frame_w/frame_h 为推理帧尺寸。
  DepthEstimate Estimate(const BBox &bbox, int frame_w, int frame_h) const;

  // 返回最近一帧深度的 PTS（纳秒），用于帧对齐检查。
  uint64_t LastDepthPtsNs() const;

private:
  struct DepthFrame {
    int width{0};
    int height{0};
    int stride{0};
    enum class Format { kUnknown, kU16Mm, kF32M } format{Format::kUnknown};
    std::vector<uint8_t> data;
    uint64_t pts_ns{0}; // 深度帧的 PTS，用于帧对齐。
  };

  float ReadDepthMeters(const DepthFrame &frame, int x, int y) const;

  Config cfg_;
  mutable std::mutex mu_;
  DepthFrame frame_;
};

} // namespace zed_ds
