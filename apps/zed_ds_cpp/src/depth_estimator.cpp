#include "depth_estimator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace zed_ds {

DepthEstimator::DepthEstimator(Config cfg) : cfg_(cfg) {}

void DepthEstimator::UpdateFromSample(GstSample* sample) {
  if (!sample) return;

  GstCaps* caps = gst_sample_get_caps(sample);
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!caps || !buffer) return;

  GstStructure* s = gst_caps_get_structure(caps, 0);
  if (!s) return;

  int w = 0;
  int h = 0;
  gst_structure_get_int(s, "width", &w);
  gst_structure_get_int(s, "height", &h);
  const char* fmt = gst_structure_get_string(s, "format");

  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) return;

  DepthFrame local;
  local.width = w;
  local.height = h;
  local.stride = (w > 0) ? static_cast<int>(map.size / std::max(1, h)) : 0;
  local.data.assign(map.data, map.data + map.size);

  if (fmt && std::strcmp(fmt, "GRAY16_LE") == 0) {
    local.format = DepthFrame::Format::kU16Mm;
  } else if (fmt && (std::strcmp(fmt, "GRAY32F") == 0 || std::strcmp(fmt, "F32LE") == 0)) {
    local.format = DepthFrame::Format::kF32M;
  } else {
    local.format = DepthFrame::Format::kUnknown;
  }

  gst_buffer_unmap(buffer, &map);

  std::lock_guard<std::mutex> lk(mu_);
  frame_ = std::move(local);
}

float DepthEstimator::ReadDepthMeters(const DepthFrame& frame, int x, int y) const {
  if (x < 0 || y < 0 || x >= frame.width || y >= frame.height) return 0.0f;

  if (frame.format == DepthFrame::Format::kU16Mm) {
    const int idx = y * frame.stride + x * 2;
    if (idx + 1 >= static_cast<int>(frame.data.size())) return 0.0f;
    uint16_t mm = static_cast<uint16_t>(frame.data[idx] | (frame.data[idx + 1] << 8));
    return static_cast<float>(mm) / 1000.0f;
  }

  if (frame.format == DepthFrame::Format::kF32M) {
    const int idx = y * frame.stride + x * 4;
    if (idx + 3 >= static_cast<int>(frame.data.size())) return 0.0f;
    float d;
    std::memcpy(&d, &frame.data[idx], sizeof(float));
    return d;
  }

  return 0.0f;
}

DepthEstimate DepthEstimator::Estimate(const BBox& bbox, int frame_w, int frame_h) const {
  std::lock_guard<std::mutex> lk(mu_);

  DepthEstimate out;
  if (frame_.data.empty() || frame_.width <= 0 || frame_.height <= 0) return out;

  const float roi_w = std::max(8.0f, bbox.width * cfg_.roi_ratio);
  const float roi_h = std::max(8.0f, bbox.height * cfg_.roi_ratio);
  const float cx = bbox.left + bbox.width * 0.5f;
  const float cy = bbox.top + bbox.height * 0.5f;

  int x0 = static_cast<int>(std::floor(cx - roi_w * 0.5f));
  int y0 = static_cast<int>(std::floor(cy - roi_h * 0.5f));
  int x1 = static_cast<int>(std::ceil(cx + roi_w * 0.5f));
  int y1 = static_cast<int>(std::ceil(cy + roi_h * 0.5f));

  x0 = std::max(0, std::min(x0, frame_w - 1));
  y0 = std::max(0, std::min(y0, frame_h - 1));
  x1 = std::max(0, std::min(x1, frame_w));
  y1 = std::max(0, std::min(y1, frame_h));

  int total = 0;
  int valid = 0;
  double sum = 0.0;
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      ++total;
      const float d = ReadDepthMeters(frame_, x, y);
      if (!std::isfinite(d)) continue;
      if (d < cfg_.min_depth || d > cfg_.max_depth) continue;
      sum += d;
      ++valid;
    }
  }

  if (total <= 0 || valid <= 0) return out;

  out.valid_ratio = static_cast<float>(valid) / static_cast<float>(total);
  if (out.valid_ratio < cfg_.min_valid_ratio) return out;

  out.depth_m = static_cast<float>(sum / valid);
  out.valid = true;
  return out;
}

}  // namespace zed_ds
