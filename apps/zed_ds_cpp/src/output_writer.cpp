#include "output_writer.h"

#include <cmath>
#include <filesystem>
#include <iomanip>

namespace zed_ds {

OutputWriter::OutputWriter(std::string path, int flush_interval_sec)
    : path_(std::move(path)), flush_interval_sec_(flush_interval_sec),
      last_flush_(std::chrono::steady_clock::now()) {}

bool OutputWriter::Open() {
  std::filesystem::path p(path_);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  out_.open(path_, std::ios::out | std::ios::app);
  return out_.is_open();
}

void OutputWriter::Write(const FrameResult &frame_result) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!out_.is_open())
    return;

  out_ << "{\"ts_ns\":" << frame_result.ts_ns
       << ",\"frame_id\":" << frame_result.frame_id << ",\"detections\":[";
  for (size_t i = 0; i < frame_result.detections.size(); ++i) {
    const auto &d = frame_result.detections[i];
    out_ << "{\"class_id\":" << d.det.class_id << ",\"label\":\"" << d.det.label
         << "\",\"conf\":";

    // 安全输出浮点数：NaN/Inf 输出为 null（合法 JSON）。
    auto safe_float = [this](float v) {
      if (std::isfinite(v)) {
        out_ << std::fixed << std::setprecision(4) << v;
      } else {
        out_ << "null";
      }
    };

    safe_float(d.det.conf);
    out_ << ",\"bbox\":[" << d.det.bbox.left << "," << d.det.bbox.top << ","
         << d.det.bbox.width << "," << d.det.bbox.height << "],\"depth_m\":";
    safe_float(d.depth.depth_m);
    out_ << ",\"x_m\":";
    safe_float(d.depth.x_m);
    out_ << ",\"y_m\":";
    safe_float(d.depth.y_m);
    out_ << ",\"z_m\":";
    safe_float(d.depth.z_m);
    out_ << ",\"depth_valid\":" << (d.depth.valid ? "true" : "false")
         << ",\"depth_valid_ratio\":";
    safe_float(d.depth.valid_ratio);
    out_ << "}";
    if (i + 1 < frame_result.detections.size())
      out_ << ",";
  }
  out_ << "]}\n";

  // 按时间间隔批量 flush，避免 120fps 下每帧 fsync。
  if (flush_interval_sec_ <= 0) {
    out_.flush();
  } else {
    auto now = std::chrono::steady_clock::now();
    if (now - last_flush_ >= std::chrono::seconds(flush_interval_sec_)) {
      out_.flush();
      last_flush_ = now;
    }
  }
}

} // namespace zed_ds
