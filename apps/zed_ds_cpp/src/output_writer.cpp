#include "output_writer.h"

#include <filesystem>
#include <iomanip>

namespace zed_ds {

OutputWriter::OutputWriter(std::string path) : path_(std::move(path)) {}

bool OutputWriter::Open() {
  std::filesystem::path p(path_);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  out_.open(path_, std::ios::out | std::ios::app);
  return out_.is_open();
}

void OutputWriter::Write(const FrameResult& frame_result) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!out_.is_open()) return;

  out_ << "{\"ts_ns\":" << frame_result.ts_ns << ",\"frame_id\":" << frame_result.frame_id << ",\"detections\":[";
  for (size_t i = 0; i < frame_result.detections.size(); ++i) {
    const auto& d = frame_result.detections[i];
    out_ << "{\"class_id\":" << d.det.class_id << ",\"label\":\"" << d.det.label << "\",\"conf\":"
         << std::fixed << std::setprecision(4) << d.det.conf << ",\"bbox\":[" << d.det.bbox.left << "," << d.det.bbox.top << ","
         << d.det.bbox.width << "," << d.det.bbox.height << "],\"depth_m\":" << d.depth.depth_m
         << ",\"x_m\":" << d.depth.x_m
         << ",\"y_m\":" << d.depth.y_m
         << ",\"z_m\":" << d.depth.z_m
         << ",\"depth_valid\":" << (d.depth.valid ? "true" : "false")
         << ",\"depth_valid_ratio\":" << d.depth.valid_ratio << "}";
    if (i + 1 < frame_result.detections.size()) out_ << ",";
  }
  out_ << "]}\n";
  out_.flush();
}

}  // namespace zed_ds
