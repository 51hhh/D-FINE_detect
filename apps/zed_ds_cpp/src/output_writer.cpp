#include "output_writer.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>

namespace zed_ds {

namespace {

std::string JsonEscape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (c < 0x20) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
        out += buf;
      } else {
        out.push_back(static_cast<char>(c));
      }
      break;
    }
  }
  return out;
}

} // namespace

OutputWriter::OutputWriter(std::string path, int flush_interval_sec,
                           int64_t max_file_bytes)
    : path_(std::move(path)), flush_interval_sec_(flush_interval_sec),
      max_file_bytes_(max_file_bytes),
      last_flush_(std::chrono::steady_clock::now()) {}

bool OutputWriter::Open() {
  std::filesystem::path p(path_);
  if (p.has_parent_path()) {
    std::filesystem::create_directories(p.parent_path());
  }
  out_.open(path_, std::ios::out | std::ios::app);
  if (out_.is_open()) {
    // 初始化已有文件大小计数。
    current_bytes_ = static_cast<int64_t>(std::filesystem::file_size(path_));
  }
  return out_.is_open();
}

void OutputWriter::Write(const FrameResult &frame_result) {
  std::lock_guard<std::mutex> lk(mu_);
  if (!out_.is_open())
    return;

  // 安全输出浮点数：NaN/Inf 输出为 null（合法 JSON）。
  auto safe_float = [this](float v) {
    if (std::isfinite(v)) {
      out_ << std::fixed << std::setprecision(4) << v;
    } else {
      out_ << "null";
    }
  };

  out_ << "{\"ts_ns\":" << frame_result.ts_ns
       << ",\"frame_id\":" << frame_result.frame_id << ",\"detections\":[";
  for (size_t i = 0; i < frame_result.detections.size(); ++i) {
    const auto &d = frame_result.detections[i];
    out_ << "{\"class_id\":" << d.det.class_id << ",\"label\":\""
         << JsonEscape(d.det.label) << "\",\"conf\":";

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
  out_ << "]";

  // 追踪信息。
  if (frame_result.track.track_id >= 0) {
    out_ << ",\"track\":{\"id\":" << frame_result.track.track_id
         << ",\"vx\":";
    safe_float(frame_result.track.vx_m_s);
    out_ << ",\"vy\":";
    safe_float(frame_result.track.vy_m_s);
    out_ << ",\"vz\":";
    safe_float(frame_result.track.vz_m_s);
    out_ << "}";
  }

  // 落点预测。
  if (frame_result.landing.valid) {
    out_ << ",\"landing\":{\"x\":";
    safe_float(frame_result.landing.x_land);
    out_ << ",\"y\":";
    safe_float(frame_result.landing.y_land);
    out_ << ",\"t\":";
    safe_float(frame_result.landing.time_to_land_s);
    out_ << ",\"conf\":";
    safe_float(frame_result.landing.confidence);
    out_ << ",\"sx\":";
    safe_float(frame_result.landing.sigma_x);
    out_ << ",\"sy\":";
    safe_float(frame_result.landing.sigma_y);
    out_ << "}";
  }

  out_ << "}\n";

  // 追踪已写入字节数（近似值，足够用于轮转判断）。
  current_bytes_ = out_.tellp();

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

  // 检查是否需要日志轮转。
  RotateIfNeeded();
}

void OutputWriter::RotateIfNeeded() {
  if (max_file_bytes_ <= 0 || current_bytes_ < max_file_bytes_)
    return;

  out_.close();

  // 轮转命名: path.1.jsonl, path.2.jsonl, ...
  ++rotation_index_;
  std::filesystem::path orig(path_);
  std::string stem = orig.stem().string();
  std::string ext = orig.extension().string();
  std::filesystem::path rotated =
      orig.parent_path() /
      (stem + "." + std::to_string(rotation_index_) + ext);

  std::error_code ec;
  std::filesystem::rename(path_, rotated, ec);
  if (ec) {
    std::cerr << "[warn] 日志轮转失败: " << ec.message() << std::endl;
  }

  out_.open(path_, std::ios::out | std::ios::trunc);
  current_bytes_ = 0;
}

} // namespace zed_ds
