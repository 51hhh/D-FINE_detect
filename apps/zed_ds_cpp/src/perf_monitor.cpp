#include "perf_monitor.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace zed_ds {

PerfMonitor::PerfMonitor(int window_sec)
    : window_sec_(std::max(1, window_sec)) {}

void PerfMonitor::MarkFrame(double latency_ms) {
  std::lock_guard<std::mutex> lk(mu_);
  auto now = std::chrono::steady_clock::now();
  ++total_frames_;
  frame_ts_.push_back(now);
  latency_ms_.push_back(latency_ms);

  // 裁剪滑动窗口外的帧时间戳。
  while (!frame_ts_.empty() &&
         now - frame_ts_.front() > std::chrono::seconds(window_sec_)) {
    frame_ts_.pop_front();
  }
  // 限制延迟样本数量，防止内存无限增长。
  while (latency_ms_.size() > 5000) {
    latency_ms_.pop_front();
  }
}

double PerfMonitor::AvgFps() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (frame_ts_.size() < 2)
    return 0.0;
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                       frame_ts_.back() - frame_ts_.front())
                       .count();
  if (dur <= 0)
    return 0.0;
  const double sec = static_cast<double>(dur) / 1000.0;
  return static_cast<double>(frame_ts_.size() - 1) / sec;
}

double PerfMonitor::P95LatencyMs() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (latency_ms_.empty())
    return 0.0;
  std::vector<double> v(latency_ms_.begin(), latency_ms_.end());
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(std::floor((v.size() - 1) * 0.95));
  return v[idx];
}

size_t PerfMonitor::FrameCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return frame_ts_.size();
}

uint64_t PerfMonitor::TotalFrameCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return total_frames_;
}

} // namespace zed_ds
