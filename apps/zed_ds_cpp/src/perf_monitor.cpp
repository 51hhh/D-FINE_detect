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
  samples_.push_back({now, latency_ms});

  // 统一裁剪滑动窗口外的样本。
  while (!samples_.empty() &&
         now - samples_.front().ts > std::chrono::seconds(window_sec_)) {
    samples_.pop_front();
  }
}

double PerfMonitor::AvgFps() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (samples_.size() < 2)
    return 0.0;
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                       samples_.back().ts - samples_.front().ts)
                       .count();
  if (dur <= 0)
    return 0.0;
  const double sec = static_cast<double>(dur) / 1000.0;
  return static_cast<double>(samples_.size() - 1) / sec;
}

double PerfMonitor::P95LatencyMs() const {
  std::lock_guard<std::mutex> lk(mu_);
  if (samples_.empty())
    return 0.0;
  std::vector<double> v;
  v.reserve(samples_.size());
  for (const auto &s : samples_) {
    v.push_back(s.latency_ms);
  }
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(std::floor((v.size() - 1) * 0.95));
  return v[idx];
}

size_t PerfMonitor::FrameCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return samples_.size();
}

uint64_t PerfMonitor::TotalFrameCount() const {
  std::lock_guard<std::mutex> lk(mu_);
  return total_frames_;
}

} // namespace zed_ds
