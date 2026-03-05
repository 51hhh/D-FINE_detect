#include "perf_monitor.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace zed_ds {

PerfMonitor::PerfMonitor(int window_sec) : window_sec_(std::max(1, window_sec)) {}

void PerfMonitor::MarkFrame(double latency_ms) {
  auto now = std::chrono::steady_clock::now();
  frame_ts_.push_back(now);
  latency_ms_.push_back(latency_ms);

  while (!frame_ts_.empty() && now - frame_ts_.front() > std::chrono::seconds(window_sec_)) {
    frame_ts_.pop_front();
  }
  while (latency_ms_.size() > 5000) {
    latency_ms_.pop_front();
  }
}

bool PerfMonitor::ShouldFallback(double fallback_fps) const {
  return AvgFps() < fallback_fps;
}

double PerfMonitor::AvgFps() const {
  if (frame_ts_.size() < 2) return 0.0;
  const auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(frame_ts_.back() - frame_ts_.front()).count();
  if (dur <= 0) return 0.0;
  const double sec = static_cast<double>(dur) / 1000.0;
  return static_cast<double>(frame_ts_.size() - 1) / sec;
}

double PerfMonitor::P95LatencyMs() const {
  if (latency_ms_.empty()) return 0.0;
  std::vector<double> v(latency_ms_.begin(), latency_ms_.end());
  std::sort(v.begin(), v.end());
  size_t idx = static_cast<size_t>(std::floor((v.size() - 1) * 0.95));
  return v[idx];
}

size_t PerfMonitor::FrameCount() const { return frame_ts_.size(); }

}  // namespace zed_ds
