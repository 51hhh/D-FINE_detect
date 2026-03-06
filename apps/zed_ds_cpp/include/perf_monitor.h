#pragma once

#include <chrono>
#include <deque>
#include <cstddef>
#include <cstdint>

namespace zed_ds {

class PerfMonitor {
 public:
  explicit PerfMonitor(int window_sec);

  void MarkFrame(double latency_ms);
  bool ShouldFallback(double fallback_fps) const;
  double AvgFps() const;
  double P95LatencyMs() const;
  size_t FrameCount() const;
  uint64_t TotalFrameCount() const;

 private:
  int window_sec_{5};
  std::deque<std::chrono::steady_clock::time_point> frame_ts_;
  std::deque<double> latency_ms_;
  uint64_t total_frames_{0};
};

}  // namespace zed_ds
