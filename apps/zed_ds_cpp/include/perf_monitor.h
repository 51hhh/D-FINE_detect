#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>

namespace zed_ds {

// 性能监控：滑动窗口 FPS 统计和 P95 延迟计算。
// 线程安全：所有读写操作均由内部互斥锁保护。
class PerfMonitor {
public:
  explicit PerfMonitor(int window_sec);

  // 标记一帧完成，记录端到端延迟（毫秒）。
  void MarkFrame(double latency_ms);

  // 滑动窗口内的平均 FPS。
  double AvgFps() const;

  // 滑动窗口内的 P95 延迟（毫秒）。
  double P95LatencyMs() const;

  // 滑动窗口内的帧数。
  size_t FrameCount() const;

  // 累计处理帧数（不受窗口裁剪影响）。
  uint64_t TotalFrameCount() const;

private:
  mutable std::mutex mu_;
  int window_sec_{5};
  std::deque<std::chrono::steady_clock::time_point> frame_ts_;
  std::deque<double> latency_ms_;
  uint64_t total_frames_{0};
};

} // namespace zed_ds
