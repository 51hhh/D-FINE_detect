#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>

namespace zed_ds {

// 性能监控：滑动窗口 FPS 统计和 P95 帧处理间隔计算。
// 说明：当前 latency 样本来源于 nvinfer probe 回调间隔，
// 不是严格端到端时延。
class PerfMonitor {
public:
  explicit PerfMonitor(int window_sec);

  // 标记一帧完成，记录帧处理间隔（毫秒）。
  void MarkFrame(double latency_ms);

  // 滑动窗口内的平均 FPS。
  double AvgFps() const;

  // 滑动窗口内的 P95 帧处理间隔（毫秒）。
  double P95LatencyMs() const;

  // 滑动窗口内的帧数。
  size_t FrameCount() const;

  // 累计处理帧数（不受窗口裁剪影响）。
  uint64_t TotalFrameCount() const;

private:
  mutable std::mutex mu_;
  int window_sec_{5};
  // 统一用 (timestamp, latency_ms) pair，确保窗口同步裁剪。
  struct Sample {
    std::chrono::steady_clock::time_point ts;
    double latency_ms;
  };
  std::deque<Sample> samples_;
  uint64_t total_frames_{0};
};

} // namespace zed_ds
