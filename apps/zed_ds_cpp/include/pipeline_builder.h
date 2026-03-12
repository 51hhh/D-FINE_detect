#pragma once

#include <gst/gst.h>

#include <memory>

#include <atomic>
#include <chrono>
#include <deque>
#include <utility>

#include "ball_tracker.h"
#include "config.h"
#include "depth_estimator.h"
#include "landing_predictor.h"
#include "output_writer.h"
#include "perf_monitor.h"
#include "trajectory_estimator.h"

namespace zed_ds {

// 运行时上下文：聚合所有运行时组件的共享所有权。
// 通过 shared_ptr 在管线回调间安全共享，消除裸指针悬垂风险。
struct RuntimeContext {
  std::shared_ptr<DepthEstimator> depth;
  std::shared_ptr<OutputWriter> writer;
  std::shared_ptr<PerfMonitor> perf;
  bool enable_osd{true};
  // 直接持有相机配置的引用，消除 CameraIntrinsics 重复定义。
  const CameraConfig *camera{nullptr};

  // probe 回调状态（替代 thread_local，GStreamer 不保证 probe 线程固定）。
  std::atomic<int64_t> last_probe_time_ns{0};
  std::atomic<uint64_t> drift_check_count{0};

  // 追踪 + 轨迹 + 落点预测组件。
  std::shared_ptr<BallTracker> tracker;
  std::shared_ptr<TrajectoryEstimator> trajectory;
  std::shared_ptr<LandingPredictor> predictor;
  uint64_t last_pts_ns{0}; // 上一帧 PTS，用于计算 EKF dt。
  int64_t last_tracker_id{-1}; // 上次 track_id，用于检测 track 切换。

  // OSD 轨迹绘制：最近 N 帧的像素坐标 (u, v)。
  static constexpr int kMaxTrailPoints = 30;
  std::deque<std::pair<float, float>> trail_pixels;
};

// 管线元素句柄，用于统一释放。
struct PipelineHandles {
  GstElement *pipeline{nullptr};
  GstElement *depth_sink{nullptr};
  GstElement *infer{nullptr};
  GstElement *sink{nullptr};
};

// 构建 GStreamer 管线。
// ctx 通过 shared_ptr 传入，确保回调生命周期安全。
bool BuildPipeline(const AppConfig &cfg, std::shared_ptr<RuntimeContext> ctx,
                   PipelineHandles *out, std::string *err);

} // namespace zed_ds
