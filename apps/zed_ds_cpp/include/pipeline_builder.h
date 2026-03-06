#pragma once

#include <gst/gst.h>

#include <memory>

#include "config.h"
#include "depth_estimator.h"
#include "output_writer.h"
#include "perf_monitor.h"

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
