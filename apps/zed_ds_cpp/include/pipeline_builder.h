#pragma once

#include <gst/gst.h>

#include <memory>

#include "config.h"
#include "depth_estimator.h"
#include "output_writer.h"
#include "perf_monitor.h"

namespace zed_ds {

struct RuntimeContext {
  std::shared_ptr<DepthEstimator> depth;
  std::shared_ptr<OutputWriter> writer;
  std::shared_ptr<PerfMonitor> perf;
  bool enable_osd{true};
};

struct PipelineHandles {
  GstElement* pipeline{nullptr};
  GstElement* depth_sink{nullptr};
  GstElement* infer{nullptr};
  GstElement* sink{nullptr};
};

bool BuildPipeline(const AppConfig& cfg, const RuntimeContext& ctx, PipelineHandles* out, std::string* err);

}  // namespace zed_ds
