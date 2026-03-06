#pragma once

#include <gst/gst.h>

#include <memory>

#include "config.h"
#include "depth_estimator.h"
#include "output_writer.h"
#include "perf_monitor.h"

namespace zed_ds {

struct CameraIntrinsics {
  float fx{700.0f};
  float fy{700.0f};
  float cx{480.0f};
  float cy{300.0f};
  float k1{0.0f};
  float k2{0.0f};
  float p1{0.0f};
  float p2{0.0f};
  float k3{0.0f};
  bool use_distortion_correction{true};
};

struct RuntimeContext {
  std::shared_ptr<DepthEstimator> depth;
  std::shared_ptr<OutputWriter> writer;
  std::shared_ptr<PerfMonitor> perf;
  bool enable_osd{true};
  CameraIntrinsics intrinsics;
};

struct PipelineHandles {
  GstElement* pipeline{nullptr};
  GstElement* depth_sink{nullptr};
  GstElement* infer{nullptr};
  GstElement* sink{nullptr};
};

bool BuildPipeline(const AppConfig& cfg, const RuntimeContext& ctx, PipelineHandles* out, std::string* err);

}  // namespace zed_ds
