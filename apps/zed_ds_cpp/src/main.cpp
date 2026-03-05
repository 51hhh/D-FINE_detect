#include <gst/gst.h>

#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

#include "config.h"
#include "depth_estimator.h"
#include "output_writer.h"
#include "perf_monitor.h"
#include "pipeline_builder.h"
#include "version.h"

namespace {
volatile sig_atomic_t g_stop = 0;
GMainLoop* g_active_loop = nullptr;

void OnSignal(int) {
  g_stop = 1;
  if (g_active_loop) {
    g_main_loop_quit(g_active_loop);
  }
}

struct LoopState {
  GMainLoop* loop{nullptr};
  std::shared_ptr<zed_ds::PerfMonitor> perf;
  double fallback_fps{60.0};
  bool allow_fallback{true};
  bool fallback_triggered{false};
};

gboolean OnPerfTick(gpointer user_data) {
  auto* st = static_cast<LoopState*>(user_data);
  if (!st || !st->perf) return G_SOURCE_CONTINUE;

  const double fps = st->perf->AvgFps();
  const size_t frames = st->perf->FrameCount();
  const double p95 = st->perf->P95LatencyMs();
  std::cout << "[perf] fps=" << fps << " frames=" << frames << " p95_latency_ms=" << p95 << std::endl;

  if (st->allow_fallback && !st->fallback_triggered && frames >= 20 && st->perf->ShouldFallback(st->fallback_fps)) {
    st->fallback_triggered = true;
    std::cerr << "[warn] FPS dropped below fallback threshold. Triggering fallback restart." << std::endl;
    g_main_loop_quit(st->loop);
    return G_SOURCE_REMOVE;
  }

  return G_SOURCE_CONTINUE;
}

gboolean OnBusMsg(GstBus*, GstMessage* msg, gpointer user_data) {
  auto* loop = static_cast<GMainLoop*>(user_data);

  switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_EOS:
      g_main_loop_quit(loop);
      break;
    case GST_MESSAGE_ERROR: {
      GError* err = nullptr;
      gchar* dbg = nullptr;
      gst_message_parse_error(msg, &err, &dbg);
      std::cerr << "[error] " << (err ? err->message : "unknown") << std::endl;
      if (dbg) std::cerr << "[debug] " << dbg << std::endl;
      if (err) g_error_free(err);
      if (dbg) g_free(dbg);
      g_main_loop_quit(loop);
      break;
    }
    default:
      break;
  }

  return G_SOURCE_CONTINUE;
}

bool RunOnce(const zed_ds::AppConfig& cfg, bool allow_fallback, bool* fallback_triggered) {
  auto depth = std::make_shared<zed_ds::DepthEstimator>(zed_ds::DepthEstimator::Config{
      .roi_ratio = cfg.depth.roi_ratio,
      .min_valid_ratio = cfg.depth.min_valid_ratio,
      .min_depth = cfg.depth.min_depth,
      .max_depth = cfg.depth.max_depth,
  });

  auto writer = std::make_shared<zed_ds::OutputWriter>(cfg.output.json_path);
  if (!writer->Open()) {
    std::cerr << "[error] failed to open json output: " << cfg.output.json_path << std::endl;
    return false;
  }

  auto perf = std::make_shared<zed_ds::PerfMonitor>(cfg.perf.window_sec);

  zed_ds::RuntimeContext ctx;
  ctx.depth = depth;
  ctx.writer = writer;
  ctx.perf = perf;
  ctx.enable_osd = cfg.output.enable_osd;

  zed_ds::PipelineHandles handles;
  std::string err;
  if (!zed_ds::BuildPipeline(cfg, ctx, &handles, &err)) {
    std::cerr << "[error] build pipeline failed: " << err << std::endl;
    return false;
  }

  GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
  GstBus* bus = gst_element_get_bus(handles.pipeline);
  gst_bus_add_watch(bus, OnBusMsg, loop);

  LoopState state;
  state.loop = loop;
  state.perf = perf;
  state.fallback_fps = cfg.perf.fallback_fps;
  state.allow_fallback = allow_fallback;

  g_timeout_add_seconds(cfg.output.perf_log_interval_sec, OnPerfTick, &state);

  gst_element_set_state(handles.pipeline, GST_STATE_PLAYING);
  g_active_loop = loop;
  if (!g_stop) {
    g_main_loop_run(loop);
  }
  g_active_loop = nullptr;

  if (g_stop) {
    g_main_loop_quit(loop);
  }

  gst_element_set_state(handles.pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(handles.depth_sink);
  gst_object_unref(handles.infer);
  gst_object_unref(handles.sink);
  gst_object_unref(handles.pipeline);
  g_main_loop_unref(loop);

  if (fallback_triggered) *fallback_triggered = state.fallback_triggered;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  zed_ds::CliOptions cli;
  if (!zed_ds::ParseCli(argc, argv, &cli)) return 1;

  zed_ds::AppConfig cfg;
  std::string err;
  if (!zed_ds::LoadYamlConfig(cli.config_path, &cfg, &err)) {
    std::cerr << "[error] failed to load config " << cli.config_path << ": " << err << std::endl;
    return 2;
  }

  zed_ds::ApplyModeOverride(cli.mode, &cfg);
  if (!cli.dump_json.empty()) cfg.output.json_path = cli.dump_json;
  if (cli.no_display) cfg.output.display = false;

  std::cout << "zed_ds_app v" << ZED_DS_APP_VERSION << std::endl;
  std::cout << "config=" << cli.config_path << " mode=" << cli.mode << " json=" << cfg.output.json_path
            << " display=" << (cfg.output.display ? "on" : "off") << std::endl;

  bool fallback_triggered = false;
  if (!RunOnce(cfg, true, &fallback_triggered)) return 3;

  if (fallback_triggered && !g_stop) {
    std::cout << "[info] restart with fallback profile" << std::endl;
    cfg.camera.fps = cfg.perf.fallback_fps;
    cfg.camera.resolution = "HD1080";
    if (!RunOnce(cfg, false, nullptr)) return 4;
  }

  return 0;
}
