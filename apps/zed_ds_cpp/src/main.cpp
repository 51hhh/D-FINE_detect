#include <gst/gst.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
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
GMainLoop *g_active_loop = nullptr;

void OnSignal(int) {
  g_stop = 1;
  // g_main_loop_quit 严格来说不是 async-signal-safe 的，
  // 但在 Linux + GLib 实际使用中是安全的。
  // 完美方案是 g_unix_signal_add，但需要 GMainContext 已创建。
  if (g_active_loop) {
    g_main_loop_quit(g_active_loop);
  }
}

// 性能统计定时输出回调。
struct LoopState {
  std::shared_ptr<zed_ds::PerfMonitor> perf;
  std::chrono::steady_clock::time_point start_tp;
};

gboolean OnPerfTick(gpointer user_data) {
  auto *st = static_cast<LoopState *>(user_data);
  if (!st || !st->perf)
    return G_SOURCE_CONTINUE;

  const double fps = st->perf->AvgFps();
  const size_t frames = st->perf->FrameCount();
  const uint64_t total_frames = st->perf->TotalFrameCount();
  const double p95 = st->perf->P95LatencyMs();
  const auto elapsed_sec = std::chrono::duration_cast<std::chrono::seconds>(
                               std::chrono::steady_clock::now() - st->start_tp)
                               .count();

  if (total_frames == 0) {
    std::cout << "[perf] 等待首帧中... elapsed_sec=" << elapsed_sec
              << std::endl;
    return G_SOURCE_CONTINUE;
  }

  std::cout << "[perf] fps=" << fps << " window_frames=" << frames
            << " total_frames=" << total_frames << " p95_latency_ms=" << p95
            << " elapsed_sec=" << elapsed_sec << std::endl;

  return G_SOURCE_CONTINUE;
}

// ── ZED X daemon 健康检查 ──
// ZED X 依赖 zed_x_daemon 服务；不同版本下进程名可能是 ZEDX_Daemon。
void CheckZedDaemonHealth() {
  const int rc_service =
      std::system("systemctl is-active --quiet zed_x_daemon >/dev/null 2>&1");
  const int rc_proc = std::system("pgrep -x ZEDX_Daemon >/dev/null 2>&1");
  if (rc_service != 0 && rc_proc != 0) {
    std::cerr << "[warn] 未检测到 zed_x_daemon，ZED X 可能无法稳定采集。"
              << std::endl;
    std::cerr << "[warn] 可尝试执行: sudo systemctl restart zed_x_daemon "
                 "或重启 ZED 服务后再运行。"
              << std::endl;
  }
}

// ── GStreamer bus 消息处理 ──
gboolean OnBusMsg(GstBus *, GstMessage *msg, gpointer user_data) {
  auto *loop = static_cast<GMainLoop *>(user_data);

  switch (GST_MESSAGE_TYPE(msg)) {
  case GST_MESSAGE_EOS:
    g_main_loop_quit(loop);
    break;
  case GST_MESSAGE_ERROR: {
    GError *err = nullptr;
    gchar *dbg = nullptr;
    gst_message_parse_error(msg, &err, &dbg);
    std::cerr << "[error] " << (err ? err->message : "unknown") << std::endl;
    if (dbg)
      std::cerr << "[debug] " << dbg << std::endl;
    if (err)
      g_error_free(err);
    if (dbg)
      g_free(dbg);
    g_main_loop_quit(loop);
    break;
  }
  default:
    break;
  }

  return G_SOURCE_CONTINUE;
}

// ── 单次运行 ──
bool RunOnce(const zed_ds::AppConfig &cfg) {
  auto depth =
      std::make_shared<zed_ds::DepthEstimator>(zed_ds::DepthEstimator::Config{
          .roi_ratio = cfg.depth.roi_ratio,
          .min_valid_ratio = cfg.depth.min_valid_ratio,
          .min_depth = cfg.depth.min_depth,
          .max_depth = cfg.depth.max_depth,
      });

  auto writer = std::make_shared<zed_ds::OutputWriter>(
      cfg.output.json_path, cfg.output.flush_interval_sec);
  if (!writer->Open()) {
    std::cerr << "[error] failed to open json output: " << cfg.output.json_path
              << std::endl;
    return false;
  }

  auto perf = std::make_shared<zed_ds::PerfMonitor>(cfg.perf.window_sec);

  // 运行时上下文：通过 shared_ptr 在所有回调间安全共享。
  auto ctx = std::make_shared<zed_ds::RuntimeContext>();
  ctx->depth = depth;
  ctx->writer = writer;
  ctx->perf = perf;
  ctx->enable_osd = cfg.output.enable_osd;
  // 直接持有 CameraConfig 指针，消除 CameraIntrinsics 重复。
  // 生命周期安全：cfg 在 RunOnce 返回前始终有效。
  ctx->camera = &cfg.camera;

  zed_ds::PipelineHandles handles;
  std::string err;
  if (!zed_ds::BuildPipeline(cfg, ctx, &handles, &err)) {
    std::cerr << "[error] build pipeline failed: " << err << std::endl;
    return false;
  }

  GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
  GstBus *bus = gst_element_get_bus(handles.pipeline);
  gst_bus_add_watch(bus, OnBusMsg, loop);

  LoopState state;
  state.perf = perf;
  state.start_tp = std::chrono::steady_clock::now();

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

  return true;
}

} // namespace

int main(int argc, char **argv) {
  gst_init(&argc, &argv);
  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);
  CheckZedDaemonHealth();

  zed_ds::CliOptions cli;
  if (!zed_ds::ParseCli(argc, argv, &cli))
    return 1;

  zed_ds::AppConfig cfg;
  std::string err;
  if (!zed_ds::LoadYamlConfig(cli.config_path, &cfg, &err)) {
    std::cerr << "[error] failed to load config " << cli.config_path << ": "
              << err << std::endl;
    return 2;
  }

  if (!cli.dump_json.empty())
    cfg.output.json_path = cli.dump_json;
  if (cli.no_display)
    cfg.output.display = false;

  std::cout << "zed_ds_app v" << ZED_DS_APP_VERSION << std::endl;
  std::cout << "config=" << cli.config_path << " json=" << cfg.output.json_path
            << " display=" << (cfg.output.display ? "on" : "off") << std::endl;
  std::cout << "camera_intrinsics fx=" << cfg.camera.fx
            << " fy=" << cfg.camera.fy << " cx=" << cfg.camera.cx
            << " cy=" << cfg.camera.cy;
  std::cout << " dist(k1,k2,p1,p2,k3)=(" << cfg.camera.k1 << ","
            << cfg.camera.k2 << "," << cfg.camera.p1 << "," << cfg.camera.p2
            << "," << cfg.camera.k3 << ")"
            << " undistort="
            << (cfg.camera.use_distortion_correction ? "on" : "off");
  if (!cfg.camera.calibration_file.empty()) {
    std::cout << " calib_file=" << cfg.camera.calibration_file;
  }
  std::cout << std::endl;

  if (!RunOnce(cfg))
    return 3;

  return 0;
}
