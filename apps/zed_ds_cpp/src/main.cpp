#include <gst/gst.h>
#include <glib-unix.h>

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>

#include "config.h"
#include "ball_tracker.h"
#include "depth_estimator.h"
#include "landing_predictor.h"
#include "output_writer.h"
#include "perf_monitor.h"
#include "pipeline_builder.h"
#include "trajectory_estimator.h"
#include "version.h"

namespace {

// 使用 g_unix_signal_add 安全地处理信号（在 GMainLoop 轮询中回调，async-signal-safe）。
gboolean OnUnixSignal(gpointer user_data) {
  auto *loop = static_cast<GMainLoop *>(user_data);
  std::cerr << "[info] 收到终止信号，正在安全退出..." << std::endl;
  if (loop)
    g_main_loop_quit(loop);
  return G_SOURCE_REMOVE;
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
            << " total_frames=" << total_frames
            << " p95_frame_interval_ms=" << p95
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

  // 追踪 + 轨迹 + 落点预测组件。
  auto tracker = std::make_shared<zed_ds::BallTracker>(
      zed_ds::BallTracker::Config{
          .gate_distance_m = cfg.tracker.gate_distance_m,
          .max_coast_frames = cfg.tracker.max_coast_frames,
          .confirm_hits = cfg.tracker.confirm_hits,
      });

  auto trajectory = std::make_shared<zed_ds::TrajectoryEstimator>(
      zed_ds::TrajectoryEstimator::Config{
          .gravity = cfg.trajectory.gravity,
          .process_noise_pos = cfg.trajectory.process_noise_pos,
          .process_noise_vel = cfg.trajectory.process_noise_vel,
          .measure_noise_xy = cfg.trajectory.measure_noise_xy,
          .measure_noise_z = cfg.trajectory.measure_noise_z,
          .dt_clamp_min = 0.0001f,
          .dt_clamp_max = 0.1f,
          .cov_trace_reset = cfg.trajectory.cov_trace_reset,
          .innovation_gate_sigma = cfg.trajectory.innovation_gate_sigma,
          .max_no_update_frames = cfg.trajectory.max_no_update_frames,
      });

  auto predictor = std::make_shared<zed_ds::LandingPredictor>(
      zed_ds::LandingPredictor::Config{
          .court_z = cfg.landing.court_z,
          .gravity = cfg.trajectory.gravity,
          .max_flight_time = cfg.landing.max_flight_time,
          .min_confidence = cfg.landing.min_confidence,
      });

  // 运行时上下文：通过 shared_ptr 在所有回调间安全共享。
  auto ctx = std::make_shared<zed_ds::RuntimeContext>();
  ctx->depth = depth;
  ctx->writer = writer;
  ctx->perf = perf;
  ctx->tracker = tracker;
  ctx->trajectory = trajectory;
  ctx->predictor = predictor;
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
  guint bus_watch_id = gst_bus_add_watch(bus, OnBusMsg, loop);

  // 注册 Unix 信号处理（在 GMainContext 中安全回调）。
  guint sig_int_id = g_unix_signal_add(SIGINT, OnUnixSignal, loop);
  guint sig_term_id = g_unix_signal_add(SIGTERM, OnUnixSignal, loop);

  LoopState state;
  state.perf = perf;
  state.start_tp = std::chrono::steady_clock::now();

  guint perf_timer_id = g_timeout_add_seconds(
      cfg.output.perf_log_interval_sec, OnPerfTick, &state);

  gst_element_set_state(handles.pipeline, GST_STATE_PLAYING);
  g_main_loop_run(loop);

  // 移除所有 GSource，避免悬空回调（尤其 LoopState 是栈变量）。
  g_source_remove(perf_timer_id);
  g_source_remove(bus_watch_id);
  g_source_remove(sig_int_id);
  g_source_remove(sig_term_id);

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
