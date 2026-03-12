#pragma once

#include <string>

namespace zed_ds {

// ── 相机配置 ──
// 包含分辨率、帧率、内参、畸变参数等，消除原 CameraIntrinsics 的重复定义。
struct CameraConfig {
  int camera_id{0};
  std::string resolution{"HD720"};
  int fps{60};
  std::string stream_type{"left+depth"};

  bool use_factory_calib{true};
  bool use_distortion_correction{true};
  std::string calibration_file;

  // 相机内参（像素单位），用于由 (u,v,Z) 解算 (X,Y,Z)。
  float fx{700.0f};
  float fy{700.0f};
  float cx{480.0f};
  float cy{300.0f};

  // Brown-Conrady 畸变参数。
  float k1{0.0f};
  float k2{0.0f};
  float p1{0.0f};
  float p2{0.0f};
  float k3{0.0f};
};

// ── 推理配置 ──
struct InferConfig {
  // DeepStream nvinfer 模板配置文件路径（必填）。
  std::string config_file{"configs/infer/config_infer_primary_dfine_ball.txt"};
  // 兼容字段：当前由 config_file 模板中的对应 key 决定，暂未在代码中直接消费。
  std::string onnx_file{"model/best_stg2.onnx"};
  std::string engine_file{"model/best_stg2.engine"};
  std::string labels{"configs/labels_ball.txt"};
  float threshold{0.75f};
};

// ── 深度估计配置 ──
struct DepthConfig {
  float roi_ratio{0.10f};
  float min_valid_ratio{0.20f};
  float min_depth{0.10f};
  float max_depth{20.0f};
};

// ── 性能配置 ──
struct PerfConfig {
  // 兼容字段：当前仅用于配置记录，未参与运行时控制。
  int target_fps{100};
  int latency_budget_ms{35};
  int window_sec{5};
  int warmup_sec{8};
};

// ── 输出配置 ──
struct OutputConfig {
  bool enable_osd{true};
  std::string json_path{"logs/frame_result.jsonl"};
  int perf_log_interval_sec{5};
  // flush 间隔（秒），0 表示每帧刷盘。默认 1 秒批量 flush。
  int flush_interval_sec{1};
  bool display{true};
};

// ── 追踪配置 ──
struct TrackerConfig {
  float gate_distance_m{2.0f};
  int max_coast_frames{15};
  int confirm_hits{3};
};

// ── 轨迹 EKF 配置 ──
struct TrajectoryConfig {
  float gravity{9.81f};
  float process_noise_pos{0.05f};
  float process_noise_vel{2.0f};
  float measure_noise_xy{0.05f};
  float measure_noise_z{0.01f};
  float cov_trace_reset{100.0f};
  float innovation_gate_sigma{5.0f};
  int max_no_update_frames{20};
};

// ── 落点预测配置 ──
struct LandingConfig {
  float court_z{0.0f};
  float max_flight_time{5.0f};
  float min_confidence{0.1f};
};

// ── 总配置 ──
struct AppConfig {
  CameraConfig camera;
  InferConfig infer;
  DepthConfig depth;
  PerfConfig perf;
  OutputConfig output;
  TrackerConfig tracker;
  TrajectoryConfig trajectory;
  LandingConfig landing;
};

// ── CLI 选项 ──
struct CliOptions {
  std::string config_path{"configs/zed_dfine.yaml"};
  std::string dump_json;
  bool no_display{false};
};

bool ParseCli(int argc, char **argv, CliOptions *out);
bool LoadYamlConfig(const std::string &path, AppConfig *cfg, std::string *err);

} // namespace zed_ds
