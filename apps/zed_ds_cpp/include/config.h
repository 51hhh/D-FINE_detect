#pragma once

#include <string>

namespace zed_ds {

struct CameraConfig {
  int camera_id{0};
  std::string resolution{"HD720"};
  int fps{60};
  std::string stream_type{"left+depth"};
};

struct InferConfig {
  std::string config_file{"configs/infer/config_infer_primary_dfine_ball.txt"};
  std::string onnx_file{"model/best_stg1.onnx"};
  std::string engine_file{"model/model.engine"};
  std::string labels{"DeepStream-Yolo/labels.txt"};
  float threshold{0.25f};
};

struct DepthConfig {
  float roi_ratio{0.10f};
  float min_valid_ratio{0.20f};
  float min_depth{0.10f};
  float max_depth{20.0f};
};

struct PerfConfig {
  int target_fps{100};
  int fallback_fps{60};
  int latency_budget_ms{35};
  int window_sec{5};
};

struct OutputConfig {
  bool enable_osd{true};
  std::string json_path{"logs/frame_result.jsonl"};
  int perf_log_interval_sec{5};
  bool display{true};
};

struct AppConfig {
  CameraConfig camera;
  InferConfig infer;
  DepthConfig depth;
  PerfConfig perf;
  OutputConfig output;
};

struct CliOptions {
  std::string config_path{"configs/zed_dfine_fast.yaml"};
  std::string mode{"max_fps"};
  std::string dump_json;
  bool no_display{false};
};

bool ParseCli(int argc, char** argv, CliOptions* out);
bool LoadYamlConfig(const std::string& path, AppConfig* cfg, std::string* err);
void ApplyModeOverride(const std::string& mode, AppConfig* cfg);

}  // namespace zed_ds
