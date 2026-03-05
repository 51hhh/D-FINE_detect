#include "config.h"

#include <yaml-cpp/yaml.h>

#include <iostream>

namespace zed_ds {
namespace {

void SetIfPresent(const YAML::Node& n, const char* key, int* v) {
  if (n[key]) *v = n[key].as<int>();
}

void SetIfPresent(const YAML::Node& n, const char* key, float* v) {
  if (n[key]) *v = n[key].as<float>();
}

void SetIfPresent(const YAML::Node& n, const char* key, bool* v) {
  if (n[key]) *v = n[key].as<bool>();
}

void SetIfPresent(const YAML::Node& n, const char* key, std::string* v) {
  if (n[key]) *v = n[key].as<std::string>();
}

}  // namespace

bool ParseCli(int argc, char** argv, CliOptions* out) {
  if (!out) return false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      out->config_path = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      out->mode = argv[++i];
    } else if (arg == "--dump-json" && i + 1 < argc) {
      out->dump_json = argv[++i];
    } else if (arg == "--no-display") {
      out->no_display = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: zed_ds_app [--config PATH] [--mode max_fps|balanced|quality] [--dump-json PATH] [--no-display]\n";
      return false;
    }
  }
  return true;
}

bool LoadYamlConfig(const std::string& path, AppConfig* cfg, std::string* err) {
  if (!cfg) {
    if (err) *err = "cfg is null";
    return false;
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const std::exception& e) {
    if (err) *err = e.what();
    return false;
  }

  if (root["camera"]) {
    const auto c = root["camera"];
    SetIfPresent(c, "camera_id", &cfg->camera.camera_id);
    SetIfPresent(c, "resolution", &cfg->camera.resolution);
    SetIfPresent(c, "fps", &cfg->camera.fps);
    SetIfPresent(c, "stream_type", &cfg->camera.stream_type);
  }

  if (root["infer"]) {
    const auto i = root["infer"];
    SetIfPresent(i, "config_file", &cfg->infer.config_file);
    SetIfPresent(i, "onnx_file", &cfg->infer.onnx_file);
    SetIfPresent(i, "engine_file", &cfg->infer.engine_file);
    SetIfPresent(i, "labels", &cfg->infer.labels);
    SetIfPresent(i, "threshold", &cfg->infer.threshold);
  }

  if (root["depth"]) {
    const auto d = root["depth"];
    SetIfPresent(d, "roi_ratio", &cfg->depth.roi_ratio);
    SetIfPresent(d, "min_valid_ratio", &cfg->depth.min_valid_ratio);
    SetIfPresent(d, "min_depth", &cfg->depth.min_depth);
    SetIfPresent(d, "max_depth", &cfg->depth.max_depth);
  }

  if (root["perf"]) {
    const auto p = root["perf"];
    SetIfPresent(p, "target_fps", &cfg->perf.target_fps);
    SetIfPresent(p, "fallback_fps", &cfg->perf.fallback_fps);
    SetIfPresent(p, "latency_budget_ms", &cfg->perf.latency_budget_ms);
    SetIfPresent(p, "window_sec", &cfg->perf.window_sec);
  }

  if (root["output"]) {
    const auto o = root["output"];
    SetIfPresent(o, "enable_osd", &cfg->output.enable_osd);
    SetIfPresent(o, "json_path", &cfg->output.json_path);
    SetIfPresent(o, "perf_log_interval_sec", &cfg->output.perf_log_interval_sec);
    SetIfPresent(o, "display", &cfg->output.display);
  }

  return true;
}

void ApplyModeOverride(const std::string& mode, AppConfig* cfg) {
  if (!cfg) return;

  if (mode == "max_fps") {
    cfg->camera.fps = cfg->perf.target_fps;
    if (cfg->camera.fps >= 100) {
      cfg->camera.resolution = "SVGA";
    }
  } else if (mode == "balanced") {
    cfg->camera.fps = 60;
    cfg->camera.resolution = "HD1080";
  } else if (mode == "quality") {
    cfg->camera.fps = 30;
    cfg->camera.resolution = "HD1080";
    cfg->infer.threshold = std::max(cfg->infer.threshold, 0.35f);
  }
}

}  // namespace zed_ds
