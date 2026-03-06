#include "config.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

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

std::string Trim(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

std::string SectionFromResolution(const std::string& resolution) {
  if (resolution == "SVGA") return "LEFT_CAM_SVGA";
  if (resolution == "VGA") return "LEFT_CAM_VGA";
  if (resolution == "HD1080") return "LEFT_CAM_FHD";
  if (resolution == "HD1200") return "LEFT_CAM_FHD1200";
  if (resolution == "HD2K") return "LEFT_CAM_2K";
  return "LEFT_CAM_SVGA";
}

std::string AutoFindFactoryCalib() {
  const std::filesystem::path dir("/usr/local/zed/settings");
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) return "";
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) continue;
    const auto name = entry.path().filename().string();
    if (name.rfind("SN", 0) == 0 && entry.path().extension() == ".conf") {
      return entry.path().string();
    }
  }
  return "";
}

bool LoadFactoryIntrinsics(const std::string& conf_file, const std::string& resolution, CameraConfig* cam, std::string* err) {
  if (!cam) return false;
  std::ifstream in(conf_file);
  if (!in.is_open()) {
    if (err) *err = "无法打开标定文件: " + conf_file;
    return false;
  }

  const std::string target_section = SectionFromResolution(resolution);
  std::string line;
  std::string section;
  bool found_fx = false;
  bool found_fy = false;
  bool found_cx = false;
  bool found_cy = false;
  bool found_k1 = false;
  bool found_k2 = false;
  bool found_p1 = false;
  bool found_p2 = false;
  bool found_k3 = false;

  while (std::getline(in, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#' || line[0] == ';') continue;
    if (line.front() == '[' && line.back() == ']') {
      section = line.substr(1, line.size() - 2);
      continue;
    }
    if (section != target_section) continue;

    const auto pos = line.find('=');
    if (pos == std::string::npos) continue;
    const std::string key = Trim(line.substr(0, pos));
    const std::string value = Trim(line.substr(pos + 1));
    if (value.empty()) continue;

    try {
      const float v = std::stof(value);
      if (key == "fx") {
        cam->fx = v;
        found_fx = true;
      } else if (key == "fy") {
        cam->fy = v;
        found_fy = true;
      } else if (key == "cx") {
        cam->cx = v;
        found_cx = true;
      } else if (key == "cy") {
        cam->cy = v;
        found_cy = true;
      } else if (key == "k1") {
        cam->k1 = v;
        found_k1 = true;
      } else if (key == "k2") {
        cam->k2 = v;
        found_k2 = true;
      } else if (key == "p1") {
        cam->p1 = v;
        found_p1 = true;
      } else if (key == "p2") {
        cam->p2 = v;
        found_p2 = true;
      } else if (key == "k3") {
        cam->k3 = v;
        found_k3 = true;
      }
    } catch (...) {
      continue;
    }
  }

  if (!(found_fx && found_fy && found_cx && found_cy)) {
    if (err) *err = "标定文件缺少目标分辨率内参段: [" + target_section + "]";
    return false;
  }
  if (!(found_k1 && found_k2 && found_p1 && found_p2 && found_k3)) {
    std::cerr << "[warn] 标定文件未完整提供畸变参数，使用默认值/已有值。" << std::endl;
  }
  return true;
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
    SetIfPresent(c, "use_factory_calib", &cfg->camera.use_factory_calib);
    SetIfPresent(c, "use_distortion_correction", &cfg->camera.use_distortion_correction);
    SetIfPresent(c, "calibration_file", &cfg->camera.calibration_file);
    SetIfPresent(c, "fx", &cfg->camera.fx);
    SetIfPresent(c, "fy", &cfg->camera.fy);
    SetIfPresent(c, "cx", &cfg->camera.cx);
    SetIfPresent(c, "cy", &cfg->camera.cy);
    SetIfPresent(c, "k1", &cfg->camera.k1);
    SetIfPresent(c, "k2", &cfg->camera.k2);
    SetIfPresent(c, "p1", &cfg->camera.p1);
    SetIfPresent(c, "p2", &cfg->camera.p2);
    SetIfPresent(c, "k3", &cfg->camera.k3);
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
    SetIfPresent(p, "warmup_sec", &cfg->perf.warmup_sec);
    SetIfPresent(p, "enable_auto_fallback", &cfg->perf.enable_auto_fallback);
  }

  if (root["output"]) {
    const auto o = root["output"];
    SetIfPresent(o, "enable_osd", &cfg->output.enable_osd);
    SetIfPresent(o, "json_path", &cfg->output.json_path);
    SetIfPresent(o, "perf_log_interval_sec", &cfg->output.perf_log_interval_sec);
    SetIfPresent(o, "display", &cfg->output.display);
  }

  if (cfg->camera.use_factory_calib) {
    std::string calib_file = cfg->camera.calibration_file;
    if (calib_file.empty()) calib_file = AutoFindFactoryCalib();
    if (!calib_file.empty()) {
      std::string calib_err;
      if (LoadFactoryIntrinsics(calib_file, cfg->camera.resolution, &cfg->camera, &calib_err)) {
        cfg->camera.calibration_file = calib_file;
      } else {
        std::cerr << "[warn] 加载出厂标定失败: " << calib_err << std::endl;
      }
    } else {
      std::cerr << "[warn] 未找到 ZED 出厂标定文件，继续使用配置内参。" << std::endl;
    }
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
