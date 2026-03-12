#include "config.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace zed_ds {
namespace {

// ── YAML 辅助：仅当 key 存在时设置值 ──

void SetIfPresent(const YAML::Node &n, const char *key, int *v) {
  if (n[key])
    *v = n[key].as<int>();
}

void SetIfPresent(const YAML::Node &n, const char *key, float *v) {
  if (n[key])
    *v = n[key].as<float>();
}

void SetIfPresent(const YAML::Node &n, const char *key, bool *v) {
  if (n[key])
    *v = n[key].as<bool>();
}

void SetIfPresent(const YAML::Node &n, const char *key, std::string *v) {
  if (n[key])
    *v = n[key].as<std::string>();
}

std::string Trim(const std::string &s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return "";
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

// ── 标定文件分辨率 → INI 段名映射 ──
// 参照 ZED 出厂标定文件实际段名，只映射存在的段。
std::string SectionFromResolution(const std::string &resolution) {
  if (resolution == "SVGA")
    return "LEFT_CAM_SVGA";
  if (resolution == "HD1080")
    return "LEFT_CAM_FHD";
  if (resolution == "HD1200")
    return "LEFT_CAM_FHD1200";
  // HD720 和 HD2K 在 ZED X 出厂标定中通常不存在独立段，
  // 返回空字符串，由调用方处理。
  return "";
}

// 自动在 /usr/local/zed/settings/ 下查找 SN*.conf。
std::string AutoFindFactoryCalib() {
  const std::filesystem::path dir("/usr/local/zed/settings");
  if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir))
    return "";
  for (const auto &entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file())
      continue;
    const auto name = entry.path().filename().string();
    if (name.rfind("SN", 0) == 0 && entry.path().extension() == ".conf") {
      return entry.path().string();
    }
  }
  return "";
}

// ── 从出厂标定文件读取内参与畸变参数 ──
bool LoadFactoryIntrinsics(const std::string &conf_file,
                           const std::string &resolution, CameraConfig *cam,
                           std::string *err) {
  if (!cam)
    return false;

  const std::string target_section = SectionFromResolution(resolution);
  if (target_section.empty()) {
    if (err) {
      *err = "分辨率 " + resolution +
             " 在 ZED X 出厂标定文件中无对应段，"
             "请使用 SVGA/HD1080/HD1200 或在 YAML 中手动配置内参。";
    }
    return false;
  }

  std::ifstream in(conf_file);
  if (!in.is_open()) {
    if (err)
      *err = "无法打开标定文件: " + conf_file;
    return false;
  }

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
    if (line.empty() || line[0] == '#' || line[0] == ';')
      continue;
    if (line.front() == '[' && line.back() == ']') {
      section = line.substr(1, line.size() - 2);
      continue;
    }
    if (section != target_section)
      continue;

    const auto pos = line.find('=');
    if (pos == std::string::npos)
      continue;
    const std::string key = Trim(line.substr(0, pos));
    const std::string value = Trim(line.substr(pos + 1));
    if (value.empty())
      continue;

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
    if (err)
      *err = "标定文件缺少目标分辨率内参段: [" + target_section + "]";
    return false;
  }
  if (!(found_k1 && found_k2 && found_p1 && found_p2 && found_k3)) {
    std::cerr << "[warn] 标定文件未完整提供畸变参数，使用默认值/已有值。"
              << std::endl;
  }
  return true;
}

// ── 运行时 nvinfer 配置生成 ──
// 从模板文件读取、替换阈值，写入 /tmp 避免污染源码树。
// 同时将相对路径转为绝对路径，消除 CWD 依赖。
bool WriteRuntimeInferConfig(AppConfig *cfg, std::string *err) {
  if (!cfg)
    return false;
  const std::filesystem::path src(cfg->infer.config_file);

  // 解析模板文件所在目录，用于后续相对路径转换。
  std::filesystem::path src_abs = std::filesystem::absolute(src);
  std::filesystem::path src_dir = src_abs.parent_path();

  std::ifstream in(src_abs);
  if (!in.is_open()) {
    if (err)
      *err = "无法打开推理配置模板: " + src_abs.string();
    return false;
  }

  // 写到 /tmp 下，避免污染源码树、多实例互不干扰。
  const std::filesystem::path tmp_dir("/tmp/zed_ds_runtime");
  std::filesystem::create_directories(tmp_dir);
  const std::filesystem::path dst =
      tmp_dir / ("runtime_" + src.filename().string());

  std::ofstream out(dst);
  if (!out.is_open()) {
    if (err)
      *err = "无法写入运行时推理配置: " + dst.string();
    return false;
  }

  std::ostringstream th;
  th.setf(std::ios::fixed);
  th << std::setprecision(3) << cfg->infer.threshold;

  // 需要转为绝对路径的 key 列表。
  const std::vector<std::string> path_keys = {
      "onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path"};

  std::string line;
  bool replaced_threshold = false;
  while (std::getline(in, line)) {
    const std::string trimmed = Trim(line);

    // 替换阈值，确保只由 YAML 单点控制。
    if (trimmed.rfind("pre-cluster-threshold=", 0) == 0) {
      out << "pre-cluster-threshold=" << th.str() << "\n";
      replaced_threshold = true;
      continue;
    }

    // 将相对路径转为绝对路径，消除 CWD 依赖。
    bool handled = false;
    for (const auto &pk : path_keys) {
      const std::string prefix = pk + "=";
      if (trimmed.rfind(prefix, 0) == 0) {
        std::string val = trimmed.substr(prefix.size());
        std::filesystem::path p(val);
        if (p.is_relative()) {
          p = std::filesystem::weakly_canonical(src_dir / p);
        }
        out << prefix << p.string() << "\n";
        handled = true;
        break;
      }
    }
    if (!handled) {
      out << line << "\n";
    }
  }

  if (!replaced_threshold) {
    out << "\n[class-attrs-all]\n";
    out << "pre-cluster-threshold=" << th.str() << "\n";
  }

  cfg->infer.config_file = dst.string();
  return true;
}

} // namespace

// ── CLI 解析 ──
bool ParseCli(int argc, char **argv, CliOptions *out) {
  if (!out)
    return false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      out->config_path = argv[++i];
    } else if (arg == "--dump-json" && i + 1 < argc) {
      out->dump_json = argv[++i];
    } else if (arg == "--no-display") {
      out->no_display = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: zed_ds_app [--config PATH] [--dump-json PATH] "
                   "[--no-display]\n";
      std::cout << "  --config PATH     YAML config path\n";
      std::cout << "  --dump-json PATH  Override JSONL output path\n";
      std::cout << "  --no-display      Disable on-screen display sink\n";
      return false;
    }
  }
  return true;
}

// ── YAML 配置加载 ──
bool LoadYamlConfig(const std::string &path, AppConfig *cfg, std::string *err) {
  if (!cfg) {
    if (err)
      *err = "cfg is null";
    return false;
  }

  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const std::exception &e) {
    if (err)
      *err = e.what();
    return false;
  }

  if (root["camera"]) {
    const auto c = root["camera"];
    SetIfPresent(c, "camera_id", &cfg->camera.camera_id);
    SetIfPresent(c, "resolution", &cfg->camera.resolution);
    SetIfPresent(c, "fps", &cfg->camera.fps);
    SetIfPresent(c, "stream_type", &cfg->camera.stream_type);
    SetIfPresent(c, "use_factory_calib", &cfg->camera.use_factory_calib);
    SetIfPresent(c, "use_distortion_correction",
                 &cfg->camera.use_distortion_correction);
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
    SetIfPresent(p, "latency_budget_ms", &cfg->perf.latency_budget_ms);
    SetIfPresent(p, "window_sec", &cfg->perf.window_sec);
    SetIfPresent(p, "warmup_sec", &cfg->perf.warmup_sec);
  }

  if (root["output"]) {
    const auto o = root["output"];
    SetIfPresent(o, "enable_osd", &cfg->output.enable_osd);
    SetIfPresent(o, "json_path", &cfg->output.json_path);
    SetIfPresent(o, "perf_log_interval_sec",
                 &cfg->output.perf_log_interval_sec);
    SetIfPresent(o, "flush_interval_sec", &cfg->output.flush_interval_sec);
    SetIfPresent(o, "display", &cfg->output.display);
  }

  if (root["tracker"]) {
    const auto t = root["tracker"];
    SetIfPresent(t, "gate_distance_m", &cfg->tracker.gate_distance_m);
    SetIfPresent(t, "max_coast_frames", &cfg->tracker.max_coast_frames);
    SetIfPresent(t, "confirm_hits", &cfg->tracker.confirm_hits);
  }

  if (root["trajectory"]) {
    const auto t = root["trajectory"];
    SetIfPresent(t, "gravity", &cfg->trajectory.gravity);
    SetIfPresent(t, "process_noise_pos", &cfg->trajectory.process_noise_pos);
    SetIfPresent(t, "process_noise_vel", &cfg->trajectory.process_noise_vel);
    SetIfPresent(t, "measure_noise_xy", &cfg->trajectory.measure_noise_xy);
    SetIfPresent(t, "measure_noise_z", &cfg->trajectory.measure_noise_z);
    SetIfPresent(t, "cov_trace_reset", &cfg->trajectory.cov_trace_reset);
    SetIfPresent(t, "innovation_gate_sigma", &cfg->trajectory.innovation_gate_sigma);
    SetIfPresent(t, "max_no_update_frames", &cfg->trajectory.max_no_update_frames);
  }

  if (root["landing"]) {
    const auto l = root["landing"];
    SetIfPresent(l, "court_z", &cfg->landing.court_z);
    SetIfPresent(l, "max_flight_time", &cfg->landing.max_flight_time);
    SetIfPresent(l, "min_confidence", &cfg->landing.min_confidence);
  }

  // 加载出厂标定内参。
  if (cfg->camera.use_factory_calib) {
    std::string calib_file = cfg->camera.calibration_file;
    if (calib_file.empty())
      calib_file = AutoFindFactoryCalib();
    if (!calib_file.empty()) {
      std::string calib_err;
      if (LoadFactoryIntrinsics(calib_file, cfg->camera.resolution,
                                &cfg->camera, &calib_err)) {
        cfg->camera.calibration_file = calib_file;
      } else {
        std::cerr << "[warn] 加载出厂标定失败: " << calib_err << std::endl;
      }
    } else {
      std::cerr << "[warn] 未找到 ZED 出厂标定文件，继续使用配置内参。"
                << std::endl;
    }
  }

  // 生成运行时 nvinfer 配置。
  std::string runtime_infer_err;
  if (!WriteRuntimeInferConfig(cfg, &runtime_infer_err)) {
    if (err)
      *err = runtime_infer_err;
    return false;
  }

  return true;
}

} // namespace zed_ds
