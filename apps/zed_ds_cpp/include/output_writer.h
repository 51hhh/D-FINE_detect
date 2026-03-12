#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>

#include "types.h"

namespace zed_ds {

// JSONL 格式帧结果输出。
// 支持按时间间隔批量 flush，避免 120fps 下每帧 fsync。
// 支持日志轮转：超过 max_file_bytes 时自动重命名旧文件并创建新文件。
class OutputWriter {
public:
  // flush_interval_sec: flush 间隔秒数，0 表示每帧刷盘。
  // max_file_bytes: 单文件最大字节数，0 表示不轮转。默认 100MB。
  explicit OutputWriter(std::string path, int flush_interval_sec = 1,
                        int64_t max_file_bytes = 100 * 1024 * 1024);
  bool Open();
  void Write(const FrameResult &frame_result);

private:
  void RotateIfNeeded();

  std::string path_;
  int flush_interval_sec_{1};
  int64_t max_file_bytes_{100 * 1024 * 1024};
  int64_t current_bytes_{0};
  int rotation_index_{0};
  std::ofstream out_;
  std::mutex mu_;
  std::chrono::steady_clock::time_point last_flush_;
};

} // namespace zed_ds
