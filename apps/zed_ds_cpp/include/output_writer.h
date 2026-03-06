#pragma once

#include <chrono>
#include <fstream>
#include <mutex>
#include <string>

#include "types.h"

namespace zed_ds {

// JSONL 格式帧结果输出。
// 支持按时间间隔批量 flush，避免 120fps 下每帧 fsync。
class OutputWriter {
public:
  // flush_interval_sec: flush 间隔秒数，0 表示每帧刷盘。
  explicit OutputWriter(std::string path, int flush_interval_sec = 1);
  bool Open();
  void Write(const FrameResult &frame_result);

private:
  std::string path_;
  int flush_interval_sec_{1};
  std::ofstream out_;
  std::mutex mu_;
  std::chrono::steady_clock::time_point last_flush_;
};

} // namespace zed_ds
