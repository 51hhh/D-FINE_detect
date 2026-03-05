#pragma once

#include <fstream>
#include <mutex>
#include <string>

#include "types.h"

namespace zed_ds {

class OutputWriter {
 public:
  explicit OutputWriter(std::string path);
  bool Open();
  void Write(const FrameResult& frame_result);

 private:
  std::string path_;
  std::ofstream out_;
  std::mutex mu_;
};

}  // namespace zed_ds
