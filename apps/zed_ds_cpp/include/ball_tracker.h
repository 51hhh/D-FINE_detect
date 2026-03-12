#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "types.h"

namespace zed_ds {

/// 单球追踪器：最近邻 3D 欧氏距离关联，coast / confirm / drop 状态机。
class BallTracker {
public:
  struct Config {
    float gate_distance_m{2.0f}; // 关联门限（米）
    int max_coast_frames{15};    // 最大连续丢检帧数
    int confirm_hits{3};         // 确认 track 所需连续命中
  };

  struct TrackState {
    int64_t track_id{-1};
    float x{0}, y{0}, z{0}; // 最近 3D 位置（米，相机坐标系）
    int coast_count{0};
    int hit_count{0};
    bool confirmed{false};
  };

  explicit BallTracker(Config cfg);

  /// 每帧调用。传入本帧所有检测结果（depth.valid 必须为 true 才参与关联）。
  /// 返回当前活跃 track（单球场景最多 1 个，可为 nullopt）。
  std::optional<TrackState> Update(const std::vector<DetectionResult> &dets);

  std::optional<TrackState> GetTrack() const;
  void Reset();

private:
  Config cfg_;
  std::optional<TrackState> track_;
  int64_t next_id_{1};
};

} // namespace zed_ds
