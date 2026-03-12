#include "ball_tracker.h"

#include <cmath>
#include <limits>

namespace zed_ds {

BallTracker::BallTracker(Config cfg) : cfg_(cfg) {}

std::optional<BallTracker::TrackState>
BallTracker::Update(const std::vector<DetectionResult> &dets) {
  // 筛选有效 3D 检测。
  const DetectionResult *best = nullptr;
  float best_dist = std::numeric_limits<float>::max();

  for (const auto &d : dets) {
    if (!d.depth.valid)
      continue;

    if (!track_) {
      // 无活跃 track，取第一个有效检测创建新 track。
      best = &d;
      best_dist = 0.0f;
      break;
    }

    // 计算 3D 欧氏距离。
    const float dx = d.depth.x_m - track_->x;
    const float dy = d.depth.y_m - track_->y;
    const float dz = d.depth.z_m - track_->z;
    const float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    if (dist < best_dist) {
      best_dist = dist;
      best = &d;
    }
  }

  if (best && (best_dist <= cfg_.gate_distance_m || !track_)) {
    // 关联成功 或 创建新 track。
    if (!track_) {
      // 创建新 track。
      TrackState ts;
      ts.track_id = next_id_++;
      ts.x = best->depth.x_m;
      ts.y = best->depth.y_m;
      ts.z = best->depth.z_m;
      ts.coast_count = 0;
      ts.hit_count = 1;
      ts.confirmed = (ts.hit_count >= cfg_.confirm_hits);
      track_ = ts;
    } else {
      // 更新现有 track。
      track_->x = best->depth.x_m;
      track_->y = best->depth.y_m;
      track_->z = best->depth.z_m;
      track_->coast_count = 0;
      track_->hit_count++;
      if (track_->hit_count >= cfg_.confirm_hits)
        track_->confirmed = true;
    }
  } else {
    // 丢检。
    if (track_) {
      track_->coast_count++;
      if (track_->coast_count > cfg_.max_coast_frames) {
        track_.reset(); // Drop track。
      }
    }
  }

  return track_;
}

std::optional<BallTracker::TrackState> BallTracker::GetTrack() const {
  return track_;
}

void BallTracker::Reset() {
  track_.reset();
}

} // namespace zed_ds
