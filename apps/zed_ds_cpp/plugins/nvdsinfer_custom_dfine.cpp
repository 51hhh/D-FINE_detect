#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" {

// D-FINE 引擎包含非图像输入 orig_target_sizes，这里按 batch 填充 [H, W]。
bool NvDsInferInitializeInputLayers(std::vector<NvDsInferLayerInfo> const& inputLayersInfo,
                                    NvDsInferNetworkInfo const& networkInfo,
                                    unsigned int maxBatchSize) {
  for (const auto& layer : inputLayersInfo) {
    if (!layer.buffer || !layer.layerName) {
      continue;
    }

    const std::string name(layer.layerName);
    if (name.find("orig_target_sizes") != std::string::npos) {
      int32_t* ptr = static_cast<int32_t*>(layer.buffer);
      if (!ptr) {
        return false;
      }
      for (unsigned int b = 0; b < maxBatchSize; ++b) {
        ptr[b * 2 + 0] = static_cast<int32_t>(networkInfo.height);
        ptr[b * 2 + 1] = static_cast<int32_t>(networkInfo.width);
      }
    }
  }

  return true;
}

// 解析 D-FINE 输出: scores[300], labels[300], boxes[300x4]。
bool NvDsInferParseDFINE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                         NvDsInferNetworkInfo const& networkInfo,
                         NvDsInferParseDetectionParams const& detectionParams,
                         std::vector<NvDsInferObjectDetectionInfo>& objectList) {
  const float* scores = nullptr;
  const int32_t* labels = nullptr;
  const float* boxes = nullptr;
  int score_count = 0;
  int box_count = 0;

  for (const auto& layer : outputLayersInfo) {
    if (!layer.layerName || !layer.buffer) {
      continue;
    }
    const std::string name(layer.layerName);

    if (name == "scores") {
      scores = static_cast<const float*>(layer.buffer);
      score_count = layer.inferDims.numElements;
    } else if (name == "labels") {
      labels = static_cast<const int32_t*>(layer.buffer);
    } else if (name == "boxes") {
      boxes = static_cast<const float*>(layer.buffer);
      box_count = layer.inferDims.numElements / 4;
    }
  }

  if (!scores || !labels || !boxes || score_count <= 0 || box_count <= 0) {
    return false;
  }

  const int n = std::min(score_count, box_count);
  const bool force_single_class = (detectionParams.numClassesConfigured == 1);
  for (int i = 0; i < n; ++i) {
    const float score = scores[i];
    const int class_id = force_single_class ? 0 : labels[i];

    if (class_id < 0 || class_id >= detectionParams.numClassesConfigured) {
      continue;
    }

    const float threshold = detectionParams.perClassPreclusterThreshold[class_id];
    if (score < threshold) {
      continue;
    }

    const float x1 = boxes[i * 4 + 0];
    const float y1 = boxes[i * 4 + 1];
    const float x2 = boxes[i * 4 + 2];
    const float y2 = boxes[i * 4 + 3];

    NvDsInferObjectDetectionInfo obj;
    obj.classId = class_id;
    obj.detectionConfidence = score;

    // 兼容归一化和像素坐标两种导出形式。
    const bool normalized = (x2 <= 1.5f && y2 <= 1.5f);
    const float sx = normalized ? static_cast<float>(networkInfo.width) : 1.0f;
    const float sy = normalized ? static_cast<float>(networkInfo.height) : 1.0f;

    const float l = x1 * sx;
    const float t = y1 * sy;
    const float r = x2 * sx;
    const float b = y2 * sy;

    obj.left = std::max(0.0f, l);
    obj.top = std::max(0.0f, t);
    obj.width = std::max(0.0f, r - l);
    obj.height = std::max(0.0f, b - t);

    if (obj.width <= 1.0f || obj.height <= 1.0f) {
      continue;
    }

    objectList.push_back(obj);
  }

  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseDFINE)

}  // extern "C"
