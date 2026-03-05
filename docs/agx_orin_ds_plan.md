# AGX Orin DeepStream-ZED 项目实施文档

## 1. 项目目标
- 平台：Jetson AGX Orin（JetPack 5.1.2）
- 方案：DeepStream 6.3 + ZED X（`zedsrc/zeddemux`）+ D-FINE
- 一期重点：单路高帧率低时延，先跑通 C++/CUDA 推理链路
- 深度规则：检测框中心小区域均值（用于排球凸面深度估计）

## 2. 目录与模块
- `apps/zed_ds_cpp/`：主程序与核心逻辑
- `configs/`：运行配置与 nvinfer 配置
- `model/`：ONNX/engine 模型文件
- `DeepStream-Yolo/`：参考仓库与自定义实现

## 3. 管线设计
主链路：
`zedsrc -> zeddemux -> (left) nvvideoconvert -> nvstreammux -> nvinfer -> nvdsosd -> sink`

深度支路：
`zeddemux.src_aux -> appsink -> DepthEstimator`

## 4. 关键接口
### CLI
- `--config <path>`
- `--mode <max_fps|balanced|quality>`
- `--dump-json <path>`
- `--no-display`

### 数据结构
- `Detection2D {bbox, class_id, conf, label}`
- `DepthEstimate {depth_m, valid_ratio, valid}`
- `FrameResult {ts_ns, frame_id, detections[]}`

## 5. D-FINE 适配说明
当前 D-FINE engine 为多输入网络（`images` + `orig_target_sizes`），因此新增自定义库：
- `NvDsInferInitializeInputLayers`：填充 `orig_target_sizes`
- `NvDsInferParseDFINE`：解析 `scores/labels/boxes`

对应文件：
- `apps/zed_ds_cpp/plugins/nvdsinfer_custom_dfine.cpp`
- `configs/infer/config_infer_primary_dfine_ball.txt`

## 6. 深度估计规则
对每个检测框：
1. 取中心 ROI（默认框宽高的 10%）
2. 过滤无效深度（0、NaN、超量程）
3. 计算有效像素均值
4. 若有效比例低于阈值，标记为 `invalid`

## 7. 性能策略
- `max_fps`：优先冲高帧率
- `balanced`：稳定优先
- `quality`：精度优先
- 运行时监控 FPS 与 P95 latency，低于阈值触发一次降档重启

## 8. 输出与验收
输出：
- OSD 叠加（可关闭）
- JSONL 结果（检测 + 深度 + 时间戳）

验收重点：
- 管线稳定运行
- 目标框与深度结果正确
- 性能模式可切换并能自动降档

## 9. 运行注意事项
`zedsrc` 在当前系统上需要：
- `LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`
- `GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0`

否则可能出现插件扫描失败。
