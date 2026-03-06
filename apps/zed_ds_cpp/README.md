# zed_ds_cpp

基于 C++ 的 DeepStream 应用，目标平台为 AGX Orin + DeepStream 6.3 + ZED X + D-FINE。

## 架构说明

- 参考 `example/zed-gstreamer`：使用 `zedsrc + zeddemux`，只对左目做推理，辅助路读取深度。
- 参考 `example/D-FINE`：D-FINE 为多输入模型（`images` + `orig_target_sizes`），通过自定义库补齐输入初始化与输出解析。
- 参考 `DeepStream-Yolo`：沿用 DeepStream 的 `nvinfer + custom-lib` 接入模式。

### 数据流

```
zedsrc(stream-type=4) → zeddemux(is-depth=true)
  ├─ src_left → videoconvert → nvvideoconvert → nvstreammux → nvinfer → [nvdsosd] → sink
  └─ src_aux  → appsink → DepthEstimator（深度帧缓存，PTS 对齐）
```

最终输出：`2D bbox(x,y,w,h) + 深度Z + 相机坐标XYZ`。

### 模型约束

- 单类别模型（`num-detected-classes=1`），标签固定为 `volleyball`。
- 阈值默认 `0.75`，用于抑制无球场景下高置信误检。

### 模型转换
cd /home/nvidia/ObjectDetection

/usr/src/tensorrt/bin/trtexec \
  --onnx=model/best_stg1.onnx \
  --saveEngine=model/best_stg1_fp16.engine \
  --fp16 \
  --minShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --optShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --maxShapes=images:1x3x640x640,orig_target_sizes:1x2 \
  --workspace=4096


## ZED 出厂标定文件

- 默认从 `camera.calibration_file` 读取。
- 若未配置路径且 `use_factory_calib=true`，程序会自动在 `/usr/local/zed/settings/` 下查找 `SN*.conf`。
- 支持的分辨率映射：
  - `SVGA → [LEFT_CAM_SVGA]`
  - `HD1080 → [LEFT_CAM_FHD]`
  - `HD1200 → [LEFT_CAM_FHD1200]`
- 若 `use_distortion_correction=true`，同时读取 `k1/k2/p1/p2/k3` 做 Brown-Conrady 去畸变。

> **注意**：HD720 和 HD2K 在 ZED X 出厂标定中无独立段，使用这些分辨率时需在 YAML 中手动配置内参。

## 编译

```bash
cd apps/zed_ds_cpp
cmake -S . -B build
cmake --build build -j
```

## 运行前检查

```bash
command -v deepstream-app
command -v nvcc
command -v trtexec
gst-inspect-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstzedsrc.so
gst-inspect-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstzeddemux.so
pgrep -x zed_x_daemon || pgrep -x ZEDX_Daemon
```

程序会自动检测 `zed_x_daemon` 服务状态，未检测到时打印告警。

## 运行

```bash
cd /home/nvidia/ObjectDetection
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
./apps/zed_ds_cpp/build/zed_ds_app --config configs/zed_dfine.yaml
```

无显示压测：

```bash
./apps/zed_ds_cpp/build/zed_ds_app --config configs/zed_dfine.yaml --no-display --dump-json logs/run.jsonl
```

### 配置文件选择

- `zed_dfine.yaml`：SVGA@120fps，低延迟高帧率
- `zed_dfine_balanced.yaml`：HD1080@60fps，平衡画质与性能
- `zed_dfine_fast.yaml`：SVGA@120fps，快速预设

### 关键设计

- `nvinfer` 运行时配置自动生成到 `/tmp/zed_ds_runtime/`，不污染源码树。
- 阈值由 YAML `infer.threshold` 单点控制。
- 深度帧 PTS 对齐检查：偏差超过 33ms（约 2 帧@120fps）时打印告警。
- JSONL 输出按 `flush_interval_sec` 批量刷盘（默认 1 秒），避免 120fps 下每帧 fsync。
