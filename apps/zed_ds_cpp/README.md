# zed_ds_cpp

基于 C++ 的 DeepStream 应用骨架，目标平台为 AGX Orin + DeepStream 6.3 + ZED X + D-FINE。

## 架构说明（对齐 example）

- 参考 `example/zed-gstreamer`：使用 `zedsrc + zeddemux`，只对左目做推理，辅助路读取深度。
- 参考 `example/D-FINE`：D-FINE 为多输入模型（`images` + `orig_target_sizes`），通过自定义库补齐输入初始化与输出解析。
- 参考 `DeepStream-Yolo`：沿用 DeepStream 的 `nvinfer + custom-lib` 接入模式，不在应用层重复实现推理框架。

数据流：

`zedsrc(stream-type=4) -> zeddemux`

`src_left -> videoconvert -> nvvideoconvert -> nvstreammux -> nvinfer -> (nvdsosd) -> sink`

`src_aux -> appsink -> 深度估计`

最终输出：`2D bbox(x,y,w,h) + 深度Z + 相机坐标XYZ`。

## ZED 出厂标定文件

- 默认从 `camera.calibration_file` 读取（例如 `/usr/local/zed/settings/SN40013030.conf`）。
- 若未配置路径且 `use_factory_calib=true`，程序会自动在 `/usr/local/zed/settings/` 下查找 `SN*.conf`。
- 程序会根据分辨率读取对应左目内参段：
  - `SVGA -> [LEFT_CAM_SVGA]`
  - `HD1080 -> [LEFT_CAM_FHD]`
  - `HD1200 -> [LEFT_CAM_FHD1200]`
- 若 `use_distortion_correction=true`，会同时读取 `k1/k2/p1/p2/k3`，对检测中心点先做去畸变再解算 XYZ。

## 编译

```bash
cd apps/zed_ds_cpp
cmake -S . -B build
cmake --build build -j
```

## 运行前检查（手动命令）

```bash
command -v deepstream-app
command -v nvcc
command -v trtexec
gst-inspect-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstzedsrc.so
gst-inspect-1.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstzeddemux.so
pgrep -x zed_x_daemon
pgrep -x ZEDX_Daemon
```

程序会优先通过 `systemctl is-active zed_x_daemon` 检查服务，
并兼容检查进程名 `ZEDX_Daemon`。
若仍未检测到，程序会打印中文告警，且 ZED X 采集可能失败或不稳定。
可在设备上执行：

```bash
sudo systemctl restart zed_x_daemon
```

## 运行

```bash
cd /home/nvidia/ObjectDetection
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0
./apps/zed_ds_cpp/build/zed_ds_app --config configs/zed_dfine_fast.yaml --mode max_fps
```

无显示压测：

```bash
./apps/zed_ds_cpp/build/zed_ds_app --config configs/zed_dfine_balanced.yaml --mode balanced --no-display --dump-json logs/run.jsonl
```

性能策略：
- `perf.warmup_sec` 为热身时长，热身期不触发自动降档。
- `perf.enable_auto_fallback` 控制是否自动降档并重启管线（默认建议 `false`，更稳定）。
