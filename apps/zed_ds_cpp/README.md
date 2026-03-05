# zed_ds_cpp

基于 C++ 的 DeepStream 应用骨架，目标平台为 AGX Orin + DeepStream 6.3 + ZED X + D-FINE。

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
