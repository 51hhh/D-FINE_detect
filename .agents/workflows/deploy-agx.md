---
description: 本地修改后同步到 AGX Orin 编译测试
---

## 前置条件
- AGX Orin: `nvidia@192.168.31.22`，密码 `nvidia`
- AGX 上项目路径: `/home/nvidia/ObjectDetection`
- 本地项目路径: `/home/rick/desktop/yolo/ObjectDetection`

## 工作流程

### 1. 同步代码到 AGX
// turbo
```bash
rsync -avz --exclude='.git' --exclude='build/' --exclude='model/*.onnx' --exclude='model/*.pth' --exclude='model/*.engine' --exclude='DeepStream-Yolo/' --exclude='example/' --exclude='logs/' /home/rick/desktop/yolo/ObjectDetection/ nvidia@192.168.31.22:/home/nvidia/ObjectDetection/
```

### 2. 远程编译
// turbo
```bash
ssh nvidia@192.168.31.22 'cd /home/nvidia/ObjectDetection/apps/zed_ds_cpp && cmake -S . -B build && cmake --build build -j$(nproc)'
```

### 3. 远程运行（带显示）
```bash
ssh -X nvidia@192.168.31.22 'cd /home/nvidia/ObjectDetection && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 && export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0 && ./apps/zed_ds_cpp/build/zed_ds_app --config configs/zed_dfine.yaml'
```

### 4. 远程运行（无显示压测）
```bash
ssh nvidia@192.168.31.22 'cd /home/nvidia/ObjectDetection && export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1 && export GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0 && ./apps/zed_ds_cpp/build/zed_ds_app --config configs/zed_dfine.yaml --no-display --dump-json logs/run.jsonl'
```

### 5. 查看远程日志
// turbo
```bash
ssh nvidia@192.168.31.22 'tail -20 /home/nvidia/ObjectDetection/logs/frame_result.jsonl'
```
