#include "pipeline_builder.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <utility>

#include "types.h"

namespace zed_ds {
namespace {

// ── zedsrc camera-resolution 枚举值映射 ──
int ResolveZedResolutionEnum(const std::string &res) {
  if (res == "HD2K")
    return 0;
  if (res == "HD1080")
    return 1;
  if (res == "HD1200")
    return 2;
  if (res == "HD720")
    return 3;
  if (res == "SVGA")
    return 4;
  if (res == "VGA")
    return 5;
  return 6;
}

// ── nvstreammux 输入尺寸（需与 zeddemux src_left 实际输出一致） ──
// 参照 example/zed-gstreamer/gst-zed-demux/gstzeddemux.cpp 中的 src_left caps。
std::pair<int, int> ResolveMuxSize(const std::string &res) {
  if (res == "SVGA")
    return {960, 600};
  if (res == "VGA")
    return {672, 376};
  if (res == "HD720")
    return {1280, 720};
  if (res == "HD1080")
    return {1920, 1080};
  if (res == "HD1200")
    return {1920, 1200};
  if (res == "HD2K")
    return {2208, 1242};
  return {1920, 1080};
}

// ── Brown-Conrady 迭代去畸变 ──
// 将畸变像素点 (u,v) 映射到校正后的像素点。
std::pair<float, float> UndistortPixel(float u, float v,
                                       const CameraConfig &cam) {
  if (!cam.use_distortion_correction || cam.fx <= 1e-6f || cam.fy <= 1e-6f) {
    return {u, v};
  }

  const float xd = (u - cam.cx) / cam.fx;
  const float yd = (v - cam.cy) / cam.fy;
  float x = xd;
  float y = yd;

  for (int i = 0; i < 5; ++i) {
    const float r2 = x * x + y * y;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float radial = 1.0f + cam.k1 * r2 + cam.k2 * r4 + cam.k3 * r6;
    const float delta_x = 2.0f * cam.p1 * x * y + cam.p2 * (r2 + 2.0f * x * x);
    const float delta_y = cam.p1 * (r2 + 2.0f * y * y) + 2.0f * cam.p2 * x * y;
    if (std::abs(radial) < 1e-8f)
      break;
    x = (xd - delta_x) / radial;
    y = (yd - delta_y) / radial;
  }

  return {x * cam.fx + cam.cx, y * cam.fy + cam.cy};
}

// ── appsink 回调：接收深度帧 ──
static GstFlowReturn OnNewDepthSample(GstAppSink *sink, gpointer user_data) {
  auto *ctx = static_cast<RuntimeContext *>(user_data);
  if (!ctx || !ctx->depth)
    return GST_FLOW_OK;

  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample)
    return GST_FLOW_OK;

  ctx->depth->UpdateFromSample(sample);
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

// ── nvinfer probe 回调用户数据 ──
// 持有 shared_ptr<RuntimeContext>，保证生命周期安全。
struct ProbeUserData {
  std::shared_ptr<RuntimeContext> ctx;
};

// ── nvinfer src pad probe：处理推理后的检测结果 ──
static GstPadProbeReturn OnInferSrcPadBuffer(GstPad *pad, GstPadProbeInfo *info,
                                             gpointer user_data) {
  (void)pad;
  auto *data = static_cast<ProbeUserData *>(user_data);
  if (!data || !data->ctx || !(info->type & GST_PAD_PROBE_TYPE_BUFFER))
    return GST_PAD_PROBE_OK;

  GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return GST_PAD_PROBE_OK;

  auto &ctx = *data->ctx;
  const auto now = std::chrono::steady_clock::now();

  // 计算端到端延迟：buffer PTS 到当前时间差。
  double latency_ms = 0.0;
  if (GST_BUFFER_PTS_IS_VALID(buf)) {
    const uint64_t buf_pts_ns = GST_BUFFER_PTS(buf);
    // 用 buffer 时间戳与系统时钟的差值估算延迟。
    // 注意：PTS 是相机时钟域，steady_clock 是系统时钟域，两者可能有偏移。
    // 这里采用帧间间隔推算，如果无法计算则记录 0。
    (void)buf_pts_ns; // 后续可对接更精确的延迟测量方案。
  }
  // 简化方案：用 probe 回调的调用间隔近似帧处理时间。
  static thread_local auto last_probe_time = now;
  latency_ms =
      std::chrono::duration<double, std::milli>(now - last_probe_time).count();
  last_probe_time = now;

  if (ctx.perf) {
    ctx.perf->MarkFrame(latency_ms);
  }

  // 帧对齐检查：推理帧 PTS vs 深度帧 PTS。
  // 推理路径经过 nvinfer 有固有延迟（通常 30-50ms），因此推理帧的 PTS
  // 总是比最新深度帧的 PTS 更早。这里的 "drift" 表示深度帧比推理帧新多少毫秒。
  // 在 SVGA@120fps + D-FINE 场景下，30-50ms 偏差（约 3-6 帧）是正常的。
  // 只有偏差异常大时才告警（可能是管线卡顿或帧丢失）。
  if (ctx.depth && GST_BUFFER_PTS_IS_VALID(buf)) {
    const uint64_t infer_pts = GST_BUFFER_PTS(buf);
    const uint64_t depth_pts = ctx.depth->LastDepthPtsNs();
    if (depth_pts > 0 && infer_pts > 0) {
      const int64_t drift_ms = static_cast<int64_t>(depth_pts / 1000000) -
                               static_cast<int64_t>(infer_pts / 1000000);
      // 超过 100ms 偏差说明管线可能存在严重卡顿，每 200 帧采样一次避免刷屏。
      static thread_local uint64_t drift_check_count = 0;
      ++drift_check_count;
      if (std::abs(drift_ms) > 100 && (drift_check_count % 200 == 1)) {
        std::cerr << "[warn] 深度帧-推理帧 PTS 偏差 " << drift_ms
                  << "ms（预期 30-50ms），管线可能卡顿。" << std::endl;
      }
    }
  }

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta)
    return GST_PAD_PROBE_OK;

  FrameResult frame_out;
  frame_out.ts_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch())
          .count());

  for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame;
       l_frame = l_frame->next) {
    auto *frame_meta = static_cast<NvDsFrameMeta *>(l_frame->data);
    if (!frame_meta)
      continue;

    frame_out.frame_id = frame_meta->frame_num;

    const int frame_w = frame_meta->source_frame_width > 0
                            ? frame_meta->source_frame_width
                            : 1920;
    const int frame_h = frame_meta->source_frame_height > 0
                            ? frame_meta->source_frame_height
                            : 1080;

    for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj;
         l_obj = l_obj->next) {
      auto *obj = static_cast<NvDsObjectMeta *>(l_obj->data);
      if (!obj)
        continue;

      DetectionResult dr;
      dr.det.class_id = obj->class_id;
      // cluster-mode=4 时 obj->confidence 可能为 -0.1 或 NaN。
      // 若无效则置 0，不影响检测结果（已通过阈值筛选）。
      dr.det.conf = (std::isfinite(obj->confidence) && obj->confidence >= 0.0f)
                        ? obj->confidence
                        : 0.0f;
      dr.det.label = obj->obj_label ? obj->obj_label : "unknown";
      dr.det.bbox.left = obj->rect_params.left;
      dr.det.bbox.top = obj->rect_params.top;
      dr.det.bbox.width = obj->rect_params.width;
      dr.det.bbox.height = obj->rect_params.height;

      // 深度估计。
      if (ctx.depth) {
        dr.depth = ctx.depth->Estimate(dr.det.bbox, frame_w, frame_h);
      }

      // 用检测框中心像素坐标 + 深度，解算相机坐标系下 XYZ。
      if (dr.depth.valid && ctx.camera && ctx.camera->fx > 1e-6f &&
          ctx.camera->fy > 1e-6f) {
        const float u_raw = dr.det.bbox.left + dr.det.bbox.width * 0.5f;
        const float v_raw = dr.det.bbox.top + dr.det.bbox.height * 0.5f;
        const auto [u, v] = UndistortPixel(u_raw, v_raw, *ctx.camera);
        dr.depth.z_m = dr.depth.depth_m;
        dr.depth.x_m = (u - ctx.camera->cx) * dr.depth.z_m / ctx.camera->fx;
        dr.depth.y_m = (v - ctx.camera->cy) * dr.depth.z_m / ctx.camera->fy;
      }

      // OSD 叠加文字。
      if (ctx.enable_osd) {
        std::ostringstream ss;
        ss.setf(std::ios::fixed);
        ss.precision(2);
        ss << dr.det.label << " " << dr.det.conf;
        if (dr.depth.valid) {
          ss << " D=" << dr.depth.depth_m << "m";
          ss << " (" << dr.depth.x_m << "," << dr.depth.y_m << ","
             << dr.depth.z_m << ")";
        } else {
          ss << " D=NA";
        }
        g_free(obj->text_params.display_text);
        obj->text_params.display_text = g_strdup(ss.str().c_str());
      }

      frame_out.detections.push_back(std::move(dr));
    }
  }

  if (ctx.writer) {
    ctx.writer->Write(frame_out);
  }

  return GST_PAD_PROBE_OK;
}

// ── 构建 GStreamer 管线描述字符串 ──
std::string BuildPipelineStr(const AppConfig &cfg) {
  std::ostringstream ss;
  const int zed_res = ResolveZedResolutionEnum(cfg.camera.resolution);
  const auto [mux_w, mux_h] = ResolveMuxSize(cfg.camera.resolution);

  // zedsrc + zeddemux：左目一路推理，辅助路提供深度。
  // stream-type=4 表示 left+depth 双路输出。
  // depth-mode=1 表示 PERFORMANCE 模式，优先保证实时性。
  ss << "zedsrc name=zedsrc camera-id=" << cfg.camera.camera_id
     << " camera-fps=" << cfg.camera.fps << " camera-resolution=" << zed_res
     << " depth-mode=1"
     << " camera-disable-self-calib=true"
     << " stream-type=4 "
     << "! zeddemux is-depth=true name=demux "
     << "demux.src_left ! queue leaky=2 max-size-buffers=1 "
     << "! videoconvert ! nvvideoconvert ! "
        "video/x-raw(memory:NVMM),format=NV12 "
     << "! mux.sink_0 "
     << "demux.src_aux ! queue leaky=2 max-size-buffers=1 "
     << "! appsink name=depth_sink emit-signals=true sync=false drop=true "
        "max-buffers=1 "
     << "nvstreammux name=mux batch-size=1 live-source=1 "
        "batched-push-timeout=40000 width="
     << mux_w << " height=" << mux_h << " "
     << "! nvinfer name=primary config-file-path=" << cfg.infer.config_file
     << " ";

  if (cfg.output.enable_osd) {
    ss << "! nvdsosd name=osd ";
  }

  if (cfg.output.display) {
    ss << "! nvegltransform ! nveglglessink name=sink sync=false";
  } else {
    ss << "! fakesink name=sink sync=false";
  }

  return ss.str();
}

} // namespace

bool BuildPipeline(const AppConfig &cfg, std::shared_ptr<RuntimeContext> ctx,
                   PipelineHandles *out, std::string *err) {
  if (!out || !ctx) {
    if (err)
      *err = "pipeline output handle or context is null";
    return false;
  }

  GError *gerr = nullptr;
  const std::string launch = BuildPipelineStr(cfg);
  GstElement *pipeline = gst_parse_launch(launch.c_str(), &gerr);
  if (!pipeline) {
    if (err)
      *err = gerr ? gerr->message : "gst_parse_launch failed";
    if (gerr)
      g_error_free(gerr);
    return false;
  }

  GstElement *depth_sink = gst_bin_get_by_name(GST_BIN(pipeline), "depth_sink");
  GstElement *infer = gst_bin_get_by_name(GST_BIN(pipeline), "primary");
  GstElement *sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");

  if (!depth_sink || !infer || !sink) {
    if (err)
      *err = "missing expected pipeline elements (depth_sink/primary/sink)";
    if (depth_sink)
      gst_object_unref(depth_sink);
    if (infer)
      gst_object_unref(infer);
    if (sink)
      gst_object_unref(sink);
    gst_object_unref(pipeline);
    return false;
  }

  // appsink 回调：传入 shared_ptr 的裸指针，通过引用计数保证生命周期安全。
  // destroy_notify 递减引用计数。
  auto *ctx_ref = new std::shared_ptr<RuntimeContext>(ctx);
  GstAppSinkCallbacks callbacks = {};
  callbacks.new_sample = OnNewDepthSample;
  gst_app_sink_set_callbacks(GST_APP_SINK(depth_sink), &callbacks,
                             ctx_ref->get(), [](gpointer p) {
                               // 注意：这里 p 是 ctx_ref->get() 返回的
                               // RuntimeContext*，
                               // 但实际释放需要从外部管理。使用
                               // g_object_set_data 绑定。
                             });

  // 用 g_object_set_data_full 把 shared_ptr 绑定到 depth_sink 的生命周期上，
  // 确保 depth_sink 销毁时 shared_ptr 也被释放。
  g_object_set_data_full(
      G_OBJECT(depth_sink), "zed_ds_ctx", ctx_ref, [](gpointer p) {
        delete static_cast<std::shared_ptr<RuntimeContext> *>(p);
      });

  // nvinfer probe 回调。
  auto *probe_ud = new ProbeUserData{ctx};
  GstPad *infer_src = gst_element_get_static_pad(infer, "src");
  if (!infer_src) {
    if (err)
      *err = "cannot get nvinfer src pad";
    delete probe_ud;
    gst_object_unref(depth_sink);
    gst_object_unref(infer);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return false;
  }

  gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER, OnInferSrcPadBuffer,
                    probe_ud,
                    [](gpointer p) { delete static_cast<ProbeUserData *>(p); });
  gst_object_unref(infer_src);

  out->pipeline = pipeline;
  out->depth_sink = depth_sink;
  out->infer = infer;
  out->sink = sink;
  return true;
}

} // namespace zed_ds
