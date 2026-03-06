#include "pipeline_builder.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <utility>

#include "types.h"

namespace zed_ds {
namespace {
int ResolveZedResolutionEnum(const std::string& res) {
  if (res == "HD2K") return 0;
  if (res == "HD1080") return 1;
  if (res == "HD1200") return 2;
  if (res == "HD720") return 3;
  if (res == "SVGA") return 4;
  if (res == "VGA") return 5;
  return 6;
}

std::pair<int, int> ResolveMuxSize(const std::string& res) {
  if (res == "SVGA") return {960, 600};
  if (res == "VGA") return {672, 376};
  if (res == "HD720") return {1280, 720};
  if (res == "HD1080") return {1920, 1080};
  if (res == "HD1200") return {1920, 1200};
  if (res == "HD2K") return {2208, 1242};
  return {1920, 1080};
}

// 使用 Brown-Conrady 模型迭代去畸变，将畸变像素点映射到校正后的像素点。
std::pair<float, float> UndistortPixel(float u, float v, const CameraIntrinsics& intr) {
  if (!intr.use_distortion_correction || intr.fx <= 1e-6f || intr.fy <= 1e-6f) {
    return {u, v};
  }

  const float xd = (u - intr.cx) / intr.fx;
  const float yd = (v - intr.cy) / intr.fy;
  float x = xd;
  float y = yd;

  for (int i = 0; i < 5; ++i) {
    const float r2 = x * x + y * y;
    const float r4 = r2 * r2;
    const float r6 = r4 * r2;
    const float radial = 1.0f + intr.k1 * r2 + intr.k2 * r4 + intr.k3 * r6;
    const float delta_x = 2.0f * intr.p1 * x * y + intr.p2 * (r2 + 2.0f * x * x);
    const float delta_y = intr.p1 * (r2 + 2.0f * y * y) + 2.0f * intr.p2 * x * y;
    if (std::abs(radial) < 1e-8f) break;
    x = (xd - delta_x) / radial;
    y = (yd - delta_y) / radial;
  }

  return {x * intr.fx + intr.cx, y * intr.fy + intr.cy};
}

struct ProbeUserData {
  RuntimeContext ctx;
};

static GstFlowReturn OnNewDepthSample(GstAppSink* sink, gpointer user_data) {
  auto* ctx = static_cast<RuntimeContext*>(user_data);
  if (!ctx || !ctx->depth) return GST_FLOW_OK;

  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (!sample) return GST_FLOW_OK;

  ctx->depth->UpdateFromSample(sample);
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

static GstPadProbeReturn OnInferSrcPadBuffer(GstPad* pad, GstPadProbeInfo* info, gpointer user_data) {
  (void)pad;
  auto* data = static_cast<ProbeUserData*>(user_data);
  if (!data || !(info->type & GST_PAD_PROBE_TYPE_BUFFER)) return GST_PAD_PROBE_OK;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf) return GST_PAD_PROBE_OK;

  if (data->ctx.perf) {
    data->ctx.perf->MarkFrame(0.0);
  }

  NvDsBatchMeta* batch_meta = gst_buffer_get_nvds_batch_meta(buf);
  if (!batch_meta) return GST_PAD_PROBE_OK;

  FrameResult frame_out;
  frame_out.ts_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch())
                                 .count());

  for (NvDsMetaList* l_frame = batch_meta->frame_meta_list; l_frame; l_frame = l_frame->next) {
    auto* frame_meta = static_cast<NvDsFrameMeta*>(l_frame->data);
    if (!frame_meta) continue;

    frame_out.frame_id = frame_meta->frame_num;

    const int frame_w = frame_meta->source_frame_width > 0 ? frame_meta->source_frame_width : 1920;
    const int frame_h = frame_meta->source_frame_height > 0 ? frame_meta->source_frame_height : 1080;

    for (NvDsMetaList* l_obj = frame_meta->obj_meta_list; l_obj; l_obj = l_obj->next) {
      auto* obj = static_cast<NvDsObjectMeta*>(l_obj->data);
      if (!obj) continue;

      DetectionResult dr;
      dr.det.class_id = obj->class_id;
      dr.det.conf = obj->confidence;
      dr.det.label = obj->obj_label ? obj->obj_label : "unknown";
      dr.det.bbox.left = obj->rect_params.left;
      dr.det.bbox.top = obj->rect_params.top;
      dr.det.bbox.width = obj->rect_params.width;
      dr.det.bbox.height = obj->rect_params.height;
      dr.depth = data->ctx.depth->Estimate(dr.det.bbox, frame_w, frame_h);
      // 用检测框中心像素坐标 + 深度，解算相机坐标系下 XYZ。
      if (dr.depth.valid && data->ctx.intrinsics.fx > 1e-6f && data->ctx.intrinsics.fy > 1e-6f) {
        const float u_raw = dr.det.bbox.left + dr.det.bbox.width * 0.5f;
        const float v_raw = dr.det.bbox.top + dr.det.bbox.height * 0.5f;
        const auto [u, v] = UndistortPixel(u_raw, v_raw, data->ctx.intrinsics);
        dr.depth.z_m = dr.depth.depth_m;
        dr.depth.x_m = (u - data->ctx.intrinsics.cx) * dr.depth.z_m / data->ctx.intrinsics.fx;
        dr.depth.y_m = (v - data->ctx.intrinsics.cy) * dr.depth.z_m / data->ctx.intrinsics.fy;
      }

      if (data->ctx.enable_osd) {
        std::ostringstream ss;
        ss.setf(std::ios::fixed);
        ss.precision(2);
        ss << dr.det.label << " " << dr.det.conf;
        if (dr.depth.valid) {
          ss << " D=" << dr.depth.depth_m << "m";
        } else {
          ss << " D=NA";
        }
        g_free(obj->text_params.display_text);
        obj->text_params.display_text = g_strdup(ss.str().c_str());
      }

      frame_out.detections.push_back(std::move(dr));
    }
  }

  if (data->ctx.writer) {
    data->ctx.writer->Write(frame_out);
  }

  return GST_PAD_PROBE_OK;
}

std::string BuildPipelineStr(const AppConfig& cfg) {
  std::ostringstream ss;
  const int zed_res = ResolveZedResolutionEnum(cfg.camera.resolution);
  const auto [mux_w, mux_h] = ResolveMuxSize(cfg.camera.resolution);

  // 采用 zedsrc + zeddemux：左目一路推理，辅助路提供深度。
  ss << "zedsrc name=zedsrc camera-id=" << cfg.camera.camera_id
     << " camera-fps=" << cfg.camera.fps
     << " camera-resolution=" << zed_res
     << " stream-type=4 "
     << "! zeddemux is-depth=true name=demux "
     << "demux.src_left ! queue leaky=2 max-size-buffers=1 "
     << "! videoconvert ! nvvideoconvert ! video/x-raw(memory:NVMM),format=NV12 "
     << "! mux.sink_0 "
     << "demux.src_aux ! queue leaky=2 max-size-buffers=1 "
     << "! appsink name=depth_sink emit-signals=true sync=false drop=true max-buffers=1 "
     << "nvstreammux name=mux batch-size=1 live-source=1 batched-push-timeout=40000 width=" << mux_w
     << " height=" << mux_h << " "
     << "! nvinfer name=primary config-file-path=" << cfg.infer.config_file << " ";

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

}  // namespace

bool BuildPipeline(const AppConfig& cfg, const RuntimeContext& ctx, PipelineHandles* out, std::string* err) {
  if (!out) {
    if (err) *err = "pipeline output handle is null";
    return false;
  }

  GError* gerr = nullptr;
  const std::string launch = BuildPipelineStr(cfg);
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &gerr);
  if (!pipeline) {
    if (err) *err = gerr ? gerr->message : "gst_parse_launch failed";
    if (gerr) g_error_free(gerr);
    return false;
  }

  GstElement* depth_sink = gst_bin_get_by_name(GST_BIN(pipeline), "depth_sink");
  GstElement* infer = gst_bin_get_by_name(GST_BIN(pipeline), "primary");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");

  if (!depth_sink || !infer || !sink) {
    if (err) *err = "missing expected pipeline elements (depth_sink/primary/sink)";
    if (depth_sink) gst_object_unref(depth_sink);
    if (infer) gst_object_unref(infer);
    if (sink) gst_object_unref(sink);
    gst_object_unref(pipeline);
    return false;
  }

  GstAppSinkCallbacks callbacks = {};
  callbacks.new_sample = OnNewDepthSample;
  gst_app_sink_set_callbacks(GST_APP_SINK(depth_sink), &callbacks, const_cast<RuntimeContext*>(&ctx), nullptr);

  auto* probe_ud = new ProbeUserData{ctx};
  GstPad* infer_src = gst_element_get_static_pad(infer, "src");
  if (!infer_src) {
    if (err) *err = "cannot get nvinfer src pad";
    delete probe_ud;
    gst_object_unref(depth_sink);
    gst_object_unref(infer);
    gst_object_unref(sink);
    gst_object_unref(pipeline);
    return false;
  }

  gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER, OnInferSrcPadBuffer, probe_ud,
                    [](gpointer p) { delete static_cast<ProbeUserData*>(p); });
  gst_object_unref(infer_src);

  out->pipeline = pipeline;
  out->depth_sink = depth_sink;
  out->infer = infer;
  out->sink = sink;
  return true;
}

}  // namespace zed_ds
