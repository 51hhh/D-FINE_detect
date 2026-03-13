# D-FINE 单类微调 Query Over-Activation 问题分析与解决方案

## 1. 问题现象

在 AGX Orin 上部署 D-FINE 单类（volleyball）模型时：

- **有排球时**：正确识别，置信度 80-90%
- **无排球时**：随机识别其他物体（人、球网、地板等）为排球，置信度同样 80%+
- **对比**：YOLO 模型在相同场景下不存在此问题

## 2. 术语定义

| 术语 | 含义 |
|---|---|
| **Query Over-Activation**（查询过激活） | DETR 系列中，预训练的 learned queries 在微调后仍保留对非目标物体的定位能力，导致错误激活 |
| **False Activation**（假激活） | D-FINE Issue #277 中使用的原词，指模型对非目标物体产生高置信度检测 |
| **Objectness-Classification Conflation** | 目标性与分类的混淆——单类分类头退化为"是否有物体"而非"是否是目标类别" |
| **Class Collapse**（类别坍缩） | 多类预训练模型微调为单类后，分类维度坍缩，丧失类别区分能力 |

## 3. 架构原理：为什么 D-FINE 有此问题而 YOLO 没有

### 3.1 DETR 系列架构（D-FINE / RT-DETR）

D-FINE 基于 RT-DETR 改进，属于 DETR（DEtection TRansformer）家族。核心架构：

```
输入图像
  │
  ▼
┌──────────────┐
│   Backbone   │  HGNet-v2 (特征提取)
│  (Encoder)   │
└──────┬───────┘
       │  多尺度特征图
       ▼
┌──────────────┐
│  Hybrid      │  高效混合编码器
│  Encoder     │  (特征融合 + 自注意力)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│      Transformer Decoder         │
│                                  │
│  300 个 Learned Object Queries   │ ◄── 问题根源
│         ×                        │
│  Cross-Attention (查询 × 特征)   │
│         ×                        │
│  6 层 Decoder Layer              │
└──────────┬───────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌──────────┐
│ 分类头   │ │  回归头   │
│ scores  │ │  boxes   │
│ labels  │ │ (x1,y1,  │
│         │ │  x2,y2)  │
└─────────┘ └──────────┘
```

**关键概念：Learned Object Queries**

- D-FINE 使用 **300 个可学习的查询向量**（query embeddings）
- 每个 query 在训练过程中"专攻"检测特定位置和类型的物体
- 在 COCO 预训练中，这 300 个 query 学会了定位 80 类物体（人、车、椅子、球...）
- 推理时，每个 query 通过 cross-attention 与图像特征交互，输出一个检测结果

**D-FINE 的独特改进：Fine-grained Distribution Refinement (FDR)**

D-FINE 将边界框回归重新定义为概率分布的精细化过程，而非直接回归坐标：

```
传统 DETR：query → [x, y, w, h]（4个确定值）
D-FINE：   query → [P(x), P(y), P(w), P(h)]（4个概率分布）
                     每个分布有 N 个离散 bin
```

这使得 D-FINE 的边界框质量更高（这也是你不想换回 YOLO 的原因），但分类问题依然存在。

### 3.2 YOLO 架构对比

```
输入图像
  │
  ▼
┌──────────────┐
│   Backbone   │  CSPDarknet / ELAN
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     Neck     │  FPN + PAN (多尺度融合)
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│        Detection Head            │
│                                  │
│  每个 anchor/grid cell 输出：     │
│  [objectness, class_probs, bbox] │
│                                  │
│  最终分数 = objectness × class_p │ ◄── 双层过滤
└──────────────────────────────────┘
```

### 3.3 核心差异导致的行为差异

```
                    YOLO                           D-FINE (DETR系)
                    ────                           ──────────────
检测机制            Dense (密集预测)                Sparse (稀疏查询)
                    每个grid cell都预测             300个query竞争

分数组成            objectness × class_prob         class_score (直接)
                    两层独立过滤                    单层分数

单类训练后          objectness高 但 class_prob低    query被激活 → 只能输出唯一类
对非目标物体        → 最终分数低 → 被过滤           → 高置信度 → 误检
的表现

背景抑制            objectness天然学会区分           依赖分类头中的no-object类
                    "有物体"vs"无物体"              单类时no-object学习不充分
```

**简单类比：**

- YOLO 像一个有两道门的安检：第一道问"这里有东西吗？"，第二道问"这东西是排球吗？"，两道都通过才放行
- D-FINE 像一个只有一道门的安检：问"这是什么？"，但只有"排球"和"没东西"两个选项。预训练记忆让它对很多物体都倾向于说"有东西"，而有东西就只能归类为"排球"

### 3.4 数学层面的解释

**YOLO 的输出层（单类情况）：**

```
P(volleyball | cell_i) = σ(obj_i) × σ(cls_i)

其中：
  σ(obj_i) = sigmoid(objectness logit)  → "这个位置有物体吗"
  σ(cls_i) = sigmoid(class logit)       → "这个物体是排球吗"

对于非排球物体（如人）：
  σ(obj_i) ≈ 0.9  (确实有物体)
  σ(cls_i) ≈ 0.1  (但不是排球)
  最终分数 ≈ 0.09  → 远低于阈值，被过滤
```

**D-FINE 的输出层（单类情况）：**

```
P(class_i | query_j) = softmax([logit_volleyball, logit_no_object])

其中 softmax 只有两个类别：
  logit_volleyball  → "是排球"
  logit_no_object   → "没有目标"

对于非排球物体（如人）：
  query 的 cross-attention 被物体特征强激活
  logit_volleyball 被推高（因为query认为"这里有东西"）
  logit_no_object 被压低
  P(volleyball) ≈ 0.85  → 超过阈值，误检！
```

**根本原因：** D-FINE 的分类头在 2-class softmax 下，"有物体" 和 "是排球" 成为了近似等价的判断。预训练的 query 擅长定位物体，微调后的分类头无法有效区分"是排球的物体"和"不是排球的物体"。

## 4. D-FINE 社区验证的解决方案

### 4.1 方案 A：空标注图片加入训练（已验证有效，推荐首选）

**来源：** [Issue #277](https://github.com/Peterande/D-FINE/issues/277)

社区成员 @PKurnikov 确认有效：
> "Yes, this approach works because the loss will be computed on false activations, which encourages the model to make fewer mistakes, assuming the data is homoscedastic and from similar domains."

@xuechaoyan 的实战经验：
> "把你的结果生成出来看，看看算法具体在哪些地方容易误报，然后把这类的图片当成没有标签的图片拿去训练。在我的实验场景下，这样可以降低很多的误报"

**原理：** 空标注图片让所有 300 个 query 都被迫匹配到 no-object 类，梯度直接惩罚 false activation，迫使 query 学会"虽然这里有物体，但不应该激活"。

**操作步骤：**

1. **收集误检场景图片**（不需要标注，工作量极小）：
   - 空场地、只有人、只有球网、观众席等
   - 在 AGX 上跑一轮推理，截取误检帧即可
   - 占训练集 15-25%

2. **COCO 格式空标注**：
   ```json
   {
     "images": [
       {"id": 10001, "file_name": "negative_001.jpg", "width": 640, "height": 640}
     ],
     "annotations": []
   }
   ```

3. **必须打 Issue #247 补丁防止死锁**：

   D-FINE 的 contrastive denoising 模块在空标注时会跳过 DN loss 计算，导致多 GPU 训练时 `reduce_dict()` 的 key 不一致而死锁。

   **补丁位置：** `DFINECriterion.forward()` 末尾添加：
   ```python
   expected_keys = [
       'loss_bbox_dn_0', 'loss_bbox_dn_1', 'loss_bbox_dn_2',
       'loss_bbox_dn_3', 'loss_bbox_dn_pre',
       'loss_ddf_dn_0', 'loss_ddf_dn_1', 'loss_ddf_dn_2',
       'loss_fgl_dn_0', 'loss_fgl_dn_1', 'loss_fgl_dn_2',
       'loss_fgl_dn_3',
       'loss_giou_dn_0', 'loss_giou_dn_1', 'loss_giou_dn_2',
       'loss_giou_dn_3', 'loss_giou_dn_pre',
       'loss_vfl_dn_0', 'loss_vfl_dn_1', 'loss_vfl_dn_2',
       'loss_vfl_dn_3', 'loss_vfl_dn_pre'
   ]
   device = outputs['pred_logits'].device
   for key in expected_keys:
       if key not in losses:
           losses[key] = torch.tensor(0.0, device=device)
   ```
   来源：[Issue #247](https://github.com/Peterande/D-FINE/issues/247)

### 4.2 方案 B：训练策略调整

**来源：** [Issue #169](https://github.com/Peterande/D-FINE/issues/169)，D-FINE 作者 @Peterande 回复：

> "对于类别少、结构简单的数据集，**从头训练（train from scratch）效果更好**，不要用 Obj365+COCO 预训练权重微调"

| 参数 | 当前值 | 建议值 | 说明 |
|---|---|---|---|
| 预训练权重 | Obj365+COCO | **无 / 仅 Objects365** | 单类简单场景从头训更好 |
| `eos_coefficient` | 0.1（默认） | **0.3-0.5** | 增大 no-object 损失权重，抑制假激活 |
| `num_queries` | 300 | **50-100** | 减少 query 数量，降低过激活概率 |
| 学习率 | - | transformer 1e-4, backbone 1e-5 | 过高导致训练崩溃 |
| `auxiliary_loss` | - | `True` | 每层 decoder 都加监督，学会正确的目标数量 |

**`eos_coefficient` 原理详解：**

D-FINE 的分类损失中，对每个 query 的预测使用 Hungarian 匹配。300 个 query 中，大部分应匹配到 "no-object"。`eos_coefficient` 控制 no-object 类的损失权重：

```
L_cls = Σ(matched queries) L_vfl(pred, gt_class)
      + eos_coefficient × Σ(unmatched queries) L_vfl(pred, no_object)

eos_coefficient = 0.1 → 模型不太在意"误报"
eos_coefficient = 0.5 → 模型更用力学习"这里没东西"
```

### 4.3 方案 C：推理端几何过滤（立即生效，无需重训）

在 DeepStream 自定义解析器中添加排球几何约束，过滤明显不合理的检测框：

```cpp
// nvdsinfer_custom_dfine.cpp — push_back 之前

// 排球 aspect ratio 接近 1:1
const float aspect = obj.width / std::max(obj.height, 1.0f);
if (aspect < 0.5f || aspect > 2.0f) continue;

// 面积约束（根据相机距离和分辨率调整）
const float area = obj.width * obj.height;
if (area < 200.0f || area > 80000.0f) continue;
```

**注意：** 此方案只是辅助手段，无法替代方案 A/B。某些误检（如篮球、人头）的几何特征与排球相似，仍会漏过。

### 4.4 方案 D：Denoising 区域负样本（高阶，需改训练代码）

来自 [Issue #277](https://github.com/Peterande/D-FINE/issues/277) @guods 的思路：

> 在 `denoising.py` 中，在与图片原有目标没有交集的区域随机生成框，将其标签视为背景类参与分类损失

原理：利用 D-FINE 已有的 contrastive denoising 机制，在非目标区域注入负样本，强制 query 学会抑制。不需要额外数据，但需要修改 `src/zoo/dfine/dfine_criterion.py` 和 `src/zoo/dfine/denoising.py`。

## 5. 推荐行动优先级

```
┌─ 立即（几分钟）──────────────────────────────┐
│  方案 C: 解析器几何过滤                        │
│  效果: 过滤掉形状明显不对的误检                 │
│  工作量: 改 5 行 C++ 代码                      │
└──────────────────────────────────────────────┘
         │
         ▼
┌─ 短期（数小时）──────────────────────────────┐
│  方案 A: 空标注图片重训                        │
│  效果: 社区验证有效，从根本抑制假激活           │
│  工作量: 收集误检截图 + 打 #247 补丁 + 重训    │
│  注意: 不需要额外标注，工作量很小               │
└──────────────────────────────────────────────┘
         │
         ▼
┌─ 中期（如果方案A效果不够）───────────────────┐
│  方案 B: 从头训 + 调参                        │
│  效果: 彻底解决，但训练时间更长                 │
│  关键: eos_coefficient↑  num_queries↓         │
└──────────────────────────────────────────────┘
```

## 6. 参考链接

- [D-FINE Issue #277: Reducing False Positives with Empty annotation Images](https://github.com/Peterande/D-FINE/issues/277)
- [D-FINE Issue #247: Deadlock fix for empty targets](https://github.com/Peterande/D-FINE/issues/247)
- [D-FINE Issue #169: Finetuning is not stable — 作者回复](https://github.com/Peterande/D-FINE/issues/169)
- [D-FINE Issue #108: Performance Issue Fine-Tuning single class](https://github.com/Peterande/D-FINE/issues/108)
- [D-FINE Issue #168: num_classes not having effect](https://github.com/Peterande/D-FINE/issues/168)
- [D-FINE 论文: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement (ICLR 2025)](https://arxiv.org/html/2410.13842v1)
- [Facebook DETR Issue #365: Background class configuration](https://github.com/facebookresearch/detr/issues/365)
- [RF-DETR Issue #348: Single class fine-tuning](https://github.com/roboflow/rf-detr/issues/348)
- [ArgoHA/custom_d_fine: 社区推荐的改进训练框架](https://github.com/ArgoHA/custom_d_fine)
