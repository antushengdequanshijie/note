# 脚本说明

本目录主要用于数字检测与识别流程，覆盖：
- 数据裁剪与数据集构建（Labelme → YOLO / 分类数据）
- 数据增强
- YOLO 检测训练
- 分类训练
- 单图/批量推理（检测、分类、检测+分类）
- 人工复核与手动分拣

---

## 一、数据准备类

### 1) `crop_from_labelme.py`
- 作用：读取图片同名 Labelme JSON，根据 `shapes` 裁剪目标区域。
- 输出：按 `label` 自动建子目录保存裁剪图。
- 适用：从检测标注中提取目标小图，构建分类数据初始集。

### 2) `data_process.py`
- 作用：把“图片 + Labelme JSON”转换为 YOLO 检测格式（`images/` + `labels/`）。
- 特点：支持 `rectangle/polygon`，生成 `classes.txt` 与 `dataset.yaml`。
- 备注：当前实现仅处理“有同名 JSON 的图片”。

### 3) `prepare_yolo_dataset.py`
- 作用：更完整的 YOLO 数据集构建脚本（含 `train/test` 划分）。
- 特点：
	- 支持无标注负样本（空标签）
	- 可配置训练/测试负样本比例（`--neg-ratio`、`--neg-ratio-test`）
	- 自动输出 `classes.txt`、`dataset.yaml`
- 适用：工业场景需要控制负样本占比时。

### 4) `data_aug.py`
- 作用：分类小图增强（模糊、仿射扭曲、亮度/对比度、噪声背景融合）。
- 输入：按类别目录组织的数据集。
- 输出：保持类别目录结构，生成增强后数据。

---

## 二、训练类

### 5) `train.py`
- 作用：YOLOv8 检测训练（基础版）。
- 输入：`dataset.yaml`。
- 输出：`runs/detect/...` 下训练结果与权重。

### 6) `train_yolo.py`
- 作用：YOLOv8 检测训练（更偏项目实战参数）。
- 特点：默认 `imgsz=1024`、`mosaic=0.0`、`rect=True`，适配小目标场景。
- 输入/输出：同 `train.py`。

### 7) `train_classify.py`
- 作用：分类模型训练（`ImageFolder` 目录结构）。
- 支持模型：`resnet18/34/50`、`mobilenet_v3_small`、`efficientnet_b0`。
- 特点：
	- 内置较丰富的数据增强
	- 保存 `best.pt` / `last.pt`
	- 额外输出 `classes.json`

---

## 三、推理类

### 8) `infer_and_crop.py`
- 作用：单张图片 YOLO 检测并保存检测框裁剪图。
- 输出：
	- 标注图 `*_det.jpg`
	- 裁剪图目录 `crops/`
	- 结果摘要 `*_det.json`

### 9) `infer_and_crop_batch.py`
- 作用：批量版检测+裁剪，复用 `infer_and_crop.py` 的逻辑。
- 特点：支持递归扫描、跳过已处理样本、生成 `batch_summary.json`。

### 10) `infer_classify.py`
- 作用：分类模型推理（单图或目录批量）。
- 输出：
	- 单图：终端打印类别与置信度
	- 批量：`infer_results.csv`

### 11) `infer_detect_and_classify.py`
- 作用：两阶段推理：先检测，再对每个检测框做分类。
- 输出：
	- 叠加标注图 `*_det_cls.jpg`
	- 每个框的裁剪图
	- 详细 JSON `*_det_cls.json`
	- 左到右组合数字结果 `*_digits.json`（`ordered_digits`）
- 支持：单图与目录批量模式。

### 12) `process_frame_crops.py`
- 作用：面向视频/帧流程。先按给定大框裁切，再对每个裁切结果执行“检测+分类”。
- 特点：按 `ts + frame_id` 组织输出，并聚合成总 `summary` JSON。
- 适用：上游已有目标框（追踪/检测）时的二次识别流程。

---

## 四、人工复核工具

### 13) `manual_digit_sort.py`
- 作用：交互式人工分拣图片，按键 `0~9` 移动到对应类别目录。
- 快捷键：
	- `0~9`：归类
	- `s`：跳过
	- `u`：撤销上一步
	- `q`：退出
- 输出：分类后的目录结构 + `sort_log.csv`。

---

## 五、推荐使用流程

1. 原始标注数据整理：`prepare_yolo_dataset.py`（或 `data_process.py`）
2. 检测模型训练：`train_yolo.py`（或 `train.py`）
3. 分类数据准备：`crop_from_labelme.py` + `data_aug.py`
4. 分类模型训练：`train_classify.py`
5. 联合推理：`infer_detect_and_classify.py`
6. 人工兜底校正：`manual_digit_sort.py`

---

## 六、依赖建议

常用依赖：
- `ultralytics`
- `torch` / `torchvision`
- `opencv-python`
- `numpy`

如需统一环境，可后续补一个 `requirements.txt`。
