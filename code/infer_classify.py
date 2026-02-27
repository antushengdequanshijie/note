#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分类模型推理脚本：支持单图与目录批量。

示例：
  # 单张图片
  python infer_classify.py --weights ../runs/classify/best.pt --img ../test_data/roi.jpg

  # 目录批量
  python infer_classify.py --weights ../runs/classify/best.pt --dir ../nums/0

要求：
  - 训练脚本保存的 best.pt/last.pt（包含 model 名称与 imgsz）
  - 同目录或输出目录有 classes.json（若无，会从权重内读取并生成）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def build_tf(imgsz: int):
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "resnet34":
        m = models.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model: {name}")


def load_weights(weights_path: Path):
    ckpt = torch.load(weights_path, map_location="cpu")
    model_name = ckpt.get("model")
    imgsz = ckpt.get("imgsz", 224)
    classes = ckpt.get("classes")
    state_dict = ckpt.get("state_dict")
    if model_name is None or state_dict is None:
        raise RuntimeError("权重文件缺少必要字段：model/state_dict")
    return model_name, imgsz, classes, state_dict


def list_images(dir_path: Path) -> List[Path]:
    files = [p for p in dir_path.rglob("*") if p.suffix in IMG_EXTS]
    return sorted(files)


def infer_one(model: nn.Module, tf: transforms.Compose, img_path: Path, device: torch.device, class_names: List[str]) -> Tuple[str, float]:
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"无法读取图片：{img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = transforms.functional.to_pil_image(img_rgb)
    x = tf(pil_img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
    idx = int(pred.item())
    label = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
    return label, float(conf.item())


def main():
    ap = argparse.ArgumentParser(description="图像分类推理(单图/批量)")
    ap.add_argument("--weights", type=str, required=True, help="训练保存的权重(best.pt/last.pt)")
    ap.add_argument("--img", type=str, help="单张图片路径")
    ap.add_argument("--dir", type=str, help="批量图片所在目录(递归)")
    ap.add_argument("--device", type=str, default="", help="设备：''自动/cpu/cuda:0")
    ap.add_argument("--out", type=str, default="../runs/classify_infer", help="结果输出目录(批量时保存CSV)")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"权重不存在：{weights_path}")

    model_name, imgsz, classes, state_dict = load_weights(weights_path)
    if classes is None:
        # 若权重中无 classes，尝试读取同目录 classes.json
        cj = weights_path.parent / "classes.json"
        if cj.exists():
            classes = json.loads(cj.read_text(encoding="utf-8"))
            # classes 可能是列表或 {idx:name} 映射
            if isinstance(classes, dict):
                classes = [classes[i] for i in sorted(classes.keys())]
        else:
            raise SystemExit("找不到类别映射 classes；请确保训练时保存或提供 classes.json")

    # 统一为列表形式
    if isinstance(classes, dict):
        classes = [classes[i] for i in sorted(classes.keys())]

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    model = build_model(model_name, num_classes=len(classes))
    model.load_state_dict(state_dict)
    model = model.to(device)
    tf = build_tf(imgsz)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.img:
        img_path = Path(args.img)
        label, conf = infer_one(model, tf, img_path, device, classes)
        print(f"{img_path.name} -> {label} ({conf:.3f})")
        return

    if args.dir:
        import csv
        dir_path = Path(args.dir)
        files = list_images(dir_path)
        csv_path = out_dir / "infer_results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "label", "conf"]) 
            for p in files:
                try:
                    label, conf = infer_one(model, tf, p, device, classes)
                    w.writerow([str(p), label, f"{conf:.6f}"])
                    print(f"{p.name} -> {label} ({conf:.3f})")
                except Exception as e:
                    print(f"跳过 {p}: {e}")
        print(f"批量结果已保存：{csv_path}")
        return

    print("请提供 --img 或 --dir 进行推理")


if __name__ == "__main__":
    main()
