#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 data 目录下的 JPG/PNG 图片与同名 LabelMe JSON 转换为 YOLO 检测数据集，
同时按比例切分为 train/test 两个子集，并生成 dataset.yaml。

功能概述：
- 扫描源目录中的图片(.jpg/.jpeg/.png)
- 对于存在同名 JSON 的图片：解析 shapes(points)为矩形框，写入 YOLO 标签(txt)
- 对于没有 JSON 的图片：作为无标注样本包含进数据集（空标签文件）
- 按指定比例随机切分到 images/train、images/test 与 labels/train、labels/test
- 生成 classes.txt（按 JSON 中出现的 label 或由 --classes 指定）
- 生成 dataset.yaml（train 指向 images/train，val/test 指向 images/test）

使用示例：
  python prepare_yolo_dataset.py \
    --src ./data \
    --out ./train_yolo_dataset \
    --classes ./train_yolo_nums/classes.txt \
    --train-ratio 0.9 \
    --seed 42

说明：
- LabelMe JSON 的 shapes 支持 rectangle(两点) 与 polygon(多点)；统一取外接矩形
- YOLO 标签格式：<class_id> <cx> <cy> <w> <h>（均为归一化到[0,1]）
- 若未提供 classes.txt，则从 JSON 中汇总 label，并排序生成类列表
- YAML 中的 val 字段指向 test 目录，兼容主流 YOLO 训练脚本
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_classes(predefined: Optional[Path], discovered_labels: List[str]) -> List[str]:
    if predefined and predefined.exists():
        with open(predefined, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    uniq: List[str] = []
    seen = set()
    for l in discovered_labels:
        if l and l not in seen:
            uniq.append(l)
            seen.add(l)
    return sorted(uniq)


def parse_labelme_json(json_path: Path) -> Tuple[Optional[int], Optional[int], List[dict]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    width = data.get("imageWidth")
    height = data.get("imageHeight")
    shapes = data.get("shapes", [])
    return width, height, shapes


def points_to_bbox(points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min = int(min(xs))
    y_min = int(min(ys))
    x_max = int(max(xs))
    y_max = int(max(ys))
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int) -> Tuple[float, float, float, float]:
    # clip to image
    x_min = max(0, min(x_min, width - 1))
    y_min = max(0, min(y_min, height - 1))
    x_max = max(0, min(x_max, width - 1))
    y_max = max(0, min(y_max, height - 1))
    w = x_max - x_min
    h = y_max - y_min
    if w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0
    return cx / width, cy / height, w / width, h / height


def convert_shapes_to_yolo(
    img_path: Path,
    json_path: Optional[Path],
    labels_map: Dict[str, int],
) -> List[Tuple[int, float, float, float, float]]:
    """将一个图片对应的 LabelMe shapes 转为 YOLO 标签条目列表。
    返回：[(class_id, cx, cy, w, h), ...]
    若无 JSON 或解析失败，返回空列表。
    """
    # 读取图片尺寸（作为兜底）
    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"无法读取图片: {img_path}")
    H, W = img.shape[:2]

    items: List[Tuple[int, float, float, float, float]] = []
    if not json_path or not json_path.exists():
        return items

    try:
        jW, jH, shapes = parse_labelme_json(json_path)
    except Exception:
        return items

    if isinstance(jW, int) and isinstance(jH, int) and jW > 0 and jH > 0:
        W, H = jW, jH

    for sh in shapes:
        label = sh.get("label")
        pts = sh.get("points")
        shape_type = sh.get("shape_type", "polygon")
        if not label or not pts:
            continue
        points = [(float(x), float(y)) for x, y in pts]
        if shape_type == "rectangle" and len(points) >= 2:
            x_min = int(min(points[0][0], points[1][0]))
            y_min = int(min(points[0][1], points[1][1]))
            x_max = int(max(points[0][0], points[1][0]))
            y_max = int(max(points[0][1], points[1][1]))
        else:
            x_min, y_min, x_max, y_max = points_to_bbox(points)
        cx, cy, w, h = bbox_to_yolo(x_min, y_min, x_max, y_max, W, H)
        if w <= 0 or h <= 0:
            continue
        cls_id = labels_map.get(str(label))
        if cls_id is None:
            # 跳过未知类
            continue
        items.append((cls_id, cx, cy, w, h))
    return items


def write_txt_labels(out_dir: Path, stem: str, items: List[Tuple[int, float, float, float, float]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{stem}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls, cx, cy, w, h in items:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def discover_all_labels(src_dir: Path) -> List[str]:
    labels: List[str] = []
    for j in src_dir.glob("*.json"):
        try:
            _, _, shapes = parse_labelme_json(j)
            for sh in shapes:
                lab = sh.get("label")
                if lab:
                    labels.append(str(lab))
        except Exception:
            continue
    return labels


def build_dataset(src_dir: Path, out_root: Path, classes_path: Optional[Path], train_ratio: float, seed: int,
                  neg_ratio_train: float, neg_ratio_test: Optional[float] = None) -> None:
    # 1) 汇总图片列表
    images: List[Path] = []
    for p in src_dir.iterdir():
        if p.is_file() and p.suffix in IMG_EXTS:
            images.append(p)
    images.sort(key=lambda p: p.name)
    if not images:
        raise SystemExit(f"源目录没有图片：{src_dir}")

    # 2) 类名与映射
    discovered = discover_all_labels(src_dir)
    classes = load_classes(classes_path, discovered)
    labels_map: Dict[str, int] = {c: i for i, c in enumerate(classes)}

    # 保存 classes.txt
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "classes.txt", "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    # 3) 预转换，区分正负样本（有标注=正，无标注=负）
    rng = random.Random(seed)
    # 构建类映射
    discovered = discover_all_labels(src_dir)
    classes = load_classes(classes_path, discovered)
    labels_map: Dict[str, int] = {c: i for i, c in enumerate(classes)}

    # 保存 classes.txt
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "classes.txt", "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")

    items_map: Dict[Path, List[Tuple[int, float, float, float, float]]] = {}
    pos_images: List[Path] = []
    neg_images: List[Path] = []
    for img_path in images:
        stem = img_path.stem
        json_path = img_path.with_suffix(".json")
        items = convert_shapes_to_yolo(img_path, json_path if json_path.exists() else None, labels_map)
        items_map[img_path] = items
        if len(items) > 0:
            pos_images.append(img_path)
        else:
            neg_images.append(img_path)

    # 独立切分正样本与负样本
    rng.shuffle(pos_images)
    rng.shuffle(neg_images)
    n_pos_train = int(len(pos_images) * train_ratio)
    n_neg_train = int(len(neg_images) * train_ratio)
    pos_train = pos_images[:n_pos_train]
    pos_test = pos_images[n_pos_train:]
    neg_train_pool = neg_images[:n_neg_train]
    neg_test_pool = neg_images[n_neg_train:]

    # 目标负样本比例（每个 split 相对于正样本数量）
    if neg_ratio_test is None:
        neg_ratio_test = neg_ratio_train

    target_neg_train = int(round(len(pos_train) * neg_ratio_train))
    target_neg_test = int(round(len(pos_test) * neg_ratio_test))
    # 从 pool 中下采样到目标数量
    neg_train = neg_train_pool[:min(len(neg_train_pool), target_neg_train)]
    neg_test = neg_test_pool[:min(len(neg_test_pool), target_neg_test)]
    # 组合最终集合
    train_set = set(pos_train + neg_train)
    test_set = set(pos_test + neg_test)

    # 4) 目录结构
    img_train_dir = out_root / "images" / "train"
    img_test_dir = out_root / "images" / "test"
    lbl_train_dir = out_root / "labels" / "train"
    lbl_test_dir = out_root / "labels" / "test"
    for d in [img_train_dir, img_test_dir, lbl_train_dir, lbl_test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 5) 复制与写标签
    converted_cnt = 0
    for img_path in images:
        stem = img_path.stem
        items = items_map.get(img_path, [])
        if img_path in train_set:
            dst_img = img_train_dir / img_path.name
            shutil.copy2(str(img_path), str(dst_img))
            write_txt_labels(lbl_train_dir, stem, items)
            converted_cnt += 1
        elif img_path in test_set:
            dst_img = img_test_dir / img_path.name
            shutil.copy2(str(img_path), str(dst_img))
            write_txt_labels(lbl_test_dir, stem, items)
            converted_cnt += 1

    # 6) 生成 dataset.yaml（val 指向 test，以便训练使用）
    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {out_root.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/test\n")
        f.write("test: images/test\n")
        f.write("names:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {c}\n")

    print(f"转换完成：共处理 {converted_cnt} 张图片")
    print(f"输出：{out_root}")
    print(f"类别（{len(classes)}）：{classes}")
    print(f"Train: 正样本 {len(pos_train)}，负样本 {len(train_set) - len(pos_train)} / 目标 {target_neg_train}")
    print(f"Test:  正样本 {len(pos_test)}，负样本 {len(test_set) - len(pos_test)} / 目标 {target_neg_test}")


def parse_args():
    ap = argparse.ArgumentParser(description="将 LabelMe 数据集转换为 YOLO 并切分 train/test")
    ap.add_argument("--src", type=str, default="/home/advan/lh/diandulun/digital_rec/data/ori_data/car_crop_results", help="源数据目录，包含图片和同名 JSON")
    ap.add_argument("--out", type=str, default="/home/advan/lh/diandulun/digital_rec/data/ori_data/train_yolo_20260203", help="输出数据集根目录")
    ap.add_argument("--classes", type=str, default="/home/advan/lh/diandulun/digital_rec/data/ori_data/classes.txt", help="可选：预定义 classes.txt 路径")
    ap.add_argument("--train-ratio", type=float, default=0.9, help="训练集占比，其余为测试集")
    ap.add_argument("--neg-ratio", type=float, default=0.1, help="train 集中负样本(空标签)相对正样本的比例，例如 0.5 表示负样本数量=0.5*正样本")
    ap.add_argument("--neg-ratio-test", type=float, default=0.1, help="test 集中负样本比例；不填则与 --neg-ratio 相同")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    return ap.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src)
    out_root = Path(args.out)
    classes_path = Path(args.classes) if args.classes else None
    if not src_dir.exists():
        raise SystemExit(f"源目录不存在：{src_dir}")
    build_dataset(src_dir, out_root, classes_path, args.train_ratio, args.seed, args.neg_ratio, args.neg_ratio_test)


if __name__ == "__main__":
    main()
