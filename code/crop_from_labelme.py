#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遍历指定目录(默认: /home/advan/lh/diandulun/digital_rec/train1127)，
对每个有同名 JSON(Labelme) 的 JPG 图片：
  - 读取 JSON 中的 shapes，获取 label 和 points
  - 以 label 作为子目录名创建文件夹
  - 按 points 对图片进行裁剪并保存到对应 label 目录

使用示例：
  python crop_from_labelme.py \
    --src /home/advan/lh/diandulun/digital_rec/train1127 \
    --out /home/advan/lh/diandulun/digital_rec/train1127_crops

说明：
  - 默认仅处理与 JSON 配对的 JPG；其它图片跳过
  - 支持 shape_type 为 rectangle(points 两点) 或 多点 polygon(自动取外接框)
  - 自动裁剪边界到图片尺寸范围内；无效框会跳过
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import cv2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _bbox_from_points(points: List[List[float]]) -> Tuple[int, int, int, int]:
    """从 Labelme points 计算外接矩形，返回 (x1,y1,x2,y2) 整数像素坐标。
    - rectangle 时，points 为两点 [lt, rb]
    - polygon 时，points 多点，取 min/max
    - 坐标做 floor/ceil，确保覆盖区域
    """
    if not points:
        return 0, 0, 0, 0
    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        left = min(x1, x2)
        top = min(y1, y2)
        right = max(x1, x2)
        bottom = max(y1, y2)
    else:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        left, right = min(xs), max(xs)
        top, bottom = min(ys), max(ys)

    # floor 左上，ceil 右下
    x1 = int(math.floor(left))
    y1 = int(math.floor(top))
    x2 = int(math.ceil(right))
    y2 = int(math.ceil(bottom))
    return x1, y1, x2, y2


def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1c = max(0, min(x1, w - 1))
    y1c = max(0, min(y1, h - 1))
    x2c = max(0, min(x2, w))
    y2c = max(0, min(y2, h))
    return x1c, y1c, x2c, y2c


def process_one(img_path: Path, json_path: Path, out_root: Path) -> int:
    # 读图
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图片：{img_path}")
        return 0
    h, w = img.shape[:2]

    # 读 JSON (Labelme)
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"读取 JSON 失败：{json_path} -> {e}")
        return 0

    shapes = data.get("shapes", [])
    saved = 0
    stem = img_path.stem

    for idx, sh in enumerate(shapes):
        label = str(sh.get("label", "_unknown")).strip()
        points = sh.get("points", []) or []
        x1, y1, x2, y2 = _bbox_from_points(points)
        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)

        if x2 <= x1 or y2 <= y1:
            # 无效框，跳过
            continue

        crop = img[y1:y2, x1:x2]
        # 以 label 建目录
        out_dir = out_root / label
        ensure_dir(out_dir)
        out_name = f"{stem}_{idx:02d}_{x1}_{y1}_{x2}_{y2}.jpg"
        out_path = out_dir / out_name
        ok = cv2.imwrite(str(out_path), crop)
        if ok:
            saved += 1
        else:
            print(f"保存失败：{out_path}")

    return saved


def run(src_dir: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    total_imgs = 0
    total_crops = 0

    # 同时支持 .jpg/.jpeg/.JPG
    patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    jpgs: List[Path] = []
    for pat in patterns:
        jpgs.extend(src_dir.glob(pat))

    # 去重
    jpgs = sorted({p.resolve() for p in jpgs})

    for jp in jpgs:
        json_path = jp.with_suffix(".json")
        if not json_path.exists():
            # 无 JSON，跳过
            continue
        total_imgs += 1
        saved = process_one(jp, json_path, out_dir)
        total_crops += saved
        print(f"处理 {jp.name} -> 保存裁剪 {saved} 张")

    print("—— 处理完成 ——")
    print(f"配对图片数：{total_imgs}")
    print(f"总裁剪数：{total_crops}")
    print(f"输出目录：{out_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="按 Labelme JSON 裁剪并按 label 归档")
    ap.add_argument(
        "--src",
        type=str,
        default="/home/advan/lh/diandulun/digital_rec/data/ori_data/car_crop_results",
        help="源目录，包含 JPG 与同名 JSON",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="/home/advan/lh/diandulun/digital_rec/data/ori_data/nums",
        help="输出根目录(按 label 建子目录)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    if not src_dir.exists():
        print(f"源目录不存在：{src_dir}")
        raise SystemExit(2)
    run(src_dir, out_dir)
