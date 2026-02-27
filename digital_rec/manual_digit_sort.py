#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式图片分类脚本：显示图片，按键 0~9 移动到对应子目录。

功能：
- 遍历源目录中的图片
- 显示图片（按任意键等待）
- 按 0~9：移动到输出目录下对应的子目录（如 out/0, out/1, ... out/9）
- 按 s：跳过
- 按 u：撤销上一次移动（仅限本次会话）
- 按 q：退出

依赖：opencv-python

示例：
  python manual_digit_sort.py \
    --src /path/to/images \
    --out /path/to/sorted \
    --recursive

可选：使用 --copy 切换为复制而非移动； --max-side 控制显示尺寸上限
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


def list_images(src: Path, recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in src.rglob("*") if p.suffix in IMG_EXTS]
    else:
        files = [p for p in src.iterdir() if p.is_file() and p.suffix in IMG_EXTS]
    files = sorted(files)
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    i = 1
    while True:
        alt = parent / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
        i += 1


def resize_to_max_side(img, max_side: int):
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def overlay_help(img, text: str):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 1
    color = (0, 255, 255)
    y0 = 24
    for i, line in enumerate(text.split("\n")):
        y = y0 + i * 22
        cv2.putText(out, line, (10, y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), font, scale, color, thickness, cv2.LINE_AA)
    return out


def write_log(log_csv: Path, row: List[str]):
    exists = log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["time", "action", "src", "dst", "label"])
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description="按键 0~9 将图片移动到对应子目录")
    ap.add_argument("--src", default="/home/advan/lh/diandulun/digital_rec/runs/infer_classify/crops" , help="源图片目录")
    ap.add_argument("--out", default="/home/advan/lh/diandulun/digital_rec/runs/infer_classify/crops0", help="输出根目录（会自动创建 0~9 子目录）")
    ap.add_argument("--recursive", action="store_true", help="递归遍历源目录")
    ap.add_argument("--copy", action="store_true", help="复制而非移动")
    ap.add_argument("--max-side", type=int, default=1200, help="显示最大边长度")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out_root = Path(args.out)
    if not src_dir.exists():
        print(f"源目录不存在：{src_dir}")
        sys.exit(2)
    ensure_dir(out_root)
    for d in map(str, range(10)):
        ensure_dir(out_root / d)

    files = list_images(src_dir, args.recursive)
    if not files:
        print("未找到图片")
        return

    log_csv = out_root / "sort_log.csv"
    print("操作说明：\n  0~9: 移动到对应子目录  s: 跳过  u: 撤销上一次移动  q: 退出")

    # 撤销栈：保存 (dst, src) 用于回退；copy 模式下保存 (dst, None) 意味着删除 dst
    undo_stack: List[Tuple[Path, Path | None]] = []

    win = "digit-sort"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for idx, p in enumerate(files, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"跳过无法读取：{p}")
            continue

        # 显示原图：不缩放、不叠加任何文字
        cv2.imshow(win, img)

        while True:
            k = cv2.waitKey(0) & 0xFF
            if k in (ord('q'), 27):  # q or ESC
                cv2.destroyAllWindows()
                return
            if k in (ord('s'), ord('S')):
                # 跳过
                break
            if k in (ord('u'), ord('U')):
                if undo_stack:
                    dst, src = undo_stack.pop()
                    try:
                        if src is None:
                            # copy 模式撤销：删除目标文件
                            if dst.exists():
                                dst.unlink()
                            write_log(log_csv, [datetime.now().isoformat(), "undo_delete", "", str(dst), ""])
                        else:
                            # move 模式撤销：移回原处
                            ensure_dir(src.parent)
                            if dst.exists():
                                shutil.move(str(dst), str(src))
                            write_log(log_csv, [datetime.now().isoformat(), "undo_move", str(dst), str(src), ""])
                        print("已撤销上一操作")
                    except Exception as e:
                        print(f"撤销失败：{e}")
                else:
                    print("无可撤销操作")
                # 继续等待按键，不跳到下一张
                continue

            # 数字键 0~9
            if ord('0') <= k <= ord('9'):
                label = chr(k)
                dst_dir = out_root / label
                ensure_dir(dst_dir)
                dst = dst_dir / p.name
                dst = unique_path(dst)
                try:
                    if args.copy:
                        shutil.copy2(str(p), str(dst))
                        undo_stack.append((dst, None))  # 撤销时删除 dst
                        action = "copy"
                    else:
                        shutil.move(str(p), str(dst))
                        undo_stack.append((dst, p))  # 撤销时移回 p
                        action = "move"
                    print(f"{action}: {p.name} -> {label}/")
                    write_log(log_csv, [datetime.now().isoformat(), action, str(p), str(dst), label])
                except Exception as e:
                    print(f"操作失败：{e}")
                break  # 下一张

            # 其他键，继续等待
            print("按 0~9 分类，s 跳过，u 撤销，q 退出")

    cv2.destroyAllWindows()
    print("完成")


if __name__ == "__main__":
    main()
