#!/usr/bin/env python3
"""
Crop big boxes from a frame, then run detect+classify on each crop
using the existing functions in infer_detect_and_classify.py.

Typical usage in a video pipeline:
  - You already have `frame` (numpy BGR image) and `big_boxes`.
  - Provide timestamp `ts` and `frame_id` to build the base filename.
  - This script saves the full frame and each crop, runs detection+classification
    per crop, and aggregates results into a combined summary JSON.

Dependencies:
  - OpenCV (cv2)
  - ultralytics, torch, torchvision (required by infer_detect_and_classify)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2

# Reuse the existing detect+classify utilities
from .infer_detect_and_classify import detect_and_classify, ensure_dir


def _clip_box_to_image(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    return x1, y1, x2, y2


def process_frame_crops(
    frame,  # numpy.ndarray (H, W, 3) in BGR
    big_boxes: Sequence[Sequence[int]],
    ts: str,
    frame_id: int,
    save_dir: str,
    det_weights: str = "../runs/detect/train/weights/best.pt",
    cls_weights: str = "../runs/classify/last.pt",
    det_imgsz: int = 640,
    det_conf: float = 0.1,
    device: str = "",
) -> Dict[str, Any]:
    """
    - Saves the full frame as frame_{ts}_{frame_id}.jpg under save_dir
    - For each (x1,y1,x2,y2) in big_boxes:
        * crops the region, saves to save_dir/frames/<stem>_box{i}.jpg
        * runs detect+classify on that crop
        * writes per-crop outputs under save_dir/det_cls/<stem>/box_{i}
    - Returns an aggregated summary dict and writes it to save_dir/<stem>_crops_summary.json
    """
    H, W = frame.shape[:2]

    save_root = Path(save_dir)
    ensure_dir(save_root)

    # Save the original frame once, following the provided naming pattern
    frame_name = f"frame_{ts}_{frame_id}.jpg"
    frame_path = save_root / frame_name
    cv2.imwrite(str(frame_path), frame)

    stem = frame_path.stem

    frames_dir = save_root / "frames"
    ensure_dir(frames_dir)

    all_entries: List[Dict[str, Any]] = []

    for i, box in enumerate(big_boxes):
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = _clip_box_to_image(x1, y1, x2, y2, W, H)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        crop_name = f"{stem}_box{i}.jpg"
        crop_path = frames_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        # Use a dedicated output directory per crop to avoid collisions
        out_dir = save_root / "det_cls" / stem / f"box_{i}"
        ensure_dir(out_dir)

        # Run detect+classify on the saved crop
        summary = detect_and_classify(
            det_weights=det_weights,
            cls_weights=cls_weights,
            img_path=str(crop_path),
            save_dir=str(out_dir),
            imgsz=det_imgsz,
            conf=det_conf,
            device=device,
        )

        entry = {
            "index": i,
            "box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
            "crop_path": str(crop_path),
            "output_dir": str(out_dir),
            "summary": summary,
        }
        all_entries.append(entry)

    combined = {
        "frame_path": str(frame_path),
        "ts": ts,
        "frame_id": frame_id,
        "image_size": {"width": W, "height": H},
        "det_weights": os.path.abspath(det_weights),
        "cls_weights": os.path.abspath(cls_weights),
        "entries": all_entries,
    }

    combined_path = save_root / f"{stem}_crops_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    return combined


def _parse_boxes(boxes_json: str | None, boxes_path: str | None) -> List[List[int]]:
    if boxes_json:
        try:
            data = json.loads(boxes_json)
            if isinstance(data, list):
                return [list(map(int, b)) for b in data]
        except Exception:
            pass
    if boxes_path and Path(boxes_path).exists():
        with open(boxes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [list(map(int, b)) for b in data]
    return []


def main():
    p = argparse.ArgumentParser(description="Crop big boxes from an image and run detect+classify per crop")
    p.add_argument("--image", type=str, required=True, help="Path to input image (as frame)")
    p.add_argument("--save-dir", type=str, default="../runs/infer", help="Output directory root")
    p.add_argument("--ts", type=str, required=True, help="Timestamp string for naming")
    p.add_argument("--frame-id", type=int, required=True, help="Frame id for naming")
    p.add_argument("--boxes-json", type=str, default=None, help="JSON string of boxes [[x1,y1,x2,y2], ...]")
    p.add_argument("--boxes-file", type=str, default=None, help="Path to JSON file with boxes list")
    p.add_argument("--det-weights", type=str, default="../runs/detect/train/weights/best.pt")
    p.add_argument("--cls-weights", type=str, default="../runs/classify/last.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.1)
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    im = cv2.imread(args.image)
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {args.image}")

    boxes = _parse_boxes(args.boxes_json, args.boxes_file)
    if not boxes:
        # If no boxes provided, process the full image as a single box
        h, w = im.shape[:2]
        boxes = [[0, 0, w, h]]

    combined = process_frame_crops(
        frame=im,
        big_boxes=boxes,
        ts=args.ts,
        frame_id=args.frame_id,
        save_dir=args.save_dir,
        det_weights=args.det_weights,
        cls_weights=args.cls_weights,
        det_imgsz=args.imgsz,
        det_conf=args.conf,
        device=args.device,
    )

    print(json.dumps({"summary_path": str(Path(args.save_dir) / f"frame_{args.ts}_{args.frame_id}_crops_summary.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
