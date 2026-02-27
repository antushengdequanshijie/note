#!/usr/bin/env python3
"""
Run YOLO detection on a single image, save annotated result and per-box crops.

Usage examples:
  python infer_and_crop.py \
    --weights ../runs/detect/train/weights/best.pt \
    --img ../test_data/roi.jpg \
    --save-dir ../runs/infer

Notes:
  - Requires `ultralytics` (YOLOv8). Install if needed: pip install ultralytics
  - Crops are saved under: <save-dir>/crops/<image_stem>_idx_label_conf.jpg
  - A JSON summary is written next to the annotated image.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

import cv2


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_model(weights: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Error: ultralytics is not installed. Please: pip install ultralytics", file=sys.stderr)
        raise
    return YOLO(weights)


def xyxy_to_xywh(x1, y1, x2, y2):
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return x1, y1, w, h


def run_infer(weights: str, img_path: str, save_dir: str, imgsz: int = 640, conf: float = 0.25, device: str = "") -> Dict[str, Any]:
    model = load_model(weights)
    names = model.names if hasattr(model, "names") else None

    # Run prediction
    results = model.predict(source=img_path, imgsz=imgsz, conf=conf, device=device, save=False, verbose=False)
    if not results:
        raise RuntimeError("No result returned by model.predict")
    res = results[0]

    # Prepare paths
    save_root = Path(save_dir)
    ensure_dir(save_root)
    crops_dir = save_root / "crops"
    ensure_dir(crops_dir)

    img_stem = Path(img_path).stem
    annotated_path = save_root / f"{img_stem}_det.jpg"
    summary_path = save_root / f"{img_stem}_det.json"

    # Save annotated image
    annotated = res.plot()  # BGR numpy array
    cv2.imwrite(str(annotated_path), annotated)

    # Use original input image from results to avoid re-read
    im = res.orig_img
    h, w = im.shape[:2]

    detections = []
    boxes = getattr(res, "boxes", None)

    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        print(f"No detections. Annotated saved to: {annotated_path}")
    else:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            # clip to image
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            cls_id = int(boxes.cls[i].item()) if hasattr(boxes, "cls") else -1
            conf_i = float(boxes.conf[i].item()) if hasattr(boxes, "conf") else 0.0
            label = None
            if names is not None:
                try:
                    label = names[cls_id] if isinstance(names, (list, tuple)) else names.get(cls_id, str(cls_id))
                except Exception:
                    label = str(cls_id)

            # Crop and save
            crop = im[y1:y2, x1:x2]
            crop_name = f"{img_stem}_{i}_{label if label is not None else cls_id}_{conf_i:.2f}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            # Collect summary entry
            xywh = list(xyxy_to_xywh(x1, y1, x2, y2))
            det = {
                "index": i,
                "cls": cls_id,
                "label": label,
                "conf": round(conf_i, 6),
                "xyxy": [x1, y1, x2, y2],
                "xywh": xywh,
                "crop_path": str(crop_path),
            }
            detections.append(det)

    summary = {
        "image": str(Path(img_path).resolve()),
        "width": w,
        "height": h,
        "weights": str(Path(weights).resolve()),
        "imgsz": imgsz,
        "conf": conf,
        "device": device,
        "annotated_path": str(annotated_path),
        "detections": detections,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved annotated: {annotated_path}")
    print(f"Saved summary:   {summary_path}")
    print(f"Saved {len(detections)} crop(s) to: {crops_dir}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="YOLO infer single image and crop detections")
    p.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", help="Path to YOLO weights (.pt)")
    p.add_argument("--img", type=str, default="/home/advan/lh/diandulun/digital_rec/train1127_yolo/test/images/192.168.1.66_01_20251030092042432_MD_WITH_TARGET.jpg", help="Path to input image")
    p.add_argument("--save-dir", type=str, default="runs/infer", help="Output directory")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    p.add_argument("--device", type=str, default="", help="Device string (''=auto, 'cpu', 'cuda:0')")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.weights).exists():
        print(f"Error: weights not found: {args.weights}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.img).exists():
        print(f"Error: image not found: {args.img}", file=sys.stderr)
        sys.exit(2)
    run_infer(args.weights, args.img, args.save_dir, imgsz=args.imgsz, conf=args.conf, device=args.device)
