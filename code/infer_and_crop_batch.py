#!/usr/bin/env python3
"""
Batch YOLO detection and crop generation over a folder of images.

- Uses `run_infer` from infer_and_crop.py for per-image processing
- Saves annotated images and crops under the provided save directory
- Writes a batch summary JSON with per-image detection counts and paths

Example:
  python infer_and_crop_batch.py \
    --weights runs/detect/train/weights/best.pt \
    --input-dir ../train1127_yolo/test/images \
    --save-dir ../runs/infer_batch \
    --conf 0.1

Requirements:
  - ultralytics (YOLOv8): pip install ultralytics
  - opencv-python: pip install opencv-python
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Reuse single-image inference
try:
    from infer_and_crop import run_infer
except Exception as e:
    print("Error importing run_infer from infer_and_crop.py. Ensure it is in the same folder.", file=sys.stderr)
    raise


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(input_dir: Path, recursive: bool = True) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        paths = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    # Sort for determinism
    return sorted(paths)


def parse_args():
    p = argparse.ArgumentParser(description="Batch YOLO infer and crop over a folder of images")
    p.add_argument("--weights", type=str, default="runs/detect/train/weights/best.pt", help="Path to YOLO weights (.pt)")
    p.add_argument("--input-dir", type=str, default="/home/advan/lh/diandulun/digital_rec/data/ori_data/train_yolo_2026/images/test", help="Directory containing test images")
    p.add_argument("--save-dir", type=str, default="../runs/infer_batch", help="Output directory for results")
    p.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    p.add_argument("--device", type=str, default="", help="Device string (''=auto, 'cpu', 'cuda:0')")
    p.add_argument("--recursive", action="store_true", help="Search images recursively in input-dir")
    p.add_argument("--max-count", type=int, default=0, help="Limit number of images to process (0 = no limit)")
    p.add_argument("--skip-existing", action="store_true", help="Skip if annotated image already exists in save-dir")
    return p.parse_args()


def main():
    args = parse_args()

    weights_p = Path(args.weights)
    if not weights_p.exists():
        print(f"Error: weights not found: {weights_p}", file=sys.stderr)
        sys.exit(2)

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input directory invalid: {input_dir}", file=sys.stderr)
        sys.exit(2)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Enumerate images
    imgs = list_images(input_dir, recursive=args.recursive)
    if args.max_count > 0:
        imgs = imgs[: args.max_count]

    if not imgs:
        print(f"No images found in: {input_dir}")
        sys.exit(0)

    print(f"Found {len(imgs)} image(s) in {input_dir}. Starting inference...")

    batch_summary: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "weights": str(weights_p.resolve()),
        "input_dir": str(input_dir.resolve()),
        "save_dir": str(save_dir.resolve()),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
        "count": 0,
        "images": [],
    }

    processed = 0
    errors = 0

    for img_path in imgs:
        img_stem = img_path.stem
        annotated_path = save_dir / f"{img_stem}_det.jpg"
        if args.skip_existing and annotated_path.exists():
            print(f"Skip existing: {annotated_path}")
            batch_summary["images"].append({
                "image": str(img_path.resolve()),
                "annotated_path": str(annotated_path.resolve()),
                "detections": [],
                "skipped": True,
            })
            processed += 1
            continue

        try:
            summary = run_infer(str(weights_p), str(img_path), str(save_dir), imgsz=args.imgsz, conf=args.conf, device=args.device)
            batch_summary["images"].append({
                "image": summary.get("image"),
                "annotated_path": summary.get("annotated_path"),
                "detections_count": len(summary.get("detections", [])),
                "detections": summary.get("detections", []),
            })
            processed += 1
        except Exception as e:
            print(f"Failed on {img_path}: {e}", file=sys.stderr)
            batch_summary["images"].append({
                "image": str(img_path.resolve()),
                "error": str(e),
            })
            errors += 1

    batch_summary["count"] = processed
    batch_summary["errors"] = errors

    # Write batch summary
    batch_json = save_dir / "batch_summary.json"
    with open(batch_json, "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"Batch complete: processed={processed}, errors={errors}")
    print(f"Batch summary: {batch_json}")


if __name__ == "__main__":
    main()
