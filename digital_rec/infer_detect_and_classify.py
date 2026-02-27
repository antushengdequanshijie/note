#!/usr/bin/env python3
"""
Detect-then-classify on a single image.

Pipeline:
  1) Use YOLO (Ultralytics) to detect targets
  2) Crop each detection and classify with a trained classifier checkpoint
  3) Save annotated image, per-crop images, and a JSON summary

Defaults assume:
  - Detection weights at ../runs/detect/train/weights/best.pt
  - Classification checkpoint at ../runs/classify/best.pt

Requirements:
  - ultralytics (for YOLOv8)
  - torch, torchvision
  - opencv-python
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2


def _import_ultralytics():
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print("Error: ultralytics is not installed. Please: pip install ultralytics", file=sys.stderr)
        raise
    return YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, int, int]:
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return x1, y1, w, h


def get_ordered_digits(detections: List[Dict[str, Any]]) -> List[str]:
    """Return a list of `cls_label` values ordered left-to-right by box center x."""
    items: List[Tuple[float, str]] = []
    for d in detections:
        try:
            x1, y1, x2, y2 = d.get("xyxy", [0, 0, 0, 0])
            cx = (float(x1) + float(x2)) / 2.0
            items.append((cx, str(d.get("cls_label", ""))))
        except Exception:
            continue
    items.sort(key=lambda t: t[0])
    return [label for _, label in items]


def build_cls_model(name: str, num_classes: int, pretrained: bool = False):
    import torch.nn as nn
    from torchvision import models

    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "resnet34":
        m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model: {name}")


def load_cls_checkpoint(ckpt_path: Path, device_str: str = ""):
    import torch
    from torchvision import transforms
    from PIL import Image

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Classification checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model_name = ckpt.get("model", "resnet18")
    classes = ckpt.get("classes", None)
    if classes is None:
        # Fallback to indices if not present
        num_classes = ckpt.get("num_classes", None)
        if num_classes is None:
            raise RuntimeError("Checkpoint missing 'classes' and 'num_classes'.")
        classes = [str(i) for i in range(num_classes)]
    num_classes = len(classes)
    imgsz = int(ckpt.get("imgsz", 224))

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_cls_model(model_name, num_classes=num_classes, pretrained=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[cls] missing keys: {sorted(missing)[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[cls] unexpected keys: {sorted(unexpected)[:10]}{' ...' if len(unexpected)>10 else ''}")
    model.eval()
    model.to(device)

    # Inference transform aligned with validation pipeline
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def classify_crop_bgr(crop_bgr) -> Tuple[int, str, float]:
        # Convert to PIL RGB
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        x = val_tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            p, idx = prob.max(dim=1)
            idx = int(idx.item())
            p = float(p.item())
            label = classes[idx] if isinstance(classes, (list, tuple)) else str(idx)
        return idx, label, p

    return model, classes, imgsz, classify_crop_bgr


def detect_and_classify(
    det_weights: str,
    cls_weights: str,
    img_path: str,
    save_dir: str,
    imgsz: int = 640,
    conf: float = 0.1,
    device: str = "",
) -> Dict[str, Any]:
    """Run YOLO detection then classify each crop. Returns summary dict."""
    YOLO = _import_ultralytics()
    det_model = YOLO(det_weights)

    # Load classification checkpoint and callable
    _, classes, cls_imgsz, classify_crop_bgr = load_cls_checkpoint(Path(cls_weights), device)
    frame_image = cv2.imread(img_path)
    results = det_model.predict(source=frame_image, imgsz=imgsz, conf=conf, device=device, save=False, verbose=False)
    if not results:
        raise RuntimeError("No result returned by model.predict")
    res = results[0]

    save_root = Path(save_dir)
    ensure_dir(save_root)
    crops_dir = save_root / "crops"
    ensure_dir(crops_dir)

    img_stem = Path(img_path).stem
    ann_det_path = save_root / f"{img_stem}_det_cls.jpg"
    summary_path = save_root / f"{img_stem}_det_cls.json"

    # original image
    im = res.orig_img
    H, W = im.shape[:2]

    detections: List[Dict[str, Any]] = []
    boxes = getattr(res, "boxes", None)

    # We'll annotate ourselves with classification labels
    annotated = im.copy()

    if boxes is None or boxes.xyxy is None or len(boxes) == 0:
        print(f"No detections. Saving original as: {ann_det_path}")
    else:
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
            x1, y1, x2, y2 = xyxy
            # clip to image
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                continue

            det_conf = float(boxes.conf[i].item()) if hasattr(boxes, "conf") else 0.0
            # crop
            crop = im[y1:y2, x1:x2]
            # classify
            cls_idx, cls_label, cls_prob = classify_crop_bgr(crop)

            crop_name = f"{img_stem}_{i}_{cls_label}_{cls_prob:.2f}.jpg"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)

            # draw box + label
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = f"{cls_label} {cls_prob:.2f} | det {det_conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = max(y1 - 5, th + 5)
            cv2.rectangle(annotated, (x1, ty - th - baseline), (x1 + tw + 4, ty + baseline), (0, 0, 0), -1)
            cv2.putText(annotated, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            det = {
                "index": i,
                "xyxy": [x1, y1, x2, y2],
                "xywh": list(xyxy_to_xywh(x1, y1, x2, y2)),
                "det_conf": round(det_conf, 6),
                "cls_idx": cls_idx,
                "cls_label": cls_label,
                "cls_prob": round(cls_prob, 6),
                "crop_path": str(crop_path),
            }
            detections.append(det)

    # save annotated image
    cv2.imwrite(str(ann_det_path), annotated)

    summary = {
        "image": str(Path(img_path).resolve()),
        "width": W,
        "height": H,
        "det_weights": str(Path(det_weights).resolve()),
        "det_imgsz": imgsz,
        "det_conf": conf,
        "device": device,
        "cls_weights": str(Path(cls_weights).resolve()),
        "cls_imgsz": cls_imgsz,
        "annotated_path": str(ann_det_path),
        "detections": detections,
    }

    # Order detected digits left-to-right, combine into a single integer inside a list
    ordered_labels = get_ordered_digits(detections)
    ordered_str = "".join([d for d in ordered_labels if str(d).isdigit()])
    ordered_digits = [int(ordered_str)] if ordered_str else []
    summary["ordered_digits"] = ordered_digits

    digits_path = save_root / f"{img_stem}_digits.json"
    digits_summary = {"image": Path(img_path).name, "ordered_digits": ordered_digits}
    with open(digits_path, "w", encoding="utf-8") as f:
        json.dump(digits_summary, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved annotated: {ann_det_path}")
    print(f"Saved summary:   {summary_path}")
    print(f"Saved {len(detections)} crop(s) to: {crops_dir}")
    return summary


def _gather_images(img_dir: Path, pattern: str = "*.jpg", recursive: bool = False) -> List[Path]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"] if pattern == "*" else [pattern]
    files: List[Path] = []
    for ext in exts:
        if recursive:
            files.extend(img_dir.rglob(ext))
        else:
            files.extend(img_dir.glob(ext))
    # unique and sorted
    files = sorted({p.resolve() for p in files})
    return files


def detect_and_classify_batch(
    det_weights: str,
    cls_weights: str,
    img_dir: str,
    save_dir: str,
    imgsz: int = 640,
    conf: float = 0.1,
    device: str = "",
    pattern: str = "*",
    recursive: bool = False,
) -> Dict[str, Any]:
    """Process all images in a folder (optionally recursively).
    Returns a batch summary with per-image entries.
    """
    YOLO = _import_ultralytics()
    det_model = YOLO(det_weights)
    _, classes, cls_imgsz, classify_crop_bgr = load_cls_checkpoint(Path(cls_weights), device)

    img_dir_p = Path(img_dir)
    if not img_dir_p.exists() or not img_dir_p.is_dir():
        raise FileNotFoundError(f"Image directory not found or not a directory: {img_dir}")

    save_root = Path(save_dir)
    ensure_dir(save_root)
    crops_dir = save_root / "crops"
    ensure_dir(crops_dir)

    files = _gather_images(img_dir_p, pattern=pattern, recursive=recursive)
    if not files:
        raise RuntimeError(f"No images matched in {img_dir} with pattern='{pattern}' recursive={recursive}")

    batch_entries: List[Dict[str, Any]] = []

    for img_path in files:
        # Run detection
        results = det_model.predict(source=str(img_path), imgsz=imgsz, conf=conf, device=device, save=False, verbose=False)
        if not results:
            print(f"[warn] No result for {img_path}")
            continue
        res = results[0]

        img_stem = img_path.stem
        ann_det_path = save_root / f"{img_stem}_det_cls.jpg"
        summary_path = save_root / f"{img_stem}_det_cls.json"

        im = res.orig_img
        H, W = im.shape[:2]
        boxes = getattr(res, "boxes", None)
        annotated = im.copy()
        detections: List[Dict[str, Any]] = []

        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            print(f"[info] No detections in {img_path}")
        else:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
                x1, y1, x2, y2 = xyxy
                x1 = max(0, min(x1, W - 1))
                y1 = max(0, min(y1, H - 1))
                x2 = max(0, min(x2, W))
                y2 = max(0, min(y2, H))
                if x2 <= x1 or y2 <= y1:
                    continue

                det_conf = float(boxes.conf[i].item()) if hasattr(boxes, "conf") else 0.0
                crop = im[y1:y2, x1:x2]
                cls_idx, cls_label, cls_prob = classify_crop_bgr(crop)

                crop_name = f"{img_stem}_{i}_{cls_label}_{cls_prob:.2f}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop)

                color = (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                text = f"{cls_label} {cls_prob:.2f} | det {det_conf:.2f}"
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                ty = max(y1 - 5, th + 5)
                cv2.rectangle(annotated, (x1, ty - th - baseline), (x1 + tw + 4, ty + baseline), (0, 0, 0), -1)
                cv2.putText(annotated, text, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                det = {
                    "index": i,
                    "xyxy": [x1, y1, x2, y2],
                    "xywh": list(xyxy_to_xywh(x1, y1, x2, y2)),
                    "det_conf": round(det_conf, 6),
                    "cls_idx": cls_idx,
                    "cls_label": cls_label,
                    "cls_prob": round(cls_prob, 6),
                    "crop_path": str(crop_path),
                }
                detections.append(det)

        cv2.imwrite(str(ann_det_path), annotated)

        summary = {
            "image": str(img_path),
            "width": W,
            "height": H,
            "det_weights": str(Path(det_weights).resolve()),
            "det_imgsz": imgsz,
            "det_conf": conf,
            "device": device,
            "cls_weights": str(Path(cls_weights).resolve()),
            "cls_imgsz": cls_imgsz,
            "annotated_path": str(ann_det_path),
            "detections": detections,
        }

        # Order digits, combine into a single integer in a list, and save JSON
        ordered_labels = get_ordered_digits(detections)
        ordered_str = "".join([d for d in ordered_labels if str(d).isdigit()])
        ordered_digits = [int(ordered_str)] if ordered_str else []
        summary["ordered_digits"] = ordered_digits
        digits_path = save_root / f"{img_stem}_digits.json"
        digits_summary = {"image": img_path.name, "ordered_digits": ordered_digits}
        with open(digits_path, "w", encoding="utf-8") as f:
            json.dump(digits_summary, f, ensure_ascii=False, indent=2)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        batch_entries.append(summary)

    batch_summary = {
        "image_dir": str(Path(img_dir).resolve()),
        "count": len(batch_entries),
        "det_weights": str(Path(det_weights).resolve()),
        "cls_weights": str(Path(cls_weights).resolve()),
        "save_dir": str(Path(save_dir).resolve()),
        "entries": batch_entries,
    }

    with open(Path(save_dir) / "summary_batch.json", "w", encoding="utf-8") as f:
        json.dump(batch_summary, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(batch_entries)} image(s). Batch summary: {Path(save_dir) / 'summary_batch.json'}")
    return batch_summary


def parse_args():
    p = argparse.ArgumentParser(description="Detect-then-classify a single image")
    p.add_argument("--det-weights", type=str, default="../runs/detect/train/weights/best.pt", help="YOLO detection weights path")
    p.add_argument("--cls-weights", type=str, default="../runs/classify/last.pt", help="Classification checkpoint path (.pt)")
    p.add_argument("--img", type=str, default="../train1127_yolo/test/images/192.168.1.66_01_20251030092042432_MD_WITH_TARGET.jpg", help="Path to input image")
    p.add_argument("--img-dir", type=str, default="../train1127_yolo/train/images", help="Process all images in a folder if provided")
    p.add_argument("--pattern", type=str, default="*", help="Glob pattern for images (e.g., *.jpg | * for common types)")
    p.add_argument("--recursive", action="store_true", help="Search images recursively in --img-dir")
    p.add_argument("--save-dir", type=str, default="../runs/infer_classify", help="Output directory")
    p.add_argument("--imgsz", type=int, default=640, help="Detection image size")
    p.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold")
    p.add_argument("--device", type=str, default="", help="Device string (''=auto, 'cpu', 'cuda:0')")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.det_weights).exists():
        print(f"Error: detection weights not found: {args.det_weights}", file=sys.stderr)
        sys.exit(2)
    if not Path(args.cls_weights).exists():
        print(f"Error: classification checkpoint not found: {args.cls_weights}", file=sys.stderr)
        sys.exit(2)
    # Folder mode if --img-dir provided
    if args.img_dir:
        if not Path(args.img_dir).exists():
            print(f"Error: image directory not found: {args.img_dir}", file=sys.stderr)
            sys.exit(2)
        detect_and_classify_batch(
            args.det_weights,
            args.cls_weights,
            args.img_dir,
            args.save_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            pattern=args.pattern,
            recursive=args.recursive,
        )
    else:
        if not Path(args.img).exists():
            print(f"Error: image not found: {args.img}", file=sys.stderr)
            sys.exit(2)
        detect_and_classify(args.det_weights, args.cls_weights, args.img, args.save_dir, imgsz=args.imgsz, conf=args.conf, device=args.device)
