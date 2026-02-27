#!/usr/bin/env python3
"""
Convert a folder of JPGs + Labelme JSON annotations into a YOLO detection dataset.

Source folder (default): ../train1127
Outputs to: ../train1127_yolo/

What it does:
- Scans the source directory for .jpg/.jpeg/.png images
- For each image, tries to find a paired Labelme .json (same stem). If missing, copies image and creates empty label file.
- Parses Labelme JSON shapes: for polygon/rectangle points, computes bounding boxes.
- Writes YOLO txt labels: <class_id> <cx> <cy> <w> <h> (normalized)
- Builds classes.txt and a dataset.yaml compatible with YOLO frameworks.

Usage:
  python data_process.py --src ../train1127 --out ../train1127_yolo --classes classes.txt

Notes:
- Class names come from JSON "label" fields. By default we collect all labels and sort to build classes.
- You can provide a predefined classes.txt to fix class ordering.
- Unsupported shapes are skipped; segmentation is not exported, only detection boxes.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2


def load_classes(predefined: Optional[Path], labels_found: List[str]) -> List[str]:
	if predefined and predefined.exists():
		with open(predefined, 'r', encoding='utf-8') as f:
			classes = [line.strip() for line in f if line.strip()]
		return classes
	# Build from discovered labels (sorted for stability)
	uniq = []
	seen = set()
	for l in labels_found:
		if l not in seen:
			uniq.append(l)
			seen.add(l)
	return sorted(uniq)


def points_to_bbox(points: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
	xs = [p[0] for p in points]
	ys = [p[1] for p in points]
	x_min = int(min(xs))
	y_min = int(min(ys))
	x_max = int(max(xs))
	y_max = int(max(ys))
	return x_min, y_min, x_max, y_max


def bbox_to_yolo(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int) -> Tuple[float, float, float, float]:
	# clip
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


def parse_labelme_json(json_path: Path) -> Tuple[str, int, int, List[Dict]]:
	with open(json_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	# Labelme keys: imagePath, imageWidth, imageHeight, shapes
	image_path = data.get('imagePath')
	width = data.get('imageWidth')
	height = data.get('imageHeight')
	shapes = data.get('shapes', [])
	return image_path, width, height, shapes


def convert_one(image_path: Path, json_path: Optional[Path], labels_map: Dict[str, int]) -> List[Tuple[int, float, float, float, float]]:
	"""Return YOLO labels for one image from Labelme shapes."""
	img = cv2.imread(str(image_path))
	if img is None:
		raise RuntimeError(f"Cannot read image: {image_path}")
	H, W = img.shape[:2]
	yolo_items: List[Tuple[int, float, float, float, float]] = []
	if not json_path or not json_path.exists():
		return yolo_items
	_, jW, jH, shapes = parse_labelme_json(json_path)
	# if json has width/height, prefer them if plausible
	if isinstance(jW, int) and isinstance(jH, int) and jW > 0 and jH > 0:
		W, H = jW, jH
	for sh in shapes:
		label = sh.get('label')
		pts = sh.get('points')
		shape_type = sh.get('shape_type', 'polygon')
		if not label or not pts:
			continue
		# normalize points to (x,y) tuples
		points = [(float(x), float(y)) for x, y in pts]
		# rectangle in labelme might specify two points (top-left, bottom-right)
		if shape_type == 'rectangle' and len(points) >= 2:
			x_min = int(min(points[0][0], points[1][0]))
			y_min = int(min(points[0][1], points[1][1]))
			x_max = int(max(points[0][0], points[1][0]))
			y_max = int(max(points[0][1], points[1][1]))
		else:
			x_min, y_min, x_max, y_max = points_to_bbox(points)
		cx, cy, w, h = bbox_to_yolo(x_min, y_min, x_max, y_max, W, H)
		if w <= 0 or h <= 0:
			continue
		cls_id = labels_map.get(label)
		if cls_id is None:
			# skip unknown class
			continue
		yolo_items.append((cls_id, cx, cy, w, h))
	return yolo_items


def build_labels_map(src_dir: Path, predefined_classes: Optional[Path]) -> List[str]:
	found_labels: List[str] = []
	for j in src_dir.glob('*.json'):
		try:
			_, _, _, shapes = parse_labelme_json(j)
			for sh in shapes:
				lab = sh.get('label')
				if lab:
					found_labels.append(lab)
		except Exception:
			continue
	classes = load_classes(predefined_classes, found_labels)
	return classes


def write_txt_labels(out_labels_dir: Path, stem: str, items: List[Tuple[int, float, float, float, float]]):
	out_labels_dir.mkdir(parents=True, exist_ok=True)
	txt_path = out_labels_dir / f"{stem}.txt"
	with open(txt_path, 'w', encoding='utf-8') as f:
		for cls, cx, cy, w, h in items:
			f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument('--src', type=str, default=str(Path(__file__).resolve().parents[1] / 'train1127'), help='source folder with images and labelme jsons')
	ap.add_argument('--out', type=str, default=str(Path(__file__).resolve().parents[1] / 'train1127_yolo'), help='output yolo dataset root')
	ap.add_argument('--classes', type=str, default='classes.txt', help='optional classes.txt to fix class order')
	ap.add_argument('--split', type=float, default=0.9, help='train split ratio (rest goes to val)')
	args = ap.parse_args()

	src_dir = Path(args.src)
	out_root = Path(args.out)
	classes_path = Path(args.classes) if args.classes else None

	images_dir = out_root / 'images'
	labels_dir = out_root / 'labels'
	images_dir.mkdir(parents=True, exist_ok=True)
	labels_dir.mkdir(parents=True, exist_ok=True)

	# Build classes
	classes = build_labels_map(src_dir, classes_path)
	if not classes:
		print('WARNING: No classes found; labels will be empty unless classes.txt provided.')
	labels_map: Dict[str, int] = {c: i for i, c in enumerate(classes)}

	# Save classes.txt
	with open(out_root / 'classes.txt', 'w', encoding='utf-8') as f:
		for c in classes:
			f.write(c + '\n')

	# Iterate images (only include those that have paired JSON)
	img_exts = {'.jpg', '.jpeg', '.png'}
	items: List[Tuple[Path, Path]] = []
	for img in src_dir.iterdir():
		if img.is_file() and img.suffix.lower() in img_exts:
			j = img.with_suffix('.json')
			if j.exists():
				items.append((img, j))

	# Process and copy only items with JSON
	index_rows = []
	for img_path, json_path in sorted(items, key=lambda t: t[0].name):
		stem = img_path.stem
		# copy image to images/
		dst_img = images_dir / img_path.name
		shutil.copy2(str(img_path), str(dst_img))
		# convert labels
		yolo_items = convert_one(img_path, json_path, labels_map)
		write_txt_labels(labels_dir, stem, yolo_items)
		index_rows.append((dst_img.name, len(yolo_items)))

	# Optional: create dataset.yaml (train/val same dir for simplicity; adjust as needed)
	yaml_path = out_root / 'dataset.yaml'
	with open(yaml_path, 'w', encoding='utf-8') as f:
		f.write('path: {}\n'.format(out_root.as_posix()))
		f.write('train: images\n')
		f.write('val: images\n')
		f.write('names:\n')
		for i, c in enumerate(classes):
			f.write(f"  {i}: {c}\n")

	# Summary
	print(f"Converted {len(index_rows)} images (with JSON) to YOLO at {out_root}")
	print('Classes:', classes)
	print('Example labels saved under', labels_dir)


if __name__ == '__main__':
	main()

