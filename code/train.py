#!/usr/bin/env python3
"""
Train YOLOv8 on the dataset converted by data_process.py.

Prereqs:
  pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

Usage:
  python train.py --data /home/advan/lh/diandulun/digital_rec/train1127_yolo/dataset.yaml \
                  --model yolov8n.pt \
                  --epochs 50 --imgsz 640 --batch 16 --device 0

Outputs:
  Runs will be saved under ./runs/detect/trainX
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='../train1127_yolo/dataset.yaml', help='path to dataset.yaml')
    ap.add_argument('--model', type=str, default='yolov8n.pt', help='base model weights or cfg')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='')  # '' for auto, or '0' for GPU 0
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--project', type=str, default='runs/detect', help='project dir')
    ap.add_argument('--name', type=str, default='train', help='run name')
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f'dataset.yaml not found: {data_path}')

    model = YOLO(args.model)
    # Train
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    print('Training finished. Results dir:', results)

    # Optional: Validate
    model.val(data=str(data_path), imgsz=args.imgsz, device=args.device)

    # Optional: Export to ONNX
    # model.export(format='onnx')


if __name__ == '__main__':
    main()
