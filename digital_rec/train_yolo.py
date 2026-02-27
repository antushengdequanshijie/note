#!/usr/bin/env python3
"""
Train YOLOv8 on the dataset converted by data_process.py.

Prereqs:
  pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

Usage:
  python train.py --data /home/advan/lh/diandulun/digital_rec/train1127_yolo/dataset.yaml \
                  --model yolov8n.pt \
                  --epochs 50 --imgsz 1024 --batch 16 --device 0

Outputs:
  Runs will be saved under ./runs/detect/trainX
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    ap = argparse.ArgumentParser(description='Train YOLOv8 on custom dataset')
    ap.add_argument('--data', type=str, default='/home/advan/lh/diandulun/digital_rec/data/ori_data/train_yolo_20260203/dataset.yaml', help='path to dataset.yaml')
    ap.add_argument('--model', type=str, default='yolov8n.pt', help='base model weights or cfg')
    ap.add_argument('--epochs', type=int, default=5000, help='number of training epochs')
    ap.add_argument('--imgsz', type=int, default=1024, help='training image size')
    ap.add_argument('--batch', type=int, default=16, help='batch size')
    ap.add_argument('--device', type=str, default='0', help='device, ""=auto, "0"=GPU0')
    ap.add_argument('--workers', type=int, default=4, help='number of data loader workers')
    ap.add_argument('--project', type=str, default='runs/detect', help='project directory for outputs')
    ap.add_argument('--name', type=str, default='train', help='run name')
    ap.add_argument('--exist_ok', action='store_true', help='overwrite existing runs with same name')
    args = ap.parse_args()

    # 检查 dataset.yaml 是否存在
    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f'dataset.yaml not found: {data_path}')

    # 加载 YOLOv8 模型
    model = YOLO(args.model)

    # ====== 训练 ======
    print(f"[INFO] Start training YOLOv8 on dataset: {data_path}")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=True,
        # 工业场景小目标建议：
        # mosaic=0.0 可以关闭 mosaic 数据增强，避免小目标过小
        # rect=True 可以按比例训练（保持长宽比）
        mosaic=0.0,
        rect=True,
    )
    print('[INFO] Training finished. Results saved in:', results)

    # ====== 验证 ======
    print('[INFO] Start validation on the same dataset...')
    model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        device=args.device,
    )

    # ====== 可选：导出模型 ======
    # 支持 torchscript / onnx / coreml / tensorflow
    # print('[INFO] Exporting model to ONNX...')
    # model.export(format='onnx', imgsz=args.imgsz)

if __name__ == '__main__':
    main()
