#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像分类训练脚本：数据目录结构为 ImageFolder 格式

data/
  ├── 0/
  │    ├── img1.jpg
  │    └── ...
  ├── 1/
  └── ...  (文件夹名即类别标签)

使用示例：
  python train_classify.py \
    --data ../nums \
    --epochs 20 \
    --batch 64 \
    --model resnet18 \
    --imgsz 224 \
    --lr 1e-3 \
    --out ../runs/classify

依赖：torch, torchvision, opencv-python (可选，若用cv2读取增强则需要)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def add_gaussian_noise(t, std=0.02):
    if not torch.is_floating_point(t):
        t = t.float()
    noise = torch.randn_like(t) * std
    t = t + noise
    return t.clamp(0.0, 1.0)
def build_transforms(imgsz: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        # 若不希望翻转，注释掉下一行
        # transforms.RandomHorizontalFlip(p=0.0),

        transforms.RandomApply([  
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomAutocontrast(),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5)
        ], p=0.4),
        transforms.RandomRotation(degrees=8),
        transforms.RandomAffine(
            degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=(-4, 4)
        ),
        transforms.RandomPerspective(distortion_scale=0.06, p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),

        transforms.ToTensor(),
        transforms.Lambda(lambda t: add_gaussian_noise(t, std=0.01)),  # 轻微噪声
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.06), ratio=(0.3, 3.3), inplace=True),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
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


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k.mul_(100.0 / batch_size)).item())
    return res


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        acc1 = accuracy(outputs.detach(), labels, topk=(1,))[0]
        running_loss += loss.item() * imgs.size(0)
        running_acc += acc1 * imgs.size(0)
    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            acc1 = accuracy(outputs, labels, topk=(1,))[0]
            running_loss += loss.item() * imgs.size(0)
            running_acc += acc1 * imgs.size(0)
    return running_loss / len(loader.dataset), running_acc / len(loader.dataset)


def main():
    ap = argparse.ArgumentParser(description="ImageFolder 图像分类训练")
    ap.add_argument("--data", type=str, default="nums/nums_aug", help="数据目录(子文件夹名为标签)")
    ap.add_argument("--epochs", type=int, default=200, help="训练轮数")
    ap.add_argument("--batch", type=int, default=64, help="批大小")
    ap.add_argument("--model", type=str, default="resnet18", help="模型名称(resnet18/resnet34/resnet50/mobilenet_v3_small/efficientnet_b0)")
    ap.add_argument("--imgsz", type=int, default=224, help="输入尺寸")
    ap.add_argument("--lr", type=float, default=1e-3, help="学习率")
    ap.add_argument("--weight-decay", type=float, default=1e-4, help="权重衰减")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader 线程数")
    ap.add_argument("--out", type=str, default="../runs/classify", help="输出目录")
    ap.add_argument("--device", type=str, default="cuda:0", help="设备字符串(空=自动, 'cpu', 'cuda:0')")
    ap.add_argument("--resume", type=str, default="../runs/classify/last.pt", help="从保存的 checkpoint (.pt) 继续训练的路径")
    ap.add_argument("--no-pretrained", action="store_true", help="禁用 ImageNet 预训练（用于从 checkpoint 继续训练）")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    # 数据集与划分（简单做法：按 9:1 随机划分）
    train_tf, val_tf = build_transforms(args.imgsz)
    full_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
    num_classes = len(full_ds.classes)
    # 打印类别映射
    with open(out_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(full_ds.classes)}, f, ensure_ascii=False, indent=2)

    # 随机划分训练/验证
    n_total = len(full_ds)
    n_train = int(n_total * 0.9)
    n_val = n_total - n_train
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    # 验证集需要用 val_tf
    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    # 模型（支持从 checkpoint 继续训练）
    ckpt = None
    model_name = args.model
    use_pretrained = not args.no_pretrained and not args.resume
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location="cpu")
        model_name = ckpt.get("model", args.model)
        # 提示 imgsz 可能不一致
        if "imgsz" in ckpt and ckpt["imgsz"] != args.imgsz:
            print(f"[warn] checkpoint imgsz={ckpt['imgsz']} 与当前 imgsz={args.imgsz} 不一致，继续使用当前 imgsz。")
        use_pretrained = False  # 继续训练时不再加载 ImageNet 预训练

    model = build_model(model_name, num_classes=num_classes, pretrained=use_pretrained)
    if ckpt is not None:
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[resume] missing keys: {sorted(missing)[:10]}{' ...' if len(missing)>10 else ''}")
        if unexpected:
            print(f"[resume] unexpected keys: {sorted(unexpected)[:10]}{' ...' if len(unexpected)>10 else ''}")
        # 类别映射提示
        if "classes" in ckpt:
            ckpt_classes = ckpt["classes"]
            if list(ckpt_classes) != list(full_ds.classes):
                print("[warn] checkpoint 类别映射与当前数据集不一致，已按当前数据集类别训练；final layer 可能未完全对齐。")
        print(f"[resume] Loaded checkpoint from {ckpt_path} using model '{model_name}'.")

    model = model.to(device)

    # 优化器与调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    # 可选恢复优化器/调度器（若保存了）
    if ckpt is not None:
        opt_sd = ckpt.get("optimizer")
        sch_sd = ckpt.get("scheduler")
        if opt_sd:
            try:
                optimizer.load_state_dict(opt_sd)
                print("[resume] Optimizer state restored.")
            except Exception as e:
                print(f"[warn] Failed to restore optimizer: {e}")
        if sch_sd:
            try:
                scheduler.load_state_dict(sch_sd)
                print("[resume] Scheduler state restored.")
            except Exception as e:
                print(f"[warn] Failed to restore scheduler: {e}")

    best_acc = 0.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:03d}: train_loss={tl:.4f} acc={ta:.2f} | val_loss={vl:.4f} acc={va:.2f}")

        # 保存最优
        if va > best_acc:
            best_acc = va
            torch.save({
                "model": args.model,
                "state_dict": model.state_dict(),
                "classes": full_ds.classes,
                "imgsz": args.imgsz,
            }, best_path)
            print(f"Saved best to {best_path} (acc={best_acc:.2f})")

    # 最终保存最后一轮
    last_path = out_dir / "last.pt"
    torch.save({
        "model": args.model,
        "state_dict": model.state_dict(),
        "classes": full_ds.classes,
        "imgsz": args.imgsz,
    }, last_path)
    print(f"Saved last to {last_path}")


if __name__ == "__main__":
    main()
