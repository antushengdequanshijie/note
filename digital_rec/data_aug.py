#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================
Image Augmentation Script
------------------------------------------------
功能：
- 概率式图像增强：模糊 / 扭曲 / 亮度变化 / 对比度变化
- 随机灰度噪声背景生成
- 前景与背景 Alpha 融合，模拟纸张与成像噪声
- 批量生成增强数据集，保持类别目录结构

依赖：
- opencv-python
- numpy

运行方式：
python image_augmentation.py
================================================
"""

import os
import cv2
import numpy as np
import random

# =========================
# 参数配置
# =========================
SRC_DIR = "nums/nums_perspective_aug"         # 原始数据集
DST_DIR = "nums/nums_aug"     # 增强后数据集
AUG_PER_IMAGE = 5        # 每张图像生成的增强数量

BLUR_PROB = 0.3
WARP_PROB = 0.3
BC_PROB = 0.5

IMG_EXTS = (".jpg", ".png", ".jpeg")

# =========================
# Alpha 掩码生成
# =========================
def generate_alpha_mask(img, thresh=245):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    alpha = mask.astype(np.float32) / 255.0
    return alpha

# =========================
# 随机灰度噪声背景
# =========================
def generate_noise_background(h, w):
    mean = random.randint(200, 245)
    var = random.randint(10, 40)

    noise = np.random.normal(mean, var, (h, w))
    noise = np.clip(noise, 0, 255).astype(np.uint8)

    bg = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    return bg

# =========================
# Alpha 融合
# =========================
def alpha_blend(fg, bg, alpha):
    alpha = np.expand_dims(alpha, axis=2)
    blended = fg * alpha + bg * (1 - alpha)
    return blended.astype(np.uint8)

# =========================
# 模糊增强
# =========================
def random_blur(img, p=0.3):
    if random.random() < p:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img

# =========================
# 扭曲增强（仿射）
# =========================
def random_warp(img, p=0.3):
    if random.random() > p:
        return img

    h, w = img.shape[:2]
    dx = w * 0.05
    dy = h * 0.05

    src = np.float32([
        [0, 0],
        [w, 0],
        [0, h]
    ])

    dst = np.float32([
        [random.uniform(0, dx), random.uniform(0, dy)],
        [w - random.uniform(0, dx), random.uniform(0, dy)],
        [random.uniform(0, dx), h - random.uniform(0, dy)]
    ])

    M = cv2.getAffineTransform(src, dst)
    warped = cv2.warpAffine(
        img, M, (w, h),
        borderMode=cv2.BORDER_REFLECT
    )
    return warped

# =========================
# 亮度 / 对比度变化
# =========================
def random_brightness_contrast(img, p=0.5):
    if random.random() > p:
        return img

    alpha = random.uniform(0.8, 1.2)   # 对比度
    beta = random.randint(-20, 20)     # 亮度
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img

# =========================
# 单张图像增强
# =========================
def enhance_image(img):
    h, w = img.shape[:2]

    alpha = generate_alpha_mask(img)
    bg = generate_noise_background(h, w)
    img = alpha_blend(img, bg, alpha)

    img = random_blur(img, BLUR_PROB)
    img = random_warp(img, WARP_PROB)
    img = random_brightness_contrast(img, BC_PROB)

    return img

# =========================
# 批量数据集增强
# =========================
def augment_dataset(src_dir, dst_dir, aug_per_img):
    os.makedirs(dst_dir, exist_ok=True)

    for cls in os.listdir(src_dir):
        src_cls_dir = os.path.join(src_dir, cls)
        if not os.path.isdir(src_cls_dir):
            continue

        dst_cls_dir = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        print(f"[INFO] Processing class: {cls}")

        for fname in os.listdir(src_cls_dir):
            if not fname.lower().endswith(IMG_EXTS):
                continue

            img_path = os.path.join(src_cls_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            base = os.path.splitext(fname)[0]

            for i in range(aug_per_img):
                aug_img = enhance_image(img)
                save_name = f"{base}_aug_{i:03d}.jpg"
                save_path = os.path.join(dst_cls_dir, save_name)
                cv2.imwrite(save_path, aug_img)

# =========================
# 主函数入口
# =========================
def main():
    print("=== Image Augmentation Started ===")
    augment_dataset(SRC_DIR, DST_DIR, AUG_PER_IMAGE)
    print("=== Image Augmentation Finished ===")

if __name__ == "__main__":
    main()
