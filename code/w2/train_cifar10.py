#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 图像分类 · 手写训练循环

不调用任何训练框架，把四步显式写出来：
    前向 forward → 算损失 loss → 反向传播 backward → 更新参数 step

用法：
    python train_cifar10.py
    python train_cifar10.py --epochs 1 --train-size 512 --test-size 256   # 快速验证链路
    python train_cifar10.py --epochs 3                                    # 正式跑

产出（均在 --out 目录下）：
    loss.csv          每轮的 train_loss / val_loss / val_acc
    loss_curve.txt    ASCII 曲线
    model.pt          权重（state_dict），训练结束会重新加载再测一遍

关于默认数据子集：
    本机为 CPU 环境（cuda.is_available() 为 False），全量 5 万张耗时过长。
    默认取 8000 张训练、2000 张测试，足够验证四步实现是否正确；
    需要更高精度时调大 --train-size。
"""

import argparse
import csv
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import torchvision
from torchvision import transforms

# Windows 上多进程 DataLoader 容易出问题（spawn 启动方式），统一用 0
NUM_WORKERS = 0

CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ────────────────────────────────────────────────────────────────
# 1. 模型：一个小型 CNN
#    规模刻意保持小：CPU 上几分钟可跑完，结构也便于逐层核对
# ────────────────────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 输入 3×32×32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # → 32×32×32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # → 64×32×32
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # → 128×32×32
        self.pool = nn.MaxPool2d(2)                                 # 每池化一次边长减半
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)                # 32→16→8→4

    def forward(self, x):
        # 每一块都是：卷积 → 归一化 → 激活 → 池化
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32×16×16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 64×8×8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 128×4×4
        x = x.flatten(1)                                  # 摊平成向量
        x = self.dropout(x)
        return self.fc(x)                                 # 10 个类别的分数(logits)


# ────────────────────────────────────────────────────────────────
# 2. 数据
# ────────────────────────────────────────────────────────────────
def build_loaders(data_root, train_size, test_size, batch_size):
    # 训练集做数据增强：随机翻转 + 随机裁切
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    # 测试集不做增强：评估需要可重复
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    print(f"[数据] 准备 CIFAR-10（首次运行会下载约 170MB 到 {data_root}）...")
    train_full = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_tf)
    test_full = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_tf)

    train_ds = Subset(train_full, range(min(train_size, len(train_full))))
    test_ds = Subset(test_full, range(min(test_size, len(test_full))))
    print(f"[数据] 训练 {len(train_ds)} 张 / 测试 {len(test_ds)} 张")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS),
    )


# ────────────────────────────────────────────────────────────────
# 3. 训练一轮 / 评估一次
#    四步的实现都在 train_one_epoch 里
# ────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()                    # 训练模式：Dropout 生效、BN 用当前 batch 统计
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        # ── 第 0 步：清空上一轮的梯度 ──────────────────────────
        # 不清空则梯度会累加，本轮更新会偏离
        optimizer.zero_grad()

        # ── 第 1 步：前向 forward ─────────────────────────────
        # 图片送入网络，得到 10 个类别的打分（尚未归一化为概率）
        logits = model(images)

        # ── 第 2 步：算损失 loss ─────────────────────────────
        # 交叉熵衡量打分与真实标签的差距，训练目标即最小化该值
        loss = criterion(logits, labels)

        # ── 第 3 步：反向传播 backward ────────────────────────
        # 自动求导计算每个参数的梯度方向与大小
        loss.backward()

        # ── 第 4 步：更新参数 step ───────────────────────────
        # 按梯度方向真正修改权重
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        if step % 20 == 0 or step == len(loader):
            acc = 100.0 * correct / total
            print(f"  epoch {epoch}/{total_epochs}  step {step}/{len(loader)}  "
                  f"loss {running_loss / total:.4f}  acc {acc:.1f}%")

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()                     # 评估不需要梯度，关闭以省内存与时间
def evaluate(model, loader, criterion, device):
    model.eval()                     # 评估模式：Dropout 关闭、BN 用滑动平均
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        running_loss += criterion(logits, labels).item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return running_loss / total, 100.0 * correct / total


# ────────────────────────────────────────────────────────────────
# 4. ASCII loss 曲线（不依赖 matplotlib）
# ────────────────────────────────────────────────────────────────
def ascii_curve(values, width=48, height=10, title="val_loss"):
    if not values:
        return "(无数据)"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        hi = lo + 1e-9
    rows = []
    for r in range(height, -1, -1):
        threshold = lo + (hi - lo) * r / height
        line = ""
        for i in range(width):
            idx = min(len(values) - 1, int(i * len(values) / width))
            line += "●" if values[idx] >= threshold - 1e-12 else " "
        rows.append(f"  {threshold:7.4f} |{line}")
    rows.append("          └" + "─" * width)
    rows.append(f"           {title}：{len(values)} 个点，{values[0]:.4f} → {values[-1]:.4f}")
    return "\n".join(rows)


# ────────────────────────────────────────────────────────────────
# 5. 主流程
# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="手写 CIFAR-10 训练循环")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-size", type=int, default=8000, help="训练图数量（全量 50000）")
    ap.add_argument("--test-size", type=int, default=2000, help="测试图数量（全量 10000）")
    ap.add_argument("--data-root", default=None, help="数据缓存目录，默认本脚本同级的 data/")
    ap.add_argument("--out", default=None, help="产出目录，默认本脚本同级的 runs/")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(here, "data")
    out_dir = args.out or os.path.join(here, "runs")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    print(f"设备      : {device}")
    print(f"torch     : {torch.__version__}")
    print(f"轮数/子集 : {args.epochs} epoch / train {args.train_size} / test {args.test_size}")
    print(f"产出目录  : {out_dir}")
    print("=" * 62)

    train_loader, test_loader = build_loaders(
        data_root, args.train_size, args.test_size, args.batch_size)

    model = SmallCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] SmallCNN，参数量 {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\n──── epoch {epoch}/{args.epochs} ────")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs)
        va_loss, va_acc = evaluate(model, test_loader, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 5),
            "train_acc": round(tr_acc, 2),
            "val_loss": round(va_loss, 5),
            "val_acc": round(va_acc, 2),
        })
        print(f"  → train loss {tr_loss:.4f} acc {tr_acc:.2f}%  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.2f}%")

    elapsed = time.time() - t0
    print(f"\n训练完成，用时 {elapsed:.1f}s")

    # ── 保存权重 ───────────────────────────────────────────────────
    ckpt = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[保存] state_dict → {ckpt}  ({os.path.getsize(ckpt) / 1024 / 1024:.1f} MB)")

    # ── 记录 CSV ───────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "loss.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"[记录] {csv_path}")

    # ── ASCII 曲线 ─────────────────────────────────────────────────
    curve = ascii_curve([h["val_loss"] for h in history], title="val_loss（每轮一个点）")
    curve_txt = os.path.join(out_dir, "loss_curve.txt")
    with open(curve_txt, "w", encoding="utf-8") as f:
        f.write(curve + "\n")
    print("\n" + curve)
    print(f"[曲线] {curve_txt}")

    # ── 重新加载权重再测一遍，确认保存/加载链路可用 ────────────────
    print("\n[验证] 重新加载 model.pt 再测一遍...")
    fresh = SmallCNN().to(device)
    fresh.load_state_dict(torch.load(ckpt, map_location=device))
    re_loss, re_acc = evaluate(fresh, test_loader, criterion, device)
    print(f"[验证] 加载后 loss {re_loss:.4f}  acc {re_acc:.2f}%  "
          f"（应与最后一轮 val 一致）")

    # ── 随机猜测基线对照 ───────────────────────────────────────────
    random_acc = 100.0 / len(CLASSES)
    gap = "高于随机基线的 2 倍" if re_acc > random_acc * 2 else "未明显高于随机基线"
    print(f"\n[对照] 随机猜测 = {random_acc:.1f}%  →  模型 {re_acc:.2f}%，{gap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
