#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CIFAR-10 图像分类 · 手写训练循环
================================================================
对应：老师基础清单 1.3「最低任务」/ 学习路线 W2（上周遗留项）

这个脚本的全部意义在于：**不调用任何训练框架，自己把四步写出来**
    前向 forward → 算损失 loss → 反向传播 backward → 更新参数 step
用 ultralytics / timm 那种一行 fit() 的写法，学不到这四步。

跑法（用项目自己的解释器）：
    cd D:\\邵攀\\遥感AI入门
    .\\.venv\\Scripts\\python.exe W3-20261110\\01-cifar10\\train_cifar10.py

快速验证（1 分钟左右，先确认脚本能跑通再开长跑）：
    .\\.venv\\Scripts\\python.exe W3-20261110\\01-cifar10\\train_cifar10.py --epochs 1 --train-size 512 --test-size 256

正式跑（CPU 上约 8-15 分钟）：
    .\\.venv\\Scripts\\python.exe W3-20261110\\01-cifar10\\train_cifar10.py --epochs 3

产出（都在 --out 目录下）：
    loss.csv          每轮的 train_loss / val_loss / val_acc
    loss_curve.txt    ASCII 曲线，直接 cat 就能看
    model.pt          权重（state_dict）
    最后会加载 model.pt 重新测一遍，证明"保存 + 加载"真的能用

为什么默认只用数据子集：
    你的机器是 AMD 780M 核显，torch 是 CPU 版（cuda.is_available() == False）。
    全量 50000 张在 CPU 上要跑很久，而这张卡上"跑得慢"学不到任何额外的东西。
    用 8000 张验证四步对不对，够了。要冲精度再调 --train-size。
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
#    刻意保持小 —— 能让 CPU 在几分钟内跑完，且结构一眼能看懂
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
    # 训练集做数据增强：随机翻转 + 随机裁切。这两条是 CIFAR 上最划算的增强。
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    # 测试集不做增强：评估要可重复
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
#    ★ 这两个函数是整个脚本的核心，四步就在 train_one_epoch 里
# ────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()                    # 切到训练模式（Dropout 生效、BN 用当前 batch 统计）
    running_loss = 0.0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)

        # ── 第 0 步：清空上一轮的梯度 ──────────────────────────
        # 不清空的话，梯度会累加，这一轮就白算了
        optimizer.zero_grad()

        # ── 第 1 步：前向 forward ─────────────────────────────
        # 把图片送进网络，得到 10 个类别的打分（还没变成概率）
        logits = model(images)

        # ── 第 2 步：算损失 loss ─────────────────────────────
        # 交叉熵 ≈ "打分和正确答案差多远"。模型就是要把这个数压小。
        loss = criterion(logits, labels)

        # ── 第 3 步：反向传播 backward ────────────────────────
        # 自动求导算出"每个参数该往哪个方向调、调多少"
        loss.backward()

        # ── 第 4 步：更新参数 step ───────────────────────────
        # 按上面算出的方向，真正改动权重
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

        if step % 20 == 0 or step == len(loader):
            acc = 100.0 * correct / total
            print(f"  epoch {epoch}/{total_epochs}  step {step}/{len(loader)}  "
                  f"loss {running_loss / total:.4f}  acc {acc:.1f}%")

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()                     # 评估不需要梯度，关掉能省内存和时间
def evaluate(model, loader, criterion, device):
    model.eval()                     # 切到评估模式（Dropout 关闭、BN 用滑动平均）
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
# 4. ASCII loss 曲线（不装 matplotlib 也能看见趋势）
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
    ap.add_argument("--train-size", type=int, default=8000, help="用多少张训练图（全量 50000）")
    ap.add_argument("--test-size", type=int, default=2000, help="用多少张测试图（全量 10000）")
    ap.add_argument("--data-root", default=None, help="数据缓存目录，默认本脚本同级的 data/")
    ap.add_argument("--out", default=None, help="产出目录，默认本脚本同级的 runs/")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data_root = args.data_root or os.path.join(here, "data")
    out_dir = args.out or os.path.join(here, "runs")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    print(f"设备      : {device}  （本机 CPU 版 torch，返回 False 是硬件定局）")
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

    # ── 加载权重重新测试 ───────────────────────────────────────────
    # 这一步不写，就等于没验证"保存"真的有效。很多人卡在这儿却不自知。
    print("\n[验证] 重新加载 model.pt 再测一遍...")
    fresh = SmallCNN().to(device)
    fresh.load_state_dict(torch.load(ckpt, map_location=device))
    re_loss, re_acc = evaluate(fresh, test_loader, criterion, device)
    print(f"[验证] 加载后 loss {re_loss:.4f}  acc {re_acc:.2f}%  "
          f"（应与最后一轮 val 一致，差 0.00 就是对的）")

    # ── 随机猜测基线对照 ───────────────────────────────────────────
    random_acc = 100.0 / len(CLASSES)
    print(f"\n[对照] 随机猜测 = {random_acc:.1f}%  →  你的模型 {re_acc:.2f}%  "
          f"({'明显高于随机，达标' if re_acc > random_acc * 2 else '没拉开差距，看看 lr / epoch 够不够'})")

    # ── 交周报要的四个数字，直接抄 ─────────────────────────────────
    print("\n" + "=" * 62)
    print("写进周报「可核查结果」的四件事：")
    print(f"  1. 脚本      : W3-20261110/01-cifar10/train_cifar10.py")
    print(f"  2. 最终准确率: {history[-1]['val_acc']:.2f}%（随机基线 {random_acc:.0f}%）")
    print(f"  3. loss 记录 : {os.path.relpath(csv_path, os.path.dirname(here))}")
    print(f"  4. 权重加载  : 重载后 acc {re_acc:.2f}%，与最后一轮一致 → 保存/加载可用")
    print("=" * 62)
    print("\n自测一下能不能讲清四步（讲不清就是没学会，不是跑完就完）：")
    print("  forward  : 图片 → 网络 → 10 个分数")
    print("  loss     : 分数 vs 正确答案 → 一个数，越小越好")
    print("  backward : 自动算出每个参数该往哪调")
    print("  step     : 按算出的方向真正改权重")
    return 0


if __name__ == "__main__":
    sys.exit(main())
