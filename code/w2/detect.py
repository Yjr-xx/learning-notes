#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
W3 · 目标检测跑通（YOLO 一次推理 + 一张带框的图）
================================================================
对应：老师基础清单「CNN 与目标检测」/ 学习路线 W3「做 A」

先装依赖（约 200MB，装进项目自己的 .venv）：
    cd D:\\邵攀\\遥感AI入门
    .\\.venv\\Scripts\\python.exe -m pip install ultralytics

跑法：
    # 0. 先确认能跑：用官方自带样图
    .\\.venv\\Scripts\\python.exe W3-20261110\\02-yolo\\detect.py --check

    # 1. 跑自己的图（把要测的图放进 02-yolo\\images\\）
    .\\.venv\\Scripts\\python.exe W3-20261110\\02-yolo\\detect.py --source W3-20261110\\02-yolo\\images

    # 2. 换个模型看看差别（n 最小最快，x 最大最准）
    .\\.venv\\Scripts\\python.exe W3-20261110\\02-yolo\\detect.py --model yolov8s.pt --source <图片或目录>

产出：runs/detect/<时间戳>/ 下有画好框的图 + 一个 txt（每行：类别 置信度 左上x 左上y 右下x 右下y）

★ 这一步的验收不是"命令跑通了"，而是能回答三个问题：
   Q1 这张图里框出了几个目标？分别是什么类别、置信度多少？
   Q2 有个目标没被框出来 / 框错了，可能是什么原因？（太小 / 太挤 / 类别不在 COCO 80 类里 / 置信度阈值太高）
   Q3 把 --conf 从 0.25 调到 0.6，框是变多了还是变少了？为什么？
   —— 答不出 Q2 Q3，就只是"会敲命令"，不是"懂检测"。
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="YOLO 推理：出带框的图 + 每行检测结果")
    ap.add_argument("--model", default=None, help="权重文件路径；默认用本目录的 yolov8n.pt")
    ap.add_argument("--source", default=None, help="图片/目录路径；不给则用官方样图")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值，调高=框变少")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值，调低=去重更狠")
    ap.add_argument("--check", action="store_true", help="只验证环境能不能跑通")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("还没装 ultralytics。先跑：")
        print(r"  .\.venv\Scripts\python.exe -m pip install ultralytics")
        return 1

    import ultralytics
    print(f"[环境] ultralytics {ultralytics.__version__}")

    here = os.path.dirname(os.path.abspath(__file__))

    # 默认用本目录已有的权重，不去 GitHub 下载。
    # 2026-11-09 实测：这台机器直连 github.com 超时（梯子没覆盖命令行），
    # 权重和样图都是经 gh-proxy 镜像预先下好的。别改回自动下载。
    model_path = args.model or os.path.join(here, "yolov8n.pt")
    if not os.path.exists(model_path) and not os.path.isabs(model_path):
        model_path = os.path.join(here, args.model)

    source = args.source
    if args.check or source is None:
        # 本地样图；没有才回退到官方 URL（会走网络，可能失败）
        local_sample = os.path.join(here, "images", "bus.jpg")
        source = local_sample if os.path.exists(local_sample) else "https://ultralytics.com/images/bus.jpg"
        print(f"[模式] 检查模式，用样图：{source}")
    elif not os.path.isabs(source):
        source = os.path.join(os.getcwd(), source)

    print(f"[模型] {model_path}  conf={args.conf}  iou={args.iou}")
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        save=True,          # 把画好框的图存下来 —— 这就是周报要的那张图
        project=os.path.join(here, "runs"),
        name="detect",
        exist_ok=False,
        verbose=False,
    )

    names = model.names
    total = 0
    for i, r in enumerate(results, 1):
        boxes = r.boxes
        n = 0 if boxes is None else len(boxes)
        total += n
        print(f"\n── 结果 {i} ── 路径: {r.path}")
        print(f"   图片尺寸: {r.orig_shape}  检出目标: {n} 个")
        if n == 0:
            print("   （0 个框：可能是图里没有 COCO 80 类中的物体，或 conf 太高）")
            continue
        print(f"   {'类别':<12}{'置信度':>8}   框 (x1, y1, x2, y2)")
        for b in boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            label = names.get(cls_id, str(cls_id))
            print(f"   {label:<12}{conf:>8.3f}   ({x1}, {y1}, {x2}, {y2})")

    print(f"\n[合计] 共 {total} 个框")
    print(f"[产出] 带框的图在: {os.path.join(here, 'runs', 'detect')}")
    print("\n下一步（这段才是学东西的地方）：")
    print("  1. 打开那张图，逐个核对：框位置对不对、有没有漏检、有没有一个物体被框两次")
    print("  2. 换 --conf 0.6 再跑一次，对比框的数量变化")
    print("  3. 把这两个观察写进 W3-20261110/03-faster-rcnn/对比笔记.md 的「失败案例」栏")
    return 0


if __name__ == "__main__":
    sys.exit(main())
