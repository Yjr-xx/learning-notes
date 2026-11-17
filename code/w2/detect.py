#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLO 推理：输出带框图与逐目标检测结果

依赖：ultralytics（pip install ultralytics）

用法：
    python detect.py --check                      # 用内置样图跑一次
    python detect.py --source images             # 跑指定目录或单张图
    python detect.py --model yolov8s.pt          # 换权重

输出：runs/detect/<时间戳>/ 下是画好框的图，另有 txt 记录
      每行「类别 置信度 左上x 左上y 右下x 右下y」
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="YOLO 推理：输出带框图与逐目标检测结果")
    ap.add_argument("--model", default=None, help="权重文件路径；默认用本目录的 yolov8n.pt")
    ap.add_argument("--source", default=None, help="图片/目录路径；不给则用内置样图")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值，调高则保留的框变少")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS 的 IoU 阈值，调低则去重更严格")
    ap.add_argument("--check", action="store_true", help="用内置样图跑一次，确认环境可用")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("缺少依赖 ultralytics，安装后重试：")
        print(r"  pip install ultralytics")
        return 1

    import ultralytics
    print(f"[环境] ultralytics {ultralytics.__version__}")

    here = os.path.dirname(os.path.abspath(__file__))

    # 权重与样图已用镜像预先下载到本目录。直连 github.com 会超时，
    # 因此不走自动下载，默认读本地文件。
    model_path = args.model or os.path.join(here, "yolov8n.pt")
    if not os.path.exists(model_path) and not os.path.isabs(model_path):
        model_path = os.path.join(here, args.model)

    source = args.source
    if args.check or source is None:
        local_sample = os.path.join(here, "images", "bus.jpg")
        source = local_sample if os.path.exists(local_sample) else "https://ultralytics.com/images/bus.jpg"
        print(f"[模式] 使用样图：{source}")
    elif not os.path.isabs(source):
        source = os.path.join(os.getcwd(), source)

    print(f"[模型] {model_path}  conf={args.conf}  iou={args.iou}")
    model = YOLO(model_path)

    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        save=True,          # 保存带框结果图
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
            print("   （0 个框：图像中可能不含 COCO 80 类物体，或 conf 阈值偏高）")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
