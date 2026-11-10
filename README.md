# 学习笔记 · 遥感场景图生成方向

三峡大学 · 王文彬老师课题组（学习阶段）

## 这里放什么

- 每周学习笔记
- 命令备忘与踩坑记录
- 公开论文的阅读笔记

## 不放什么（安全红线）

- 组会 PPT 原文、老师未发表论文的细节、师兄的内部工作进展
- 服务器地址、账号、密码、API key
- 受限数据集的原始数据

## 目录

```
notes/        每周笔记（W1-W24）
```

## 笔记索引

| 文件 | 内容 |
|---|---|
| `notes/w1-环境与git.md` | Python 环境（venv 是什么）、tensor 是什么、git 四步、安全铁律 |
| `notes/w1-python补课.md` | 六项基础语法的真实进度（1 项已过 / 1 项半会 / 3 项未开始）+ 三个坑 |
| `notes/w1-linux命令速查.md` | ssh/scp/rsync、ps/top/df -h、tmux、tail -f、nvidia-smi（未实操的三项已标注） |
| `notes/w1-sarnet-reading.md` | SARNet 200 字理解笔记 + 四条自检|
| `notes/w2-python六项收尾.md` | Python 六项语法补完：三个必记点、两个编码坑、自测四问 |
| `notes/w2-cifar10训练循环.md` | **手写训练循环四步原理**（前向/损失/反向/更新）+ 实测 53.10% + 四个坑 |
| `notes/w2-l20五项补做.md` | L20 服务器五项实操：建目录/传代码/建环境/tmux/断线重连，含实机环境差异 |

## 代码

| 路径 | 内容 |
|---|---|
| `code/w2/train_cifar10.py` | CIFAR-10 手写训练循环（不依赖训练框架），实测验证准确率 53.10% |
| `code/w2/练习.py` | Python 六项语法练习，六关全通过 |
| `code/w2/原始成绩.csv` | 上面练习的输入（6 行成绩） |
| `code/w2/scores.csv` | 上面练习的产出（读 CSV → 排序 → 加等级列 → 写 CSV） |
| `code/w2/loss.csv` | CIFAR-10 三轮训练的 loss / 准确率记录 |
| `code/w2/loss_curve.txt` | 终端 ASCII loss 曲线（不依赖 matplotlib） |
| `code/w2/detect.py` | YOLO 目标检测脚本（W3 使用） |
| `notes/w1_practice.py` | 第一个跑通的脚本：认识 tensor 的 shape/dtype/device |

