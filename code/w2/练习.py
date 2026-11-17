#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 基础补完 · 六个语法点 + 读写文件小工具

覆盖：
    ① 变量与类型   ② list 与 dict   ③ for / if
    ④ 函数         ⑤ import         ⑥ 读写文件

用法：
    python 练习.py

运行后逐项打印结果并做断言校验，最后产出 scores.csv。
"""

import csv
import os
import sys

# Windows 控制台默认 GBK，打印 ✅ 这类符号会抛 UnicodeEncodeError。
# 这两行强制走 UTF-8，否则后面的输出会中断。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # 旧版 Python 没有这个方法，退化为直接打印

OK = "✅"
HERE = os.path.dirname(os.path.abspath(__file__))


def hr(title):
    print("\n" + "─" * 58)
    print(f"  {title}")
    print("─" * 58)


# ══════════════════════════════════════════════════════════
# ① 变量与类型
# ══════════════════════════════════════════════════════════
hr("① 变量与类型")
a = 5
b = "5"
print(f'a = {a!r}   类型 {type(a).__name__}')
print(f'b = {b!r}   类型 {type(b).__name__}')
print(f'a + a = {a + a}     ← 数字相加，按算术计算')
print(f'b + b = {b + b!r}   ← 字符串相加，是拼接')
try:
    a + b
except TypeError as e:
    print(f'a + b 报错：{e}   ← 类型不同不能直接相加')

int_of_b = int(b)          # 字符串转数字
print(f'int(b) = {int_of_b}  ← 转换后可以参与运算：{a + int_of_b}')
assert a + a == 10 and b + b == "55" and a + int_of_b == 10
print(f"{OK} 第 1 项通过")


# ══════════════════════════════════════════════════════════
# ② list 与 dict
# ══════════════════════════════════════════════════════════
hr("② list 与 dict")
scores = [88, 92, 75, 60]                 # list：有顺序，按位置取
print(f"list  = {scores}")
print(f"第一个 = {scores[0]}   最后一个 = {scores[-1]}   长度 = {len(scores)}")

student = {"name": "杨静茹", "score": 88}   # dict：按键取，不按位置
print(f"dict  = {student}")
print(f"student['score'] = {student['score']}")
student["grade"] = "大二"                   # 新增一个键
print(f"新增后 = {student}")

try:
    student["loss"]                         # 不存在的键
except KeyError as e:
    print(f"取不存在的键报 KeyError: {e}")

assert scores[0] == 88 and student["score"] == 88 and len(scores) == 4
print(f"{OK} 第 2 项通过")


# ══════════════════════════════════════════════════════════
# ③ for / if
# ══════════════════════════════════════════════════════════
hr("③ for / if")
print("for i in range(10): 执行 10 次（i 从 0 到 9）")
count = 0
for i in range(10):
    count = count + 1
print(f"实际循环 {count} 次")
assert count == 10, "range(10) 应为 10 次"

print("\n带 if 的循环：筛出不及格的分数")
failed = []
for s in scores:
    if s < 70:
        failed.append(s)
print(f"原分数 {scores} → 不及格 {failed}")
assert failed == [60]

print("\n缩进决定语句是否在循环体内：")
for i in range(2):
    print(f"  缩进内（执行 2 次）：i = {i}")
print("  缩进外（执行 1 次）")
print(f"{OK} 第 3 项通过")


# ══════════════════════════════════════════════════════════
# ④ 函数
# ══════════════════════════════════════════════════════════
hr("④ 函数")


def average(numbers):
    """求平均值。参数为 list，返回 float。"""
    if not numbers:            # 空列表直接返回，避免除零
        return 0
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)


def grade_of(score):
    """分数转等级。if / elif / else 自上而下判断，命中即返回。"""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "D"


avg = average(scores)
print(f"平均分 = {avg:.2f}   等级 = {grade_of(avg)}")
assert average([88, 92]) == 90.0
assert average([]) == 0
assert grade_of(95) == "A" and grade_of(65) == "C" and grade_of(10) == "D"
print(f"{OK} 第 4 项通过")


# ══════════════════════════════════════════════════════════
# ⑤ import
# ══════════════════════════════════════════════════════════
hr("⑤ import")
print(f"文件顶部已写 import csv / import os —— {OK}")
print(f"os.path.dirname(__file__) = {HERE}")
print("   ↑ 取当前文件所在目录的写法")
print("csv 模块负责读、写 CSV（下一项用到）")
print(f"{OK} 第 5 项通过")

# import 失败会抛 ModuleNotFoundError，常见原因是包装到了另一个 Python 环境，
# 与当前运行脚本的解释器不是同一个。


# ══════════════════════════════════════════════════════════
# ⑥ 读写文件：读 → 处理 → 写
# ══════════════════════════════════════════════════════════
hr("⑥ 读写文件：读 → 处理 → 写")

raw_path = os.path.join(HERE, "原始成绩.csv")
out_path = os.path.join(HERE, "scores.csv")

# ── 输入数据：不存在时生成一份，已存在则不覆盖 ──────────
if not os.path.exists(raw_path):
    with open(raw_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["姓名", "分数"])
        w.writerows([
            ["张三", 88], ["李四", 92], ["王五", 75],
            ["赵六", 60], ["钱七", 45], ["孙八", 96],
        ])
    print(f"（已生成示例输入：{os.path.basename(raw_path)}）")

# ── 读 ──────────────────────────────────────────────────
with open(raw_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))

header, data = rows[0], rows[1:]
print(f"\n读入 {len(data)} 行，表头为 {header}")

# ── 处理 ────────────────────────────────────────────────
result = []
for name, score_str in data:
    score = int(score_str)              # 文件读出的是字符串，需转数字
    result.append({
        "姓名": name,
        "分数": score,
        "等级": grade_of(score),
        "是否及格": "是" if score >= 60 else "否",
    })

# 按分数降序：key 指定比较字段
result.sort(key=lambda r: r["分数"], reverse=True)

# ── 写 ──────────────────────────────────────────────────
with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["姓名", "分数", "等级", "是否及格"])
    w.writeheader()
    w.writerows(result)

print(f"处理完成，写出 {len(result)} 行到 {os.path.basename(out_path)}")
print("\n结果预览：")
print(f"  {'姓名':<6}{'分数':>5}{'等级':>5}{'及格':>6}")
for r in result:
    print(f"  {r['姓名']:<6}{r['分数']:>5}{r['等级']:>5}{r['是否及格']:>6}")

assert os.path.exists(out_path), "输出文件未生成"
assert len(result) == 6
assert result[0]["分数"] == 96, "排序未生效"
print(f"\n{OK} 第 6 项通过：完成读文件 → 处理 → 写文件")


# ══════════════════════════════════════════════════════════
hr("全部通过")
print("""
产出文件（均在本目录）：
  原始成绩.csv    输入
  scores.csv      输出（含排序与等级列）
""")
