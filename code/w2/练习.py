#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 补课收尾 · 六个语法点 + 读写文件小工具
================================================================
对应：你发给老师的信里「Python 补课正在进行，六项语法中两项已能独立使用」
目标：本周把剩下四项补完，并且做出那个「读文件 → 处理 → 写文件」的 20 行小脚本。

怎么用（**别只看，每关都自己改一处再跑**）：
    cd "D:/邵攀/遥感AI入门"
    ./.venv/Scripts/python.exe W3-20261110/05-python收尾/练习.py

跑完会看到六关全部 ✅，最后产出一个 scores.csv。
每一关末尾有「改一处试试」，照着改，跑一遍看结果变没变——这才是学会。

只学这六样，别的全跳过（类与继承、装饰器、async、生成器、多线程、类型注解）：
    ① 变量与类型   ② list 与 dict   ③ for / if
    ④ 函数         ⑤ import         ⑥ 读写文件
"""

import csv
import os
import sys

# Windows 控制台默认用 GBK，打印 ✅ 这类符号会直接抛 UnicodeEncodeError
# （2026-11-09 实测踩到）。这两行强制走 UTF-8，不然后面每一关都会崩。
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # 老版本 Python 没这个方法，退化成直接打印

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
print(f'a + a = {a + a}     ← 数字相加，真的算数')
print(f'b + b = {b + b!r}   ← 字符串相加，是"接在一起"')
try:
    a + b
except TypeError as e:
    print(f'a + b 会报错：{e}   ← 类型不对就不能加')

int_of_b = int(b)          # 文字转数字
print(f'int(b) = {int_of_b}  ← 转成数字后就能算了：{a + int_of_b}')
assert a + a == 10 and b + b == "55" and a + int_of_b == 10
print(f"{OK} 第一关过：知道 5 和 \"5\" 不是一回事")

# 改一处试试：把 b = "5" 改成 b = "7"，看 b + b 变成什么


# ══════════════════════════════════════════════════════════
# ② list 与 dict
# ══════════════════════════════════════════════════════════
hr("② list 与 dict")
scores = [88, 92, 75, 60]                 # list：有顺序，按位置取
print(f"list  = {scores}")
print(f"第一个 = {scores[0]}   最后一个 = {scores[-1]}   长度 = {len(scores)}")

student = {"name": "杨静茹", "score": 88}   # dict：按名字取，不按位置
print(f"dict  = {student}")
print(f"student['score'] = {student['score']}")
student["grade"] = "大二"                   # 加一项
print(f"加一项后 = {student}")

try:
    student["loss"]                         # 不存在的键
except KeyError as e:
    print(f"取不存在的键会报 KeyError: {e}   ← 周报验收标准说的就是这个")

assert scores[0] == 88 and student["score"] == 88 and len(scores) == 4
print(f"{OK} 第二关过：会取元素、会加键、认识 KeyError")

# 改一处试试：把 scores[1] 改成 100，看打印出来的 list


# ══════════════════════════════════════════════════════════
# ③ for / if
# ══════════════════════════════════════════════════════════
hr("③ for / if")
print("for i in range(10): → 跑 10 次（i 从 0 到 9）")
count = 0
for i in range(10):
    count = count + 1
print(f"实际循环了 {count} 次")
assert count == 10, "range(10) 应该跑 10 次"

print("\n带 if 的循环：把不及格的挑出来")
failed = []
for s in scores:
    if s < 70:
        failed.append(s)
print(f"原分数 {scores} → 不及格 {failed}")
assert failed == [60]

print("\n缩进决定谁在循环里：")
for i in range(2):
    print(f"  缩进内（跑 2 次）：i = {i}")
print("  缩进外（只跑 1 次）")
print(f"{OK} 第三关过：能看懂缩进块的执行顺序")

# 改一处试试：把 range(10) 改成 range(100)，看 count 变成多少


# ══════════════════════════════════════════════════════════
# ④ 函数
# ══════════════════════════════════════════════════════════
hr("④ 函数")


def average(numbers):
    """求平均分。参数是 list，返回一个数字。"""
    if not numbers:            # 空列表要挡一下，否则会除零
        return 0
    total = 0
    for n in numbers:
        total = total + n
    return total / len(numbers)


def grade_of(score):
    """把一个分数翻译成等级。if / elif / else 从上往下判断，命中就停。"""
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
print(f"{OK} 第四关过：知道 def / 参数 / return 各是什么")

# 改一处试试：把 grade_of 的 90 改成 85，看 88 分变成哪个等级


# ══════════════════════════════════════════════════════════
# ⑤ import
# ══════════════════════════════════════════════════════════
hr("⑤ import")
print(f"这个文件顶部写了 import csv / import os —— {OK} 已经用上了")
print(f"os.path.dirname(__file__) = {HERE}")
print("   ↑ 这就是'拿到当前文件所在目录'的标准写法")
print(f"csv 模块能：读 CSV、写 CSV（下面第六关就用它）")
print(f"{OK} 第五关过：知道 import 是「把别人的代码拿进来用」")

# 说明：import 失败会报 ModuleNotFoundError —— 你上周遇到的
#   ModuleNotFoundError: No module named 'torch'
# 就是这个。原因通常是「装到了另一个 Python 上」。


# ══════════════════════════════════════════════════════════
# ⑥ 读写文件 —— 这就是老师说的那个「20 行小工具」
# ══════════════════════════════════════════════════════════
hr("⑥ 读写文件：读 → 处理 → 写")

raw_path = os.path.join(HERE, "原始成绩.csv")
out_path = os.path.join(HERE, "scores.csv")

# ── 准备一份原始数据（第一次跑时创建；已存在就不覆盖）───────
if not os.path.exists(raw_path):
    with open(raw_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["姓名", "分数"])
        w.writerows([
            ["张三", 88], ["李四", 92], ["王五", 75],
            ["赵六", 60], ["钱七", 45], ["孙八", 96],
        ])
    print(f"（已创建示例输入：{os.path.basename(raw_path)}）")

# ── 读 ──────────────────────────────────────────────────
with open(raw_path, "r", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))

header, data = rows[0], rows[1:]
print(f"\n读进来 {len(data)} 行，表头是 {header}")

# ── 处理 ────────────────────────────────────────────────
result = []
for name, score_str in data:
    score = int(score_str)              # 文件里读出来是字符串，要转数字
    result.append({
        "姓名": name,
        "分数": score,
        "等级": grade_of(score),
        "是否及格": "是" if score >= 60 else "否",
    })

# 按分数从高到低排序：key 指定"按哪个字段比"
result.sort(key=lambda r: r["分数"], reverse=True)

# ── 写 ──────────────────────────────────────────────────
with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["姓名", "分数", "等级", "是否及格"])
    w.writeheader()
    w.writerows(result)

print(f"处理完，写出 {len(result)} 行到 {os.path.basename(out_path)}")
print("\n结果预览：")
print(f"  {'姓名':<6}{'分数':>5}{'等级':>5}{'及格':>6}")
for r in result:
    print(f"  {r['姓名']:<6}{r['分数']:>5}{r['等级']:>5}{r['是否及格']:>6}")

assert os.path.exists(out_path), "输出文件没写出来"
assert len(result) == 6
assert result[0]["分数"] == 96, "排序没生效"
print(f"\n{OK} 第六关过：完成了一个真正的「读文件 → 处理 → 写文件」工具")

# 改一处试试：把 range 之外的 if score >= 60 改成 70，看「是否及格」列变化


# ══════════════════════════════════════════════════════════
hr("全部通过")
print("""
六关都过了，Python 补课这一项就算收尾了。周报里可以这样写：

  「Python 补课六项已完成：能解释 5+5 / "5"+"5" / "5"+5 三种结果为何不同；
   会取 list 元素、会加 dict 键、认识 KeyError；
   看懂 for/if 的缩进块；会写带参数和返回值的函数；
   能读 CSV → 处理 → 写 CSV，产出 scores.csv（6 行，含排序与等级列）。」

产出文件（都在本目录）：
  原始成绩.csv     输入
  scores.csv       输出 ← 周报的"可核查结果"

最后自测（不看代码回答，答不上就回去再读一遍）：
  Q1 for i in range(10): 里面的代码跑几次？
  Q2 报 KeyError: 'loss' 是什么原因？
  Q3 dict 和 list 取数据的区别是什么？
  Q4 def 里面没有 return 会怎样？
""")
