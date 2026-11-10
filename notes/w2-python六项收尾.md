# W2 · Python 六项补课收尾

> 对应周报「计划 1」。目标不是背语法，是**能对着自己的项目代码讲清一个函数在做什么**。
> 代码：`code/w2/练习.py`　产出：`code/w2/scores.csv`

---

## 一、只学这六样，别的全跳过

| # | 语法点 | 学到什么程度算够 | 我的自测 |
|---|---|---|---|
| 1 | 变量与类型 | 知道 `x = 5` 和 `x = "5"` 不是一回事 | 能解释 `5+5` / `"5"+"5"` / `"5"+5` 三种结果为何不同 |
| 2 | list 与 dict | 会取元素、会加键、认识 `KeyError` | 能说清 list 按位置取、dict 按名字取 |
| 3 | for / if | 能看懂缩进块的执行顺序 | 看到 `for i in range(10):` 知道跑 10 次 |
| 4 | 函数 | 知道 `def` / 参数 / `return` 各是什么 | 会写带参数和返回值的函数 |
| 5 | import | 知道 `import torch` 是「把别人的代码拿进来用」 | 能解释 `ModuleNotFoundError` 在说什么 |
| 6 | 读写文件 | `open()` / `with` 能照抄改路径 | 能做出「读文件→处理→写文件」的小工具 |

**明确不学**（现在学也用不上）：类与继承、装饰器、异步 `async/await`、生成器、多线程、类型注解。

---

## 二、三个必须记住的点

### 1. `5 + 5` 和 `"5" + "5"` 是两件不同的事

```python
5 + 5        # → 10        数字相加，真的算数
"5" + "5"    # → "55"      字符串相加，是"接在一起"
"5" + 5      # → TypeError 类型不对，不能加
int("5") + 5 # → 10        先转成数字就能算了
```

**为什么重要**：从文件里读出来的数**默认是字符串**。`csv.reader` 给你的 `"88"` 不是 `88`，不转换就会出各种怪问题。

### 2. `KeyError` 是最常见的错

```python
student = {"name": "杨静茹", "score": 88}
student["score"]   # → 88
student["loss"]    # → KeyError: 'loss'
```

**含义**：你以为字典里有这个键，其实没有。

**为什么重要**：配置文件、模型状态、实验结果，全是键值对。报 `KeyError` 时，第一反应应该是「我拼错键名了吗？这个键真的被写进去了吗？」

### 3. 缩进决定谁在循环里

```python
for i in range(2):
    print("缩进内：跑 2 次")
print("缩进外：只跑 1 次")
```

Python 用**缩进**划分代码块，不像 C/Java 用花括号。缩进错了，逻辑就错了，而且不一定报错——**这才是最坑的地方**。

---

## 三、实际做的工具：读 → 处理 → 写

```python
# 读
with open("原始成绩.csv", "r", encoding="utf-8-sig") as f:
    rows = list(csv.reader(f))

# 处理
for name, score_str in rows[1:]:
    score = int(score_str)          # 关键：字符串转数字
    result.append({"姓名": name, "分数": score, "等级": grade_of(score)})
result.sort(key=lambda r: r["分数"], reverse=True)

# 写
with open("scores.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[...])
    w.writeheader()
    w.writerows(result)
```

**为什么这三步重要**：所有工程都是这个骨架换内容。

| 场景 | 同样是三步 |
|---|---|
| 训练模型 | 读图片数据集 → 前向反向更新 → 写权重文件 |
| 跑评测 | 读测试集 → 模型推理 → 写指标 JSON |
| 以后配 STAR | 读标注文件 → 转成三元组 → 写新标注 |

---

## 四、踩到的坑（值得记住）

### 坑 1：Windows 控制台默认 GBK，打印符号直接崩

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'
```

**原因**：Windows 终端默认用 GBK 编码，脚本里打印 ✅ 这类符号超出 GBK 范围。

**修法**（在脚本开头加）：
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
```

### 坑 2：CSV 用 Excel 打开中文乱码

**原因**：缺 BOM 头，Excel 不知道这是 UTF-8。

**修法**：读写都用 `encoding="utf-8-sig"`（注意多出来的 `-sig`）。

---

## 五、自测四问（答不上就回去再看）

1. `for i in range(10):` 里面的代码跑几次？→ **10 次**
2. 报 `KeyError: 'loss'` 是什么原因？→ **字典里没有 `loss` 这个键**
3. dict 和 list 取数据的区别？→ **list 按位置（0,1,2），dict 按名字**
4. `def` 里面没有 `return` 会返回什么？→ **`None`**
