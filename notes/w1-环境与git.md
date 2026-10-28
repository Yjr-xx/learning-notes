# W1 笔记 · 环境与 git

日期：2026-10-28　状态：已完成

## 一、Python 环境（虚拟环境 = 独立小柜子）

**我的环境现状**

| 项 | 值 |
|---|---|
| Python | 3.12.10 |
| PyTorch | 2.14.1+cpu（只能用 CPU，没有显卡加速） |
| 位置 | D盘 |
| 包数量 | 82 个 |

**建一个环境的完整两步**

```bash
python -m venv .venv                                        # 第 1 步：建柜子
.venv/Scripts/python.exe -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple   # 第 2 步：装包
```

**关键理解**

- `.venv/` 就是个普通文件夹，里面有 `Scripts/python.exe`（独立的一份 Python）和 `Lib/site-packages/`（装的所有包）
- 电脑上有两个 Python：系统的（公共柜子，没 torch）和柜子里的（有 torch）
- 写 `.venv/Scripts/python.exe` 就是在说"用柜子里那个"
- 装包 = 下载文件放进了 `site-packages` 文件夹（亲眼验证过：装前 82 个，装后变多）
- `-i https://pypi.tuna.tsinghua.edu.cn/simple` 是清华镜像，因为官方源在国内太慢

**踩坑记录**

- 柜子 10-14 就建好了（做遥感AI入门项目时），我一开始以为要重装，其实早就有
- `torch.cuda.is_available()` 返回 `False` 不是报错，是"没有可用显卡"

## 二、tensor 是什么（PyTorch 的基本单位）

**一句话：tensor = 表格。** `tensor([[1., 2.], [3., 4.]])` 就是一张 2 行 2 列的表格。

| 属性 | 例子 | 意思 |
|---|---|---|
| `shape` | `torch.Size([2, 2])` | 几行几列 |
| `dtype` | `float32` / `int64` | 格子里装小数还是整数（写 `1.` 是小数，写 `1` 是整数） |
| `device` | `cpu` | 放在哪算 |

**记住的坑**：整数除法和小数除法不同（`10/3=3.333`，`10//3=3`）；算 loss、准确率必须用小数，否则精度被砍掉。

## 三、Python 类型（5 道题学到的）

```python
5 + 5       # 10   数字相加
"5" + "5"   # 55   文字拼接
"5" + 5     # TypeError: can only concatenate str (not "int") to str
int("7") + 1   # 8    int() 把文字转成数字
```

**结论**：数字和文字长得像，行为完全不同。`int()` / `str()` 是一对翻译官。

## 四、git 三件事（本周学会）

```bash
git init                   # 1. 建仓：文件夹开始记住改动
git add 文件名              # 2. 挑文件（不用 git add .）
git commit -m "说明"        # 3. 存档（只在本机）
git push                   # 4. 上传到 GitHub（别人才能看到）
```

**安全铁律**

- `.env`、`*.key`、Token 绝不进仓库（已写 .gitignore）
- 提交前必看 `git status` 和 `git diff --cached`
- 提交用的邮箱是 GitHub 的 noreply 地址，不是真实邮箱

## 下一篇

W2：PyTorch 训练循环