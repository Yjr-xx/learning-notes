# W1 练习：认识 tensor（张量）
# 运行方式：必须用项目 .venv 里那个 Python，不要用机器上其他 Python。
#   D:\邵攀\遥感AI入门\.venv\Scripts\python.exe w1_practice.py
# （VS Code 里：Ctrl+Shift+P → Python: Select Interpreter → 选上面这个路径）

import torch

print("=" * 50)
print("第 1 段：造一张 2 行 2 列的表格")
print("=" * 50)

# 注意：数字后面有小圆点（1. 而不是 1），表示"小数"
x = torch.tensor([[1., 2.],
                  [3., 4.]])

print("x 长这样：")
print(x)
print()
print("x.shape  =", x.shape, "  <- 几行几列")
print("x.dtype  =", x.dtype, "  <- 每格里装的是小数还是整数")
print("x.device =", x.device, "  <- 这张表放在 CPU 内存里")

print()
print("=" * 50)
print("第 2 段：同样是 2 行 2 列，但数字后面没有小圆点")
print("=" * 50)

y = torch.tensor([[1, 2],
                  [3, 4]])

print("y 长这样：")
print(y)
print()
print("y.dtype  =", y.dtype, "  <- 和上面不一样！这是整数")
print("y.shape  =", y.shape, "  <- 形状一样，都是 2 行 2 列")

print()
print("=" * 50)
print("第 3 段：给每个格子 +1")
print("=" * 50)

print("x + 1 的结果：")
print(x + 1)
print()
print("看：每个数都加了 1，表格形状不变。这就是 tensor 最基本的运算")

print()
print("=" * 50)
print("第 4 段：一张 2 行 3 列的表格")
print("=" * 50)

z = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])

print("z 长这样：")
print(z)
print()
print("z.shape =", z.shape, "  <- 2 行 3 列")
print()
print("所有段落跑完了。现在去文件里改数字，再跑一次，看输出怎么变。")
