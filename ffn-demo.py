# -*- coding: utf-8 -*-
# 验证 SwiGLU 手算例子: 1 个 token, 2 维 -> 3 通道 -> 2 维
import numpy as np

def silu(x):
    return x / (1.0 + np.exp(-x))

x  = np.array([1., 2.])                                   # [1, 2]
Wg = np.array([[1., 0.], [0., 1.], [1., 1.]])             # gate: 2 -> 3
Wu = np.array([[0., 1.], [1., 0.], [1., 1.]])             # up:   2 -> 3
Wd = np.array([[1., 0., 0.], [0., 1., 1.]])               # down: 3 -> 2

g = Wg @ x        # gate 打分
u = Wu @ x        # up 内容
print("gate(x) =", g)
print("up(x)   =", u)
print("silu(gate) =", np.round(silu(g), 3))
h = silu(g) * u
print("silu(gate) * up =", np.round(h, 3), "  <- 逐通道相乘(门控)")
y = Wd @ h
print("down(h) =", np.round(y, 3))

print()
print("对照: 若去掉 gate, 标准 FFN(relu(up)) 会得到:")
print("relu(up) =", np.round(np.maximum(Wu @ x, 0), 3), " -> down:", np.round(Wd @ np.maximum(Wu @ x, 0), 3))
