# -*- coding: utf-8 -*-
# 验证: (1) 3 token x 2 维的注意力手算例子 (2) .T 只是视图, 不复制数据 (3) QK^T 的行都来自连续内存
import numpy as np

print("======== 手算注意力: 3 个 token, 头维 d=2 ========")
T, d = 3, 2
Q = np.array([[1., 0.], [0., 1.], [1., 1.]])   # 3 个 token 的 q 向量
K = np.array([[1., 0.], [1., 0.], [0., 1.]])   # 3 个 token 的 k 向量
V = np.array([[1., 2.], [3., 4.], [5., 6.]])   # 3 个 token 的 v 向量

scores = Q @ K.T / np.sqrt(d)
print("scores = QK^T / sqrt(2):\n", np.round(scores, 3))

# causal mask: 第 i 行只能看 j<=i (未来位置置 -inf)
mask = np.triu(np.ones((T, T), dtype=bool), k=1)
scores_masked = np.where(mask, -np.inf, scores)
e = np.exp(scores_masked - scores_masked.max(axis=1, keepdims=True))
attn = e / e.sum(axis=1, keepdims=True)
print("mask+softmax 后的注意力权重:\n", np.round(attn, 3))

out = attn @ V
print("输出 = 权重 @ V:\n", np.round(out, 3))
print("手算第3行: (1/3)*([1,2]+[3,4]+[5,6]) =", np.round((V[0]+V[1]+V[2])/3, 3))
print()

print("======== 转置是视图, 不是数据搬运 ========")
X = np.arange(12).reshape(3, 4).astype(np.float64)
Xt = X.T
print("X:\n", X.astype(int))
print("X.T:\n", Xt.astype(int))
print("X.T 是否拥有独立数据副本(OWNDATA):", Xt.flags['OWNDATA'], "  <- False = 没复制, 只是换了寻址步长")
print("X 的行步长:", X.strides, " X.T 的行步长:", Xt.strides, " <- 步长互换而已")
print()

print("======== 点积双方是否连续 ========")
row_x = X[1]            # x 的一行
row_w = np.arange(8).reshape(2, 4)[1]  # 按 [out,in] 存的 W 的一行
print("x 的行连续(相邻元素步长=8字节):", row_x.strides == (8,))
print("W([out,in]存法) 的行也连续:", row_w.strides == (8,))
