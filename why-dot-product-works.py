# -*- coding: utf-8 -*-
# 验证: 训练如何用梯度把"匹配"雕刻进 q·k —— 玩具版单步注意力
# 设定: 2 个候选 token, k1=[1,0] k2=[0,1]; 当前 query 的 q 初始随机
# 目标: token2 的内容有用(相当于"预测下一个词需要 token2 的信息")
# loss = -ln(alpha2), 手工梯度下降, 观察 q 的方向变化
import numpy as np

np.set_printoptions(precision=4, suppress=True)
d = 2
k1 = np.array([1.0, 0.0])   # token1 的"标签向量"
k2 = np.array([0.0, 1.0])   # token2 的"标签向量"
K = np.stack([k1, k2])
y  = np.array([0.0, 1.0])   # 想要的结果: 全部注意力给 token2

q = np.array([0.1, 0.5])    # 初始 query: 方向接近 k2 但混杂
eta = 0.5

print("目标: softmax(q·k) 尽量 = [0, 1]  (即 q 应对准 k2)")
print(f"初始 q = {q}")
for step in range(6):
    s = K @ q / np.sqrt(d)              # scores = q·k / sqrt(d)
    e = np.exp(s - s.max())
    alpha = e / e.sum()                 # softmax -> 注意力权重
    loss = -np.log(alpha[1])
    grad_s = alpha - y                  # softmax+交叉熵的梯度, 漂亮的形式
    grad_q = K.T @ grad_s / np.sqrt(d)  # 链式法则: 传回 q
    cos = q @ k2 / (np.linalg.norm(q) * np.linalg.norm(k2))
    print(f"step{step}: scores={s}  alpha={alpha}  loss={loss:.4f}  q={q}  cos(q,k2)={cos:.4f}")
    q = q - eta * grad_q                # 梯度下降更新 q
print()
print("最终 q 的方向几乎与 k2 重合:", q)
