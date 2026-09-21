# -*- coding: utf-8 -*-
# 演示 GGUF dims 约定 vs 数学矩阵约定，以及行/列两种视角的等价性
import numpy as np

print("======== 约定 1：GGUF 的 dims 书写法 ========")
# GGUF 把"内存中连续的一维"写在最前面（C 数组声明顺序）
# token_embd 的 dims=[3584, 152064] 意味着等价于 C 语言: float E[152064][3584]
E_math = np.zeros((152064, 3584), dtype=np.float16)
print("GGUF 写法 dims = [3584, 152064]")
print("数学记法      = [152064, 3584]  （152064 行 × 3584 列）")
print("C 声明        = float E[152064][3584]  内层 3584 是连续存储的")
print("token id=t 的向量在文件中的字节偏移 = t * 3584 * 2")
print()

print("======== 约定 2：Linear 权重的 dims 读法（验证三条）========")
for name, dims, in_dim, out_dim in [
    ("attn_q.weight     ", [3584, 3584], 3584, 3584),
    ("attn_k.weight     ", [3584, 512], 3584, 512),
    ("ffn_gate.weight   ", [3584, 18944], 3584, 18944),
    ("ffn_down.weight   ", [18944, 3584], 18944, 3584),
]:
    print(f"{name} GGUF dims={dims}  ->  输入维={in_dim}  输出维={out_dim}")
print("规律：dims 第一个数=输入维，第二个数=输出维")
print()

print("======== 行视角 vs 列视角：同一件事的两副眼镜 ========")
E = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6],
              [0.7, 0.8, 0.9],
              [1.0, 1.1, 1.2]])          # 4 个 token × 3 维的小词表
ids = [3, 0]                                # 输入两个 token
X = E[ids]                                  # 查表 -> [2, 3]
print("输入 token ids:", ids)
print("查表后 X（行=token，每行是一个 token 的向量）:\n", X)

W = np.array([[1.0, 0.0, -1.0],
              [0.5, 0.5, 0.5]])           # nn.Linear(3, 2) 的权重，形状 [out, in] = [2, 3]
b = np.array([0.01, -0.02])

y_row = X @ W.T + b                       # 行视角：y = x W^T + b
y_col = (W @ X.T + b[:, None]).T          # 列视角：y^T = W x^T + b
print("\n行视角  X @ W.T + b =\n", y_row)
print("列视角  (W @ X.T + b).T =\n", y_col)
print("两者完全相等:", np.allclose(y_row, y_col))
print("输出形状 [T, out] =", y_row.shape)
print()

print("======== Qwen2.5-7B 完整形状漂流（T=557）========")
T, H, V, F = 557, 3584, 152064, 18944
chain = [
    ("token ids", "[T]", f"[{T}]"),
    ("查 embedding 表", "[T, 3584]", f"[{T}, {H}]"),
    ("RMSNorm（逐行）", "[T, 3584]", f"[{T}, {H}]"),
    ("attn_q: x W^T", "[T, 3584]", f"[{T}, {H}]"),
    ("attn_k / attn_v", "[T, 512]", f"[{T}, 512]"),
    ("scores = Q K^T（每头）", "[T, T]", f"[{T}, {T}]"),
    ("weighted V, 拼接", "[T, 3584]", f"[{T}, {H}]"),
    ("FFN gate/up", "[T, 18944]", f"[{T}, {F}]"),
    ("FFN down", "[T, 3584]", f"[{T}, {H}]"),
    ("x28 层后 output_norm", "[T, 3584]", f"[{T}, {H}]"),
    ("输出头 logits", "[T, 152064]", f"[{T}, {V}]"),
]
for step, generic, actual in chain:
    print(f"  {step:28s} {generic:14s} {actual}")
print("\n注意：T 始终挂在第一维，像行李牌一样全程不动")
