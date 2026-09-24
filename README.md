# 嵌入式大模型学习档案

端侧（嵌入式）大模型部署学习仓库，用于跨设备同步学习档案与代码。

## 主线

从 Ollama 量化对比实验起步 → 深入 llama.cpp 底层（GGUF 格式、Q4_K_M 量化原理）→ 从"使用量化"进入"理解量化" → 为部署到 Jetson / RK3588 等端侧设备做铺垫。

## 目录结构

```
.
├── 学习内容/    # 学习讲义、学习记录、对话记录（活文档）
│   ├── 学习记录.md                          # 学习主记录（由 Kimi Work 维护）
│   ├── GGUF与Q4_K_M量化原理_讲义_20260920.md
│   ├── 网络结构_从Linear到Transformer_讲义_20260920.md
│   ├── 注意力机制_从点积到QKV_讲义_20260922.md
│   ├── llama_cpp原生实操_讲义_20260924.md
│   ├── 对话记录_端侧大模型部署探索_20260920.md
│   └── note.md
└── scripts/     # 实验脚本与输出存档
    ├── parse-gguf.py            # 解析 Ollama blob 中真实 GGUF 文件头部结构
    ├── gguf-dump.txt            # parse-gguf.py 的输出
    ├── read-cmd-screen.ps1      # 读取终端窗口屏幕内容的实验脚本
    ├── read-ps-screen.ps1       # 读取 PowerShell 屏幕内容的实验脚本
    ├── inspect-cmd.ps1          # 命令行探测脚本
    ├── probe-cmd.ps1            # 命令行探针脚本
    └── *.txt                    # 各脚本对应的运行输出存档

根目录另有教学演示脚本：shape-conventions.py（形状约定）、attention-demo.py（注意力手算）、why-dot-product-works.py（梯度雕刻匹配的玩具实证）
```

## 硬件环境

| 机器 | 配置 | 角色 |
|---|---|---|
| A | AMD 9800X3D · RTX 4080 · 32GB DDR5-6000 | 主力实验机 |
| B | i3-13100 · Intel UHD 730 核显 · 16GB | 低端对照机（纯 CPU 推理） |

## 维护说明

- `学习内容/学习记录.md` 为活文档，由 Kimi Work 在学习节点处统一修订。
- 脚本运行输出（`scripts/*.txt`）与脚本一并存档，便于跨设备对照复现。
