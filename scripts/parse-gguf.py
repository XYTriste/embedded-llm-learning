# -*- coding: utf-8 -*-
# 解析 Ollama blob 中真实 GGUF 文件的头部结构
import struct, sys, io
import argparse

args = argparse.ArgumentParser()
args.add_argument("--path", help="Path to the GGUF file")
args = args.parse_args()

PATH = args.path if args.path else r"'D:\\llm_models\\blobs\\sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730'"
out = io.StringIO()
def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")

f = open(PATH, "rb")
def rd(fmt):
    sz = struct.calcsize(fmt)
    b = f.read(sz)
    return struct.unpack(fmt, b)

# ---- 1. 文件头：magic / version / counts ----
magic = f.read(4)
version, = rd("<I")
tensor_count, = rd("<Q")
kv_count, = rd("<Q")
p("magic bytes:", magic, "(ascii:", magic.decode("ascii", "replace") + ")")
p("version:", version)
p("tensor_count:", tensor_count)
p("metadata_kv_count:", kv_count)

# ---- 2. metadata 键值对 ----
VT = {0:"u8",1:"i8",2:"u16",3:"i16",4:"u32",5:"i32",6:"f32",7:"bool",8:"string",9:"array",10:"u64",11:"i64",12:"f64"}
def rd_str():
    n, = rd("<Q")
    return f.read(n).decode("utf-8", "replace")
def rd_val(t, depth=0):
    if t == 0: return rd("<B")[0]
    if t == 1: return rd("<b")[0]
    if t == 2: return rd("<H")[0]
    if t == 3: return rd("<h")[0]
    if t == 4: return rd("<I")[0]
    if t == 5: return rd("<i")[0]
    if t == 6: return round(rd("<f")[0], 6)
    if t == 7: return bool(rd("<B")[0])
    if t == 8: return rd_str()
    if t == 10: return rd("<Q")[0]
    if t == 11: return rd("<q")[0]
    if t == 12: return rd("<d")[0]
    if t == 9:
        et, = rd("<I"); n, = rd("<Q")
        vals = [rd_val(et, depth+1) for _ in range(n)]
        return f"array<{VT.get(et,et)}>[{n}] " + (str(vals[:8]) + (" ..." if n > 8 else "") if et != 8 else str(vals[:4]) + (" ..." if n > 4 else ""))
    raise ValueError("bad type " + str(t))

p("\n---- metadata (general.* 全部展示，其余只列键名) ----")
meta = {}
for _ in range(kv_count):
    k = rd_str()
    t, = rd("<I")
    v = rd_val(t)
    meta[k] = v
for k, v in meta.items():
    if k.startswith("general.") or k in ("tokenizer.chat_template",):
        p(" ", k, "=", v if not isinstance(v, str) or len(v) < 200 else v[:200] + "...<truncated>")
p("  (其余键名):", ", ".join(k for k in meta if not k.startswith("general.") and k != "tokenizer.chat_template"))

align = meta.get("general.alignment", 32)

# ---- 3. tensor infos ----
GT = {0:("F32",1,4),1:("F16",1,2),2:("Q4_0",32,18),3:("Q4_1",32,20),6:("Q5_0",32,22),7:("Q5_1",32,24),
      8:("Q8_0",32,34),9:("Q8_1",32,40),10:("Q2_K",256,84),11:("Q3_K",256,110),12:("Q4_K",256,144),
      13:("Q5_K",256,176),14:("Q6_K",256,210),15:("Q8_K",256,292),16:("IQ2_XXS",256,66),17:("IQ2_XS",256,74),
      18:("IQ3_XXS",256,98),19:("IQ1_S",256,50),20:("IQ4_NL",32,18),21:("IQ3_S",256,110),22:("IQ2_S",256,82),
      23:("IQ4_XS",256,136),24:("I8",1,1),25:("I16",1,2),26:("I32",1,4),27:("I64",1,8),28:("F64",1,8),
      29:("IQ1_M",256,56),30:("BF16",1,2)}

p("\n---- 前 8 个 tensor info ----")
infos = []
hist = {}
total_elems = 0
for i in range(tensor_count):
    name = rd_str()
    nd, = rd("<I")
    dims = [rd("<Q")[0] for _ in range(nd)]
    t, = rd("<I")
    off, = rd("<Q")
    ne = 1
    for d in dims: ne *= d
    infos.append((name, dims, t, off, ne))
    total_elems += ne
    tn = GT.get(t, (f"TYPE{t}", 1, 4))
    hist[t] = hist.get(t, [0, 0, 0])
    hist[t][0] += 1
    hist[t][1] += ne
    nb = (ne + tn[1] - 1) // tn[1] * tn[2]
    hist[t][2] += nb
    if i < 8:
        p(f"  {name}  dims={dims}  type={tn[0]}  nelem={ne}")

p("\n---- tensor 类型直方图（全部 %d 个 tensor）----" % tensor_count)
size_sum = 0
for t in sorted(hist):
    tn = GT.get(t, (f"TYPE{t}", 1, 4))
    c, ne, nb = hist[t]
    size_sum += nb
    p(f"  {tn[0]:8s}  tensors={c:4d}  elements={ne/1e6:10.1f}M  bytes={nb/1e9:6.3f} GB")
p("元素总数: %.4f B" % (total_elems / 1e9))
p("张量数据合计: %.3f GB (对齐前)" % (size_sum / 1e9))

# ---- 4. 验证数据区偏移与文件总大小 ----
pos = f.tell()
data_off = (pos + align - 1) // align * align
p("\ntensor info 结束位置: %d, 对齐(%d)后数据区起点: %d" % (pos, align, data_off))
import os
fsize = os.path.getsize(PATH)
p("实际文件大小: %.3f GB (%d bytes)" % (fsize / 1e9, fsize))
p("头部+元数据占比: %.2f%%" % (data_off / fsize * 100))

# ---- 5. 现场解码一个 Q4_K 块的真实权重 ----
p("\n---- 现场解码: blk.0.ffn_gate 的 Q4_K 第一个 super-block ----")
target = None
for name, dims, t, off, ne in infos:
    if name == "blk.0.ffn_gate.weight" and t == 12:  # 12 = Q4_K
        target = (name, dims, t, off, ne)
        break
name, dims, t, off, ne = target
f.seek(data_off + off)
# Q4_K 块 = 144 字节: d(f16) + dmin(f16) + scales/mins(12B, 各8个6bit) + qs(128B, 256个4bit)
d, = struct.unpack("<e", f.read(2))
dmin, = struct.unpack("<e", f.read(2))
sb = f.read(12)
qs = f.read(128)
# 解包 8 个 6bit scale 与 8 个 6bit min（前4个低6位分别在 sb[0..3]/sb[4..7]，高2位压缩在字节高位）
scales = [sb[j] & 0x3F for j in range(4)] + [((sb[j+4] & 0x0F) | ((sb[j-4] >> 6) << 4)) for j in range(4, 8)]
mins   = [sb[j+4] & 0x3F for j in range(4)] + [((sb[j+4] >> 4) | ((sb[j] >> 6) << 4)) for j in range(4, 8)]
p(f"tensor: {name} dims={dims} type=Q4_K")
p(f"super-block: d={d:.6f}  dmin={dmin:.6f}")
p(f"8 个子块 scale(6bit)={scales}")
p(f"8 个子块 min (6bit)={mins}")
p("原始 q 值(4bit, 低半字节, 前16个):", [qs[j] & 0xF for j in range(16)])
w = []
for i in range(16):
    q = qs[i] & 0xF
    w.append(round(d * scales[0] * q - dmin * mins[0], 5))
p("解码出的前 16 个权重:", w)

f.close()
with open("gguf-dump.txt", "w", encoding="utf-8") as g:
    g.write(out.getvalue())
print("saved -> gguf-dump.txt")
