# Fourier 分析 → 注意力机制

> Fourier 分析说：任何信号都能拆成**正弦波的叠加**——一组正交基上的展开。注意力机制说：每个 token 都对其他 token 算一组**加权叠加**，权重由相似度决定。两者都是"**用一组基表示信号**"的思想，只是基从"固定的三角函数"变成了"数据自适应的键值对"。

**难度**：[进阶]（需要 Fourier 级数 + 内积空间 + 注意力机制基础）

## 一、纯数学版：Fourier 分解

### 1.1 Fourier 级数

对周期 $2\pi$ 的函数 $f$，若满足 Dirichlet 条件，Fourier 级数在连续点收敛到 $f(x)$，跳跃点收敛到左右极限的平均。下面等式按连续点书写：

$$f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left(a_n \cos nx + b_n \sin nx\right)$$

复数形式更紧凑：

$$f(x) = \sum_{n=-\infty}^{\infty} c_n e^{inx}, \quad c_n = \frac{1}{2\pi}\int_{-\pi}^{\pi} f(x) e^{-inx}\,dx$$

**核心**：把函数表示成一组**正交基** $\{e^{inx}\}$ 的线性组合。

> 📖 无穷级数与 Fourier：[无穷级数详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/06-无穷级数/无穷级数详解.md)

### 1.2 正交性：内积空间的视角

基函数 $e_n(x) = e^{inx}$ 在内积 $\langle f, g\rangle = \frac{1}{2\pi}\int f\bar g$ 下正交：

$$\langle e_m, e_n\rangle = \delta_{mn}$$

**展开系数**就是投影：$c_n = \langle f, e_n\rangle$。

**这是 Hilbert 空间的标准操作**：找正交基 → 投影 → 得系数 → 重建。

> 📖 内积空间：[向量与向量空间详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/04-线性代数/03-向量与向量空间/向量与向量空间详解.md)

### 1.3 连续 Fourier 变换

非周期函数 $f \in L^1(\mathbb{R})$ 的 Fourier 变换：

$$\hat f(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i \xi x}\,dx$$

**物理解读**：$\hat f(\xi)$ 是 $f$ 在频率 $\xi$ 处的"含量"。若还满足 $f\in L^2$，$|\hat f|^2$ 是能量谱密度；平稳随机过程的功率谱密度需从自相关等另行定义。

---

## 二、注意力机制：数据驱动的"基展开"

### 2.1 标准 Self-Attention

对序列 $X \in \mathbb{R}^{N \times d}$，定义查询/键/值：

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

注意力输出：

$$\text{Attn}(X) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

### 2.2 逐行拆解：加权叠加的类比

对第 $i$ 个 token 的输出：

$$\text{out}_i = \sum_{j=1}^N \underbrace{\text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)_j}_{\text{权重 } \alpha_{ij}} \cdot v_j$$

**与 Fourier 的对应**：
| Fourier | Attention |
|---------|-----------|
| 基函数 $e^{inx}$（固定） | 值向量 $v_j$（数据自适应） |
| 系数 $c_n = \langle f, e_n\rangle$（内积） | 权重 $\alpha_{ij} \propto e^{q_i \cdot k_j/\sqrt{d_k}}$（点积） |
| 重建 $f = \sum c_n e_n$ | 输出 $\text{out}_i = \sum \alpha_{ij} v_j$ |

**类比的边界**：Fourier 使用固定正交基；注意力值向量可以线性相关、不正交，通常不构成基。注意力权重非负且和为 1，Fourier 系数可为负或复数。因此注意力是数据相关的凸组合，不是 Fourier 分解或正交投影。

### 2.3 softmax：连续的"频率选择"

Fourier 里 $|c_n|$ 大表示该频率含量高。Attention 里 $\alpha_{ij}$ 大表示 token $i$ 与 token $j$ 相关性强。

**softmax 的作用**：把点积分数归一化成概率分布，相当于"软选择"——哪些 token 该被聚合到当前位置。

```
token i 对所有 token 的注意力:
  α_i = [0.05, 0.7, 0.2, 0.05]   ← softmax 归一化
                          ↑ 最相关的 token 贡献最大
out_i = 0.05·v₁ + 0.7·v₂ + 0.2·v₃ + 0.05·v₄
```

---

## 三、为什么除以 $\sqrt{d_k}$：方差与梯度

### 3.1 点积的方差

设 $q, k \in \mathbb{R}^{d_k}$ 各分量独立、均值 0、方差 1。则：

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \quad \mathbb{E}[q\cdot k] = 0, \quad \text{Var}(q\cdot k) = d_k$$

**点积的标准差随 $d_k$ 增大**。$d_k$ 大时，$q\cdot k$ 数值很大，softmax 进入饱和区（梯度近乎 0）。

### 3.2 缩放的数学根据

除以 $\sqrt{d_k}$ 把方差拉回 1：

$$\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = 1$$

**意义**：保持 softmax 在梯度敏感区，避免训练停滞。这是个**方差归一化**操作。

---

## 四、Multi-Head Attention：多组基并行

### 4.1 多头 = 多组可学习投影（不要求正交）

单头注意力相当于用一组基展开。多头把 $d$ 维拆成 $h$ 个 $d/h$ 维子空间，**每组独立做加权聚合**：

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

$$\text{head}_i = \text{Attn}(X W_Q^i, X W_K^i, X W_V^i)$$

### 4.2 与多分辨率 Fourier 的类比

- 短时 Fourier 变换（STFT）：用通常固定大小的滑动窗口分析局部频率；多种窗口大小是进一步的多分辨率扩展
- 多头注意力：在不同子空间里捕获不同类型的关系

**意义**：不同头能学到不同的"模式"——语法头、语义头、位置头等。

---

## 五、工程实现层：PyTorch Attention

### 5.1 手写 Scaled Dot-Product Attention

```python
import torch
import torch.nn.functional as F
import math

def attention(Q, K, V, mask=None):
    # Q, K, V: (batch, heads, seq, d_k)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., seq, seq)
    if mask is not None:
        allowed = torch.broadcast_to(mask != 0, scores.shape)
        if not allowed.any(dim=-1).all():
            raise ValueError("each query must have at least one unmasked key")
        scores = scores.masked_fill(~allowed, float('-inf'))
    alpha = F.softmax(scores, dim=-1)                     # 注意力权重
    return alpha @ V                                      # (..., seq, d_k)

Q = torch.randn(2, 8, 10, 64)   # batch=2, heads=8, seq=10, d_k=64
K = torch.randn(2, 8, 10, 64)
V = torch.randn(2, 8, 10, 64)
out = attention(Q, K, V)        # (2, 8, 10, 64)
```

### 5.2 Multi-Head 封装

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        if n_heads < 1 or d_model % n_heads:
            raise ValueError("d_model must be divisible by positive n_heads")
        self.h = n_heads
        self.dk = d_model // n_heads
        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
        self.W_o = torch.nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.h, self.dk).transpose(1, 2)
        out = attention(q, k, v)              # (B, h, N, dk)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.W_o(out)
```

### 5.3 验证缩放的作用

```python
# softmax 具有平移不变性：起作用的是分数之间的"差"，不是绝对值。
# 因此以 0 为基准分数，观察分数差大小对 softmax 分布与梯度的影响
d_k = 512
q = torch.randn(d_k); k = torch.randn(d_k)
raw = q @ k                              # 未缩放: Var(q·k)=d_k, 分数差量级 ~√d_k≈23
for s in [raw, raw / math.sqrt(d_k)]:    # 未缩放 vs 除以 √d_k（分数差量级 ~1）
    p = F.softmax(torch.tensor([0.0, s]), dim=0)
    grad_scale = (p[0] * p[1]).item()    # softmax 输出对 logit 的梯度尺度 ~ p(1-p)
    print(f"分数差={s.item():9.3f}  softmax={p.tolist()}  梯度尺度≈{grad_scale:.4f}")
# 未缩放: |分数差|≈23 → 分布接近 one-hot（如 [1.0, 0.0]）→ 梯度尺度→0（饱和）
# 缩放后: |分数差|≈1   → 分布平缓（如 [0.68, 0.32]）      → 梯度健康
```

---

## 六、Fourier 视角能解释什么

### 6.1 注意力 = 低通滤波

某些局部平滑注意力可以近似低通滤波，但局部关注或行随机性本身不推出低通。一般注意力依赖输入、未必平移不变，也未必能由固定 Fourier 频响描述。

### 6.2 傅里叶特征解决位置外推

正弦位置特征可在任意位置计算，但这不等于模型能可靠外推到训练长度之外。周期性还可能带来混叠；外推效果依赖频率、训练长度和注意力结构，必须实验验证。

### 6.3 谱视角的 Transformer 分析

SVD 给出奇异值谱，特征值分解给出特征值谱，两者不能混称。它们可诊断注意力的低秩现象；训练稳定性还需分析完整层的雅可比与残差。

> 📖 谱分析视角：[05-Transformer的谱分析](../../PART-04-从理论到工程/05-Transformer的谱分析.md)

---

## 七、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  Fourier: f = Σcₙeⁱⁿˣ, cₙ = ⟨f, eₙ⟩                     │
│  正交基 + 投影 + 叠加                                      │
│  Hilbert 空间的标准操作                                     │
│  [[Mathematics-Universe/03-高等数学/06-无穷级数/...]]       │
└──────────────────────────┬───────────────────────────────┘
                           │ 基从"固定三角函数"→"数据自适应 V"
                           ▼
┌──────────────────────────────────────────────────────────┐
│  注意力机制                                                 │
│  out_i = Σⱼ softmax(qᵢ·kⱼ/√d)·vⱼ                          │
│  数据驱动的基展开; softmax=软选择; √d=方差归一化            │
│  多头 = 多组并行基(多分辨率 Fourier 类比)                  │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  scores = QKᵀ/√d → softmax → @V                           │
│  MultiHead: 拆维度并行 → concat → 线性投影                 │
└──────────────────────────────────────────────────────────┘
```

---

## 八、直觉总结

```
Fourier: 信号 = 正弦波(固定基)的加权叠加, 权重 = 内积
Attention: token = 其他 token 的值(自适应基)的加权叠加, 权重 = 点积

创新: 基不再是数学家预设的, 而是从数据学出来的
多头: 同时用多组基, 捕获不同模式(类比多分辨率 Fourier)
√d:  把点积方差拉回 1, 防 softmax 饱和
```

**Fourier 与注意力都使用加权叠加，但数学结构不同。** 前者是正交基展开，后者是数据相关的加权聚合。

---

## 九、延伸阅读

### 论文
- **Vaswani et al. (2017)** "Attention Is All You Need"——Transformer 原文
- **Lee et al. (2021)** "FNet: Mixing Tokens with Fourier Transforms"——用 Fourier 变换替换自注意力子层，是"注意力 vs Fourier"的直接对照实验

### 关联文章
- [01-Weierstrass逼近→通用近似定理](01-Weierstrass逼近→通用近似定理.md)（函数逼近的另一组基）
- [03-正交基→残差与变换](03-正交基→残差与变换.md)（正交基与残差网络）
- [04-注意力机制的线性代数本质](../../PART-04-从理论到工程/04-注意力机制的线性代数本质.md)（线性代数视角）

---

## 联系网络

⬆ 上游：[无穷级数详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/06-无穷级数/无穷级数详解.md)（Fourier 级数），[向量与向量空间详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/04-线性代数/03-向量与向量空间/向量与向量空间详解.md)（内积与正交）

⬇ 下游：[04-注意力机制的线性代数本质](../../PART-04-从理论到工程/04-注意力机制的线性代数本质.md)，[05-Transformer的谱分析](../../PART-04-从理论到工程/05-Transformer的谱分析.md)

↔ 横联：[03-正交基→残差与变换](03-正交基→残差与变换.md)，[01-Weierstrass逼近→通用近似定理](01-Weierstrass逼近→通用近似定理.md)

🔗 跨域：信号处理（滤波器、谱分析）、图像压缩（DCT）、量子力学（态叠加）

━━━━━━━━━━━━━━
