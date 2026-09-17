# Transformer 的谱分析

> 把 Transformer 的注意力矩阵做**特征值分解**或 **SVD**，能看到它的"信息骨架"：哪些方向被放大、哪些被压缩。谱分析揭示了深 Transformer 难训练的根源（谱坍缩）、为什么需要残差连接（保持恒等谱）、以及部分模型中注意力的近似低秩现象。这些是诊断工具，不是所有 Transformer 的统一稳定性定理。

**难度**：[前沿]（需要 SVD + 特征值 + 范数 + 注意力机制）

## 一、纯数学：矩阵的谱

### 1.1 特征值谱

对方阵 $M$，特征值 $\{\lambda_i\}$ 的集合称为**谱**。关键量：
- **谱半径** $\rho(M) = \max |\lambda_i|$
- **谱范数** $\|M\|_2 = \sigma_{\max}(M)$（最大奇异值）
- **条件数** $\kappa(M) = \sigma_{\max}/\sigma_{\min}$

> 📖 特征值与 SVD：[特征值与特征向量详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md)

### 1.2 谱半径与稳定性

对矩阵幂 $M^L$（深网络 $L$ 层连乘）：

$$\rho(M) > 1 \Rightarrow \|M^L\| \to \infty; \quad \rho(M) < 1 \Rightarrow M^L \to 0$$

这个结论仅适用于重复使用同一矩阵。Transformer 各层雅可比不同，梯度由其**有序乘积的奇异值**控制。即使每层谱半径为 1，非正规矩阵也可产生增长。

> 📖 梯度病：[02-梯度消失爆炸的数学根源](02-梯度消失爆炸的数学根源.md)

---

## 二、注意力矩阵的谱结构

### 2.1 行随机矩阵的特征值

注意力矩阵 $A = \text{softmax}(QK^\top/\sqrt{d})$ 是**行随机矩阵**（每行和为 1）。这类矩阵有特殊谱性质：

- **最大特征值恒为 1**：$A\mathbf{1} = \mathbf{1}$（$\mathbf{1}$ 是全 1 向量，因为行和为 1）
- 其余特征值 $|\lambda_i| \leq 1$

**意义**：行随机矩阵可有多个模 1 特征值（如置换矩阵），特征模态与"频率"无一般对应；"低通"仅对局部平滑型注意力是经验事实，不能由行随机性推出。

### 2.2 注意力可能近似低秩

部分模型和数据上的注意力奇异值衰减较快，可做低秩近似。但 softmax 不保持秩，低秩的 $QK^\top$ 也可能得到满秩 $A$；接近单位矩阵的注意力更不适合低秩压缩。

```
奇异值 σ₁ ≥ σ₂ ≥ ... ≥ σ_N
   ████
   ██
   █
   ▏▏▏▏▏▏▏▏▏▏▏▏   ← 多数 σ 很小, 近似低秩
```

**应用**：Linformer 等基于（近似）低秩投影把 $O(N^2)$ 复杂度降到 $O(N)$；Linear Attention 基于核函数与结合律、Performer 基于随机特征，同样绕开显式 $N \times N$ 注意力矩阵（三者机理不同，不都是"低秩近似"）。

### 2.3 谱坍缩问题

深 Transformer 里，注意力矩阵的谱可能逐渐**坍缩**——少数方向垄断信息，其余被压扁。这导致：
- 梯度集中在少数参数
- 表示退化（所有 token 趋同）

**缓解**：残差连接（提供恒等传播项）、多头（提供多种混合方式），效果需结合模型验证。

---

## 三、残差与 LayerNorm 的谱作用

### 3.1 残差保持恒等谱

无残差的层 $\mathbf{h}_{l+1} = f_l(\mathbf{h}_l)$，雅可比 $J = \partial f/\partial \mathbf{h}$ 谱半径易偏离 1。

加残差 $\mathbf{h}_{l+1} = \mathbf{h}_l + f_l(\mathbf{h}_l)$，雅可比 $J = I + \partial f/\partial \mathbf{h}$。**恒等项 $I$ 把谱中心拉到 1 附近**：

$$\lambda_i(I + A) = 1 + \lambda_i(A)$$

这里 $A$ 表示残差分支的**雅可比**，不是注意力权重矩阵；后者还依赖输入，且有值投影。若 $\|A\|_2\leq\varepsilon$，单块奇异值位于 $[1-\varepsilon,1+\varepsilon]$，但深层连乘仍需控制累计误差，不能仅看特征值。

> 📖 残差的数学：[03-正交基→残差与变换](../PART-03-专题映射/轴线D-函数逼近/03-正交基→残差与变换.md)

### 3.2 LayerNorm 的谱影响

LayerNorm 的标准化部分控制特征尺度，再施加可学习仿射参数；它不保证雅可比谱范数小于 1。其导数尺度受 $\gamma/\sqrt{\operatorname{Var}(h)+\epsilon}$ 影响，低方差时可能放大梯度，去均值还会产生零导数方向。

---

## 四、工程诊断：监控谱

### 4.1 计算注意力矩阵的奇异值

```python
import torch

def attention_spectra(Q, K):
    A = torch.softmax(Q @ K.T / (K.size(-1) ** 0.5), dim=-1)
    S = torch.linalg.svdvals(A)   # 奇异值, 降序
    return S

Q, K = torch.randn(10, 8), torch.randn(10, 8)
S = attention_spectra(Q, K)
print("前 5 个奇异值:", S[:5])
print("有效秩:", (S > 0.01 * S[0]).sum().item())
# 有效秩小说明注意力高度集中(低秩)
```

### 4.2 监控雅可比谱范数（近似）

```python
import torch
from torch.func import jvp
from torch.autograd import grad

# 幂迭代估计 f 在 x 处雅可比 J 的谱范数 σ_max = ‖J‖₂
# u ← normalize(Jv)（前向模式 jvp）；v ← normalize(Jᵀu)（反向模式 VJP，autograd.grad）
def spectral_norm_estimate(f, x, iters=50):
    if iters < 1:
        raise ValueError("iters must be positive")
    x0 = x.detach()
    v = torch.randn_like(x0)
    v = v / v.norm()
    for _ in range(iters):
        u = jvp(f, (x0,), (v,))[1]                   # u = Jv
        sigma = u.norm()                             # ‖Jv‖ → σ_max
        if sigma == 0:
            return sigma
        u = u / sigma                                # u 归一化
        leaf = x0.clone().requires_grad_(True)
        (v,) = grad(f(leaf), leaf, grad_outputs=u)   # v = Jᵀu
        v = v / v.norm()                             # v 归一化
    return sigma
```

### 4.3 用谱归一化稳定训练

```python
from torch import nn
from torch.nn.utils import spectral_norm as sn
# 对注意力或 FFN 用谱归一化, 归一化到 σ_max=1(默认无 τ 参数; 需 τ 时可再乘 scale)
layer = sn(nn.Linear(512, 512))
```

---

## 五、谱视角能解释的现象

### 5.1 为什么深 Transformer 需要残差 + Norm

两者常能改善优化：残差提供恒等通路，LayerNorm 控制激活尺度。但它们不把完整雅可比谱锁定为 1，也不是所有架构的必要条件；具体稳定性还取决于初始化、残差缩放和学习率。

### 5.2 注意力的低秩 → 加速

既然 $A$ 近似低秩 $k$，可用 $A \approx U_k V_k^\top$ 把 $AV$ 的 $O(N^2 d)$ 降到 $O(N k d)$。这是低秩方法的一条加速路线；核化、稀疏注意力和精确的 FlashAttention 使用不同机制，且直接先求完整 SVD 并不能省去二次开销。

### 5.3 token 趋同（representation collapse）

深网络里若无残差，注意力反复作用会让 token 表示趋同（谱坍缩到少数方向）。残差 + 多头缓解此问题。

---

## 六、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  谱半径 ρ; 谱范数 σ_max; SVD; 条件数                       │
│  行随机矩阵: λ₁=1, 其余 |λ|≤1                              │
│  [[Mathematics-Universe/04-线性代数/05-特征值与特征向量/...]] │
└──────────────────────────┬───────────────────────────────┘
                           │ 谱决定连乘行为
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Transformer 的谱现象                                       │
│  注意力: 行随机；低秩和低通需要额外条件                             │
│  纯注意力可表示退化；残差+Norm 有助优化                     │
│  低秩、核化、稀疏是不同加速路线                            │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  svdvals 算谱; spectral_norm 稳定; 残差+Norm 是标配        │
└──────────────────────────────────────────────────────────┘
```

---

## 七、直觉总结

```
注意力 A 行随机 ⇒ A1=1、所有特征值模≤1；不保证保平均或低通
A 常低秩 ⇒ 少数方向垄断信息 ⇒ 可低秩近似加速
深 Transformer: 检查完整雅可比乘积，残差与 Norm 改善优化

谱半径判据适用于同一矩阵的幂；一般网络看乘积奇异值
```

**谱分析能诊断表示压缩与梯度传播。** 注意力矩阵本身的谱不能替代整个网络雅可比的分析。

---

## 八、延伸阅读

### 论文
- **Dong et al. (2021)** "Attention is not all you need: pure attention loses rank doubly exponentially"（纯注意力秩双指数坍缩——谱坍缩分析）
- **Choromanski et al. (2020)** "Performer"——随机特征注意力近似
- **Jing et al. (2022)** "Understanding Dimensional Collapse in Contrastive Self-supervised Learning"——维度坍缩分析

### 关联文章
- [04-注意力机制的线性代数本质](04-注意力机制的线性代数本质.md)（注意力的矩阵结构）
- [02-梯度消失爆炸的数学根源](02-梯度消失爆炸的数学根源.md)（谱半径与梯度）
- [01-特征值分解→PCA→自编码器](../PART-03-专题映射/轴线A-矩阵分解/01-特征值分解→PCA→自编码器.md)（SVD 应用）

---

## 联系网络

⬆ 上游：[特征值与特征向量详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md)（谱），[02-梯度消失爆炸的数学根源](02-梯度消失爆炸的数学根源.md)

⬇ 下游：高效 Transformer 设计（低秩近似、稀疏注意力）

↔ 横联：[04-注意力机制的线性代数本质](04-注意力机制的线性代数本质.md)，[01-特征值分解→PCA→自编码器](../PART-03-专题映射/轴线A-矩阵分解/01-特征值分解→PCA→自编码器.md)

🔗 跨域：随机过程（Markov 链的谱分析 = 收敛速率）、图论（图 Laplacian 谱）

━━━━━━━━━━━━━━
