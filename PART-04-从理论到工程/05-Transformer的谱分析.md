# Transformer 的谱分析

> 把 Transformer 的注意力矩阵做**特征值分解**或 **SVD**，能看到它的"信息骨架"：哪些方向被放大、哪些被压缩。谱分析揭示了深 Transformer 难训练的根源（谱坍缩）、为什么需要残差连接（保持恒等谱）、以及注意力为何本质是低秩的。

**难度**：[前沿]（需要 SVD + 特征值 + 范数 + 注意力机制）

## 一、纯数学：矩阵的谱

### 1.1 特征值谱

对方阵 $M$，特征值 $\{\lambda_i\}$ 的集合称为**谱**。关键量：
- **谱半径** $\rho(M) = \max |\lambda_i|$
- **谱范数** $\|M\|_2 = \sigma_{\max}(M)$（最大奇异值）
- **条件数** $\kappa(M) = \sigma_{\max}/\sigma_{\min}$

> 📖 特征值与 SVD：[[Mathematics-Universe/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md]]

### 1.2 谱半径与稳定性

对矩阵幂 $M^L$（深网络 $L$ 层连乘）：

$$\rho(M) > 1 \Rightarrow \|M^L\| \to \infty; \quad \rho(M) < 1 \Rightarrow M^L \to 0$$

**Transformer 深度可达上百层，每层的雅可比谱半径必须接近 1**——否则梯度爆/消。

> 📖 梯度病：[[02-梯度消失爆炸的数学根源]]

---

## 二、注意力矩阵的谱结构

### 2.1 行随机矩阵的特征值

注意力矩阵 $A = \text{softmax}(QK^\top/\sqrt{d})$ 是**行随机矩阵**（每行和为 1）。这类矩阵有特殊谱性质：

- **最大特征值恒为 1**：$A\mathbf{1} = \mathbf{1}$（$\mathbf{1}$ 是全 1 向量，因为行和为 1）
- 其余特征值 $|\lambda_i| \leq 1$

**意义**：注意力作用在任何信号上，"直流分量"（平均）被保留，高频分量被衰减——**注意力有低通滤波特性**。

### 2.2 注意力是低秩的

实测发现，长序列的注意力矩阵 $A$ 的奇异值衰减很快——**少数大奇异值占据主要能量**。

```
奇异值 σ₁ ≥ σ₂ ≥ ... ≥ σ_N
   ████
   ██
   █
   ▏▏▏▏▏▏▏▏▏▏▏▏   ← 多数 σ 很小, 近似低秩
```

**应用**：Linear Attention、Performer 等用低秩近似把 $O(N^2)$ 复杂度降到 $O(N)$。

### 2.3 谱坍缩问题

深 Transformer 里，注意力矩阵的谱可能逐渐**坍缩**——少数方向垄断信息，其余被压扁。这导致：
- 梯度集中在少数参数
- 表示退化（所有 token 趋同）

**缓解**：残差连接（$I + A$ 保持恒等谱）、多头（多组谱分散信息）。

---

## 三、残差与 LayerNorm 的谱作用

### 3.1 残差保持恒等谱

无残差的层 $\mathbf{h}_{l+1} = f_l(\mathbf{h}_l)$，雅可比 $J = \partial f/\partial \mathbf{h}$ 谱半径易偏离 1。

加残差 $\mathbf{h}_{l+1} = \mathbf{h}_l + f_l(\mathbf{h}_l)$，雅可比 $J = I + \partial f/\partial \mathbf{h}$。**恒等项 $I$ 把谱中心拉到 1 附近**：

$$\lambda_i(I + A) = 1 + \lambda_i(A)$$

只要 $A$ 的谱小，$\lambda_i \approx 1$，连乘稳定。

> 📖 残差的数学：[[轴线D/03-正交基→残差与变换]]

### 3.2 LayerNorm 的谱影响

LayerNorm 把激活归一化到均值 0、方差 1——**控制了激活的尺度**，间接稳定了雅可比的谱范数。没有它，谱范数可能随训练漂移导致不稳定。

---

## 四、工程诊断：监控谱

### 4.1 计算注意力矩阵的奇异值

```python
import torch

def attention_spectra(Q, K):
    A = torch.softmax(Q @ K.T / (K.size(-1) ** 0.5), dim=-1)
    S = torch.linalg.svdvals(A)   # 奇异值, 降序
    return S

S = attention_spectra(Q, K)
print("前 5 个奇异值:", S[:5])
print("有效秩:", (S > 0.01 * S[0]).sum().item())
# 有效秩小说明注意力高度集中(低秩)
```

### 4.2 监控雅可比谱半径（近似）

```python
# 近似估计网络某层雅可比的谱范数(用 power iteration)
def spectral_norm_estimate(J_fn, x, iters=10):
    u = torch.randn(x.shape); u = u / u.norm()
    for _ in range(iters):
        v = J_fn(u); v = v / v.norm()
        u = J_fn(v, transpose=True)
    return v.norm()
```

### 4.3 用谱归一化稳定训练

```python
import torch.nn.utils.spectral_norm as sn
# 对注意力或 FFN 用谱归一化, 强制 ‖W‖ ≤ τ
layer = sn(nn.Linear(512, 512))
```

---

## 五、谱视角能解释的现象

### 5.1 为什么深 Transformer 需要残差 + Norm

两者从不同角度把谱半径锁定在 1 附近：
- 残差：$I$ 项让特征值中心移到 1
- LayerNorm：归一化控制谱范数上限

缺一不可——去掉任一，上百层 Transformer 训不动。

### 5.2 注意力的低秩 → 加速

既然 $A$ 近似低秩 $k$，可用 $A \approx U_k V_k^\top$ 把 $AV$ 的 $O(N^2 d)$ 降到 $O(N k d)$。这是高效 Transformer 的统一原理。

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
│  注意力: 行随机, 低秩, 低通滤波                             │
│  谱坍缩 ⇒ 表示退化; 残差+Norm 锁定 ρ≈1                     │
│  低秩性 ⇒ Linear Attention 加速                            │
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
注意力矩阵 A 是行随机 ⇒ 最大特征值=1(保平均), 其余<1(滤高频)
A 常低秩 ⇒ 少数方向垄断信息 ⇒ 可低秩近似加速
深 Transformer: 谱易坍缩 ⇒ 残差(I 项, λ→1) + Norm(控谱范数)

谱半径 ρ:
  ρ≈1 ⇒ 稳定; ρ>1 爆; ρ<1 消
  残差 + Norm = 把 ρ 锁在 1
```

**Transformer 的成败藏在注意力矩阵的谱里。** 谱半径接近 1 是深网络稳定的数学底线。

---

## 八、延伸阅读

### 论文
- **Dong et al. (2021)** "Attention is Not All You Need: Pure Attention Dies"——谱坍缩分析
- **Choromanski et al. (2020)** "Performer"——低秩注意力近似
- **Bjorck et al. (2021)** "Understanding Representation Collapse in Transformers"

### 关联文章
- [[04-注意力机制的线性代数本质]]（注意力的矩阵结构）
- [[02-梯度消失爆炸的数学根源]]（谱半径与梯度）
- [[轴线A/01-特征值分解→PCA→自编码器]]（SVD 应用）

---

## 联系网络

⬆ 上游：[[Mathematics-Universe/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md]]（谱），[[02-梯度消失爆炸的数学根源]]

⬇ 下游：高效 Transformer 设计（低秩近似、稀疏注意力）

↔ 横联：[[04-注意力机制的线性代数本质]]，[[轴线A/01-特征值分解→PCA→自编码器]]

🔗 跨域：随机过程（Markov 链的谱分析 = 收敛速率）、图论（图 Laplacian 谱）

━━━━━━━━━━━━━━
