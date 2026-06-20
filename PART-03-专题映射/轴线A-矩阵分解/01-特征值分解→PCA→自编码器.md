# 特征值分解 → PCA → 自编码器

> 同一个数学操作——对矩阵做谱分解——从纸笔推导（考研线代）到数据降维（PCA）再到神经网络训练（自编码器），本质完全不变，只是"谁在做"和"为什么做"在变。

**难度**：[标准]（需要线性代数基础 + 能写简单PyTorch）

## 一、纯数学版

### 1.1 定义

设 $A$ 是 $n \times n$ 实对称矩阵，则存在正交矩阵 $Q$ 和对角矩阵 $\Lambda$，使得：

$$A = Q\Lambda Q^T$$

其中：
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$，$\lambda_i$ 是 $A$ 的**特征值**
- $Q$ 的列 $\mathbf{q}_1, \mathbf{q}_2, \ldots, \mathbf{q}_n$ 是 $A$ 的**特征向量**，彼此正交：$\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$

**几何意义**：$A$ 是一个线性变换。特征向量是这个变换的"不动方向"——变换只改变长度（按 $\lambda_i$ 缩放），不改变方向。

> 📖 详细推导与性质：[[Mathematics-Universe/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md]]

### 1.2 关键性质

| 性质 | 内容 | 为什么重要 |
|------|------|-----------|
| 特征值非负 | 实对称矩阵的特征值全为实数 | PCA 需要排序，实数才有大小关系 |
| 特征向量正交 | $\mathbf{q}_i \perp \mathbf{q}_j$（$i \neq j$） | 正交基 = 无冗余的坐标系统 |
| 排序不变性 | 可以按 $|\lambda_1| \geq |\lambda_2| \geq \cdots$ 排序 | 最大的特征值 = 方差最大的方向 |
| Frobenius范数 | $\|A\|_F^2 = \sum_i \lambda_i^2$ | 矩阵的"能量"全部集中在特征值上 |

### 1.3 手动计算示例

设 $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$

**Step 1**：求特征值
$$|\lambda I - A| = \begin{vmatrix} \lambda-2 & -1 \\ -1 & \lambda-2 \end{vmatrix} = (\lambda-2)^2 - 1 = 0$$
$$\lambda^2 - 4\lambda + 3 = 0 \Rightarrow \lambda_1 = 3, \lambda_2 = 1$$

**Step 2**：求特征向量

$\lambda_1 = 3$：$(3I - A)\mathbf{x} = 0 \Rightarrow \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}\mathbf{x} = 0 \Rightarrow \mathbf{q}_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$

$\lambda_2 = 1$：$(I - A)\mathbf{x} = 0 \Rightarrow \begin{pmatrix} -1 & -1 \\ -1 & -1 \end{pmatrix}\mathbf{x} = 0 \Rightarrow \mathbf{q}_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$

**Step 3**：写出分解
$$A = \begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{pmatrix} \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{pmatrix}$$

验证：$Q$ 的列正交（内积为0），$\Lambda$ 是对角的，$A = Q\Lambda Q^T$ 成立。

---

## 二、第一次应用：PCA（主成分分析）

### 2.1 从特征值分解到数据降维

**PCA 的本质**：对**数据协方差矩阵**做特征值分解，找到方差最大的方向。

给定数据中心化后的数据矩阵 $X \in \mathbb{R}^{n \times d}$（$n$ 个样本，$d$ 维）：

$$\Sigma = \frac{1}{n-1}X^TX \quad (\text{协方差矩阵，} d \times d)$$

对 $\Sigma$ 做特征值分解：
$$\Sigma = Q\Lambda Q^T$$

**投影到前 $k$ 个主成分**：
$$X_{\text{reduced}} = XQ_k \quad (Q_k \text{ 是前 } k \text{ 个特征向量组成的矩阵，} d \times k)$$

**这就是 PCA**——没有神经网络，没有反向传播，只是矩阵乘法。

### 2.2 为什么是"主成分"

按特征值大小排序：$\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d$

- $\lambda_1$ 对应的方向 $\mathbf{q}_1$ 是数据**方差最大**的方向 → **第一主成分**
- $\lambda_2$ 对应的方向 $\mathbf{q}_2$ 是与 $\mathbf{q}_1$ 正交的方向中方差最大的 → **第二主成分**
- 以此类推

**保留前 $k$ 个主成分 = 保留最大的 $k$ 个特征值对应的方向**。

> 📖 深入理解：[[Mathematics-Universe/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md#52-重要性质]] 和 [[Mathematics-Universe/05-概率论与数理统计/04-数字特征/数字特征详解.md#33-协方差与相关系数]]

### 2.3 PCA 的数学目标

PCA 等价于求解以下优化问题：

$$\mathbf{q}_1 = \arg\max_{\|\mathbf{q}\|=1} \text{Var}(X\mathbf{q}) = \arg\max_{\|\mathbf{q}\|=1} \mathbf{q}^T\Sigma\mathbf{q}$$

用 Lagrange 乘数法：
$$\mathcal{L}(\mathbf{q}, \lambda) = \mathbf{q}^T\Sigma\mathbf{q} - \lambda(\mathbf{q}^T\mathbf{q} - 1)$$
$$\frac{\partial\mathcal{L}}{\partial\mathbf{q}} = 2\Sigma\mathbf{q} - 2\lambda\mathbf{q} = 0 \Rightarrow \Sigma\mathbf{q} = \lambda\mathbf{q}$$

**结论**：PCA 的第一主成分就是协方差矩阵的最大特征值对应的特征向量。

这直接来自 [[Mathematics-Universe/03-高等数学/04-多元微分学/多元微分学详解.md#45-条件极值与Lagrange乘数法]]。

### 2.4 手写 PCA

```python
import torch

def pca_manual(X, k=2):
    """
    纯矩阵运算实现 PCA
    X: (n, d) 数据矩阵
    k: 降到 k 维
    """
    # 1. 中心化
    X_centered = X - X.mean(dim=0)

    # 2. 协方差矩阵 (d, d)
    Sigma = X_centered.T @ X_centered / (X.shape[0] - 1)

    # 3. 特征值分解
    eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)

    # 4. 按特征值降序排列
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx]
    Q_k = eigenvectors[:, idx[:k]]  # (d, k)

    # 5. 投影
    X_reduced = X_centered @ Q_k  # (n, k)

    # 6. 重构（验证信息保留比例）
    X_recon = X_reduced @ Q_k.T + X.mean(dim=0)
    info_ratio = eigenvalues[:k].sum() / eigenvalues.sum()

    return X_reduced, Q_k, info_ratio

# 测试
torch.manual_seed(42)
X = torch.randn(500, 10)  # 500个样本，10维
X_2d, components, ratio = pca_manual(X, k=2)
print(f"保留信息比例: {ratio:.1%}")  # 通常 60-90%
```

**关键行解析**：

| 代码行 | 对应的数学操作 | 数学含义 |
|--------|---------------|---------|
| `X - X.mean(dim=0)` | 中心化 | 使协方差矩阵 = 相关矩阵 × 方差 |
| `X.T @ X / (n-1)` | $\Sigma = \frac{1}{n-1}X^TX$ | 样本协方差 |
| `torch.linalg.eigh(Sigma)` | 特征值分解 | $Q\Lambda Q^T$（`eigh` 专为对称矩阵优化） |
| `eigenvalues.sort(descending)` | $\lambda_1 \geq \lambda_2 \geq \cdots$ | 按方差贡献排序 |
| `X_centered @ Q_k` | $XQ_k$ | 投影到主成分空间 |

---

## 三、第二次应用：自编码器（Autoencoder）

### 3.1 PCA 的局限性

PCA 是**线性**降维：
$$X_{\text{reduced}} = XQ_k \quad (\text{只有矩阵乘法})$$

它假设数据在**线性子空间**中。但真实数据往往不是：

- 人脸图像：人脸位于一个**非线性流形**上（姿态、光照的变化是非线性的）
- 文本嵌入：语义空间有复杂的弯曲结构

### 3.2 用神经网络替代线性投影

**核心思想**：把 $XQ_k$ 中的矩阵 $Q_k$ 换成一个神经网络 $f_\theta$。

```
PCA:                          Autoencoder:
X ──[Q_k]──→ 低维表示          X ──[Encoder f_θ]──→ Z ──[Decoder g_φ]──→ X̂
              ↓                                       ↓
            XQ_k (线性)                            MSE(X, X̂) (训练目标)
```

**编码器** $f_\theta: \mathbb{R}^d \to \mathbb{R}^k$ 是一个神经网络（MLP）

**解码器** $g_\phi: \mathbb{R}^k \to \mathbb{R}^d$ 也是一个神经网络

**训练目标**：最小化重构误差
$$\mathcal{L} = \|X - g_\phi(f_\theta(X))\|^2$$

### 3.3 为什么这"继承"了 PCA

| PCA | 自编码器（线性编码器+线性解码器） |
|-----|-------------------------------|
| $Z = XQ_k$（线性投影） | $Z = W_{\text{enc}}X$（线性层） |
| $\hat{X} = ZQ_k^T$（线性重构） | $\hat{X} = W_{\text{dec}}Z$（线性层） |
| 目标：$\min\|X - XQ_kQ_k^T\|^2$ | 目标：$\min\|X - W_{\text{dec}}W_{\text{enc}}X\|^2$ |

**当编码器和解码器都是线性的时候，最优解就是 PCA！**

证明概要：
- 令 $W = W_{\text{dec}}W_{\text{enc}}$，则目标是 $\min\|X - WX\|^2$
- 这等价于求 $W$ 使得 $\|(I-W)X\|_F^2$ 最小
- 最优的 $W$ 的前 $k$ 个奇异向量就是 PCA 的主成分

> 这就是为什么自编码器的**瓶颈层（bottleneck）** 被称为"非线性 PCA"——线性时退化为 PCA，非线性时超越 PCA。

### 3.4 带非线性激活的自编码器

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=10, latent_dim=2):
        super().__init__()
        # 编码器：输入 → 低维
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),    # 扩展（非线性变换的"中间层"）
            nn.ReLU(),                    # 引入非线性——PCA 没有这一步
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),   # 瓶颈：k 维
        )
        # 解码器：低维 → 输出
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)     # 编码
        x_hat = self.decoder(z) # 重构
        return x_hat, z

# 训练
model = Autoencoder(input_dim=10, latent_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    optimizer.zero_grad()
    x_hat, z = model(X)                    # 前向传播
    loss = nn.functional.mse_loss(x_hat, X) # 重构损失
    loss.backward()                         # 反向传播
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.4f}")

# 提取低维表示
with torch.no_grad():
    _, z = model(X)  # z 的形状: (500, 2)
```

**对比 PCA 和自编码器的输出**：

```python
# PCA 结果
X_pca, _, ratio_pca = pca_manual(X, k=2)

# 自编码器结果
with torch.no_grad():
    _, z_ae = model(X)

# 可视化对比（matplotlib）
# PCA: 方差最大的线性方向
# AE: 非线性变换学到的流形结构
```

| | PCA | 自编码器 |
|--|-----|----------|
| 映射类型 | 线性投影 $XQ_k$ | 非线性函数 $f_\theta(X)$ |
| 可学习参数 | $Q_k$（固定，$d \times k$） | $\theta$（通过反向传播优化） |
| 能捕捉 | 线性结构 | 线性 + 非线性结构 |
| 计算复杂度 | $O(d^3)$ 特征值分解 | $O(\text{参数量} \times n)$ 迭代优化 |
| 优势 | 解析解、可解释、快 | 表达能力强、可处理非线性 |
| 劣势 | 只能做线性降维 | 需要调参、可解释性差 |

---

## 四、同一条线上的下一个跳跃：VAE

### 4.1 从确定性到概率性

PCA 和上面的自编码器都是**确定性**的：给定输入 $X$，输出固定的 $Z$。

但真实数据往往有**不确定性**：
- 同一张人脸的不同角度 → 不同的潜在表示
- 同一句话的不同表达 → 相同的语义

**变分自编码器（VAE）** 把自编码器的瓶颈层从**点估计**变为**概率分布**：

```
PCA:            AE:             VAE:
X → Z (固定)    X → z (固定)    X → N(μ, σ²) → z (随机)
                                         ↓
                                  z = μ + σ ⊙ ε（重参数化）
```

### 4.2 重参数化技巧的数学本质

$$z \sim \mathcal{N}(\mu, \sigma^2 I)$$

直接采样 $z$ 无法反向传播（随机节点没有梯度）。**重参数化**把随机性移到外部：

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

现在 $z$ 是 $\mu$ 和 $\sigma$ 的**确定性函数**，梯度可以正常传播：

$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \cdot \frac{\partial z}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial z} \cdot 1$$

```python
# 重参数化技巧——5行代码
def reparameterize(mu, logvar):
    """mu: (batch, latent_dim), logvar: (batch, latent_dim)"""
    std = torch.exp(0.5 * logvar)          # σ = exp(0.5 * log(σ²))
    eps = torch.randn_like(std)            # ε ~ N(0, I)
    return mu + eps * std                  # z = μ + σ⊙ε
```

> 📖 变分推断的数学基础：[[Mathematics-Universe/06-超纲拓展/实变函数与测度论.md]]

### 4.3 VAE 损失函数：重构 + 正则

$$\mathcal{L} = \underbrace{\|X - \hat{X}\|^2}_{\text{重构损失}} + \underbrace{D_{KL}(q_\phi(z|X) \| p(z))}_{\text{KL 正则}}$$

- **重构损失**：让解码器能准确还原输入（和AE一样）
- **KL 散度**：让编码器输出的分布 $\mathcal{N}(\mu, \sigma^2)$ 接近标准正态 $\mathcal{N}(0, I)$
  - 这迫使潜在空间**连续、完整**（插值有意义）
  - 防止过拟合（潜在空间不过度记忆训练数据）

```python
def vae_loss(x_hat, x, mu, logvar):
    """标准 VAE 损失"""
    # 重构损失（MSE 或 BCE）
    recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')

    # KL 散度：N(μ,σ²) vs N(0,I)
    # 解析解（不需要蒙特卡洛采样！）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss
```

---

## 五、完整进化链

```
┌─────────────────────────────────────────────────────────────────┐
│  纯数学                                                          │
│  特征值分解 A = QΛQ^T  （手算 3×3 矩阵，考研水平）              │
│  [[Mathematics-Universe/04-线性代数/...]]                        │
└────────────────────────────┬────────────────────────────────────┘
                             │ 第一次映射：数据降维
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  PCA                                                             │
│  对协方差矩阵做特征值分解 → 投影到主成分方向                      │
│  5 行代码，解析解，不需要训练                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │ 第二步：引入非线性
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  自编码器（AE）                                                   │
│  用神经网络替代线性投影 → 可学习非线性降维                         │
│  训练目标 = MSE 重构损失，30 行 PyTorch                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ 第三步：引入概率
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  变分自编码器（VAE）                                              │
│  潜在变量从点估计 → 概率分布                                      │
│  重参数化技巧 = 把随机采样变成可微操作                            │
│  KL 散度正则 = 让潜在空间结构良好                                │
└─────────────────────────────────────────────────────────────────┘
```

**这条链上的每一步，核心数学操作始终是"矩阵分解"**：
- PCA：直接分解协方差矩阵
- AE：用神经网络"隐式"分解（编码器学到特征方向，但不显式写出 $Q$）
- VAE：在概率框架下做分解（编码器输出 $\mu$ 和 $\sigma$，描述潜在方向上的分布）

---

## 六、直觉总结

**一张图看懂**：

```
纸笔推导（数学仓库）     矩阵乘法（PCA）    神经网络训练（AE/VAE）
────────────────     ──────────────     ─────────────────────────
A = QΛQ^T           X @ Q_k            encoder(X) → z
                    (2行代码)           (30行代码 + 100轮训练)
                        │                    │
                        └──── 本质相同 ───────┘
                              ↓
                      对数据做"轴对齐"
                      保留最重要的方向
```

**一句话总结**：特征值分解告诉我们"数据的最大方差在哪条轴上"。PCA 直接用这个轴做投影。自编码器用神经网络**学**出这些轴。VAE 进一步告诉你在这些轴上，数据长什么样（均值和方差）。

---

## 七、延伸阅读

### 论文
- **PCA**：Pearson (1901) 原始论文 — 最早的降维方法
- **自编码器**：Hinton & Salakhutdinov (2006) "Reducing the Dimensionality of Data with Neural Networks" — 用栈式AE做预训练
- **VAE**：Kingma & Welling (2013) "Auto-Encoding Variational Bayes" — 重参数化技巧的原始出处

### 课程
- [MIT 6.041 Probabilistic Systems Analysis](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/) — PCA 的概率论视角
- [Stanford CS229](https://cs229.stanford.edu/) — 降维和因子分析的数学推导

### 代码
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) — 生产级PCA实现
- [PyTorch 自编码器教程](https://pytorch.org/tutorials/beginner/vae.html) — 官方VAE教程

### 关联文章
- [[SVD → 推荐系统 → 协同过滤]]（轴线A下一篇）
- [[矩阵指数 → Neural ODE → 连续深度网络]]（轴线A第三篇）
- [[特征函数 → Fourier → 注意力机制]]（轴线D）

---

## 联系网络

⬆ 上游: [[Mathematics-Universe/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md]]（特征值分解），[[Mathematics-Universe/04-线性代数/02-矩阵/矩阵详解.md]]（矩阵运算与秩），[[Mathematics-Universe/05-概率论与数理统计/04-数字特征/数字特征详解.md]]（协方差矩阵·PCA）

⬇ 下游: [[SVD → 推荐系统 → 协同过滤]]（轴线A下一篇），[[矩阵指数 → Neural ODE]]

↔ 横联: [[03-优化算法]]（PCA 是凸优化问题），[[05-损失函数]]（AE的重构损失 = MSE）

🔗 跨域: 计算机视觉（人脸编码·风格迁移），推荐系统（用户-物品矩阵分解），生物学（单细胞RNA降维）

━━━━━━━━━━━━━━
