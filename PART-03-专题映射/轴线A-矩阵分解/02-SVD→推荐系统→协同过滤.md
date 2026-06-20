# SVD → 推荐系统 → 协同过滤

> 奇异值分解（SVD）比特征值分解更通用——它不需要矩阵是对称的。这个"不需要对称"的性质，恰好是推荐系统的核心。

**难度**：[标准]（需要线性代数基础 + 了解矩阵乘法）

## 一、纯数学版

### 1.1 从特征值分解到奇异值分解

**特征值分解**的局限：只对**方阵**且最好是对称矩阵有效。

真实世界中大量矩阵不是方阵的：
- 用户-物品评分矩阵：$m$ 个用户 × $n$ 个物品（$m \neq n$）
- 词-文档矩阵：$V$ 个词 × $D$ 个文档（$V \neq D$）
- 图像矩阵：$H$ 像素 × $W$ 像素（通常 $H \neq W$）

**SVD 解决了这个问题**：任何 $m \times n$ 矩阵都可以分解。

### 1.2 SVD 定义

对任意 $A \in \mathbb{R}^{m \times n}$，存在：

$$A = U\Sigma V^T$$

其中：
- $U \in \mathbb{R}^{m \times m}$：左奇异向量（列正交）
- $\Sigma \in \mathbb{R}^{m \times n}$：奇异值对角矩阵（对角元 $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$）
- $V \in \mathbb{R}^{n \times n}$：右奇异向量（列正交）

**截断SVD**（保留前 $k$ 个奇异值）：
$$A \approx U_k \Sigma_k V_k^T$$

其中 $U_k \in \mathbb{R}^{m \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$, $V_k \in \mathbb{R}^{n \times k}$。

这是**最低秩近似**——在秩为 $k$ 的所有矩阵中，$U_k\Sigma_k V_k^T$ 是离 $A$ 最近的（Frobenius范数意义下）。

> 📖 矩阵分解的背景：[[Mathematics-Universe/04-线性代数/02-矩阵/矩阵详解.md#23-矩阵分解因式分解的矩阵版]]

### 1.3 SVD 与特征值分解的关系

$$A^TA \text{ 的特征向量} = V \text{ 的列（右奇异向量）}$$
$$AA^T \text{ 的特征向量} = U \text{的列（左奇异向量）}$$
$$\text{奇异值 } \sigma_i = \sqrt{\lambda_i(A^TA)}$$

**关键洞察**：SVD 的奇异值 = 特征值分解后开方。所以 SVD 的稳定性比特征值分解更好（开方是单调函数，不改变大小顺序）。

### 1.4 手动计算示例

设 $A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \\ -1 & 1 \end{pmatrix}$（$3 \times 2$ 矩阵）

**Step 1**：$A^TA = \begin{pmatrix} 2 & 0 \\ 0 & 2 \end{pmatrix}$

**Step 2**：特征值 $\lambda_1 = 2, \lambda_2 = 2$，奇异值 $\sigma_1 = \sqrt{2}, \sigma_2 = \sqrt{2}$

**Step 3**：$V = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$（因为 $A^TA = 2I$）

**Step 4**：$U = AV\Sigma^{-1} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}$（第二列需要Gram-Schmidt正交化补全）

**SVD**：
$$A = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 1 \\ 1 & 1 & 0 \end{pmatrix} \begin{pmatrix} \sqrt{2} & 0 \\ 0 & \sqrt{2} \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$

---

## 二、第一次应用：推荐系统的数学基础

### 2.1 评分矩阵

推荐系统的核心数据结构是**用户-物品评分矩阵** $R$：

$$
R = \begin{pmatrix}
r_{11} & r_{12} & r_{13} & \cdots & r_{1n} \\
r_{21} & r_{22} & r_{23} & \cdots & r_{2n} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
r_{m1} & r_{m2} & r_{m3} & \cdots & r_{mn}
\end{pmatrix}
$$

- $m$ 个用户，$n$ 个物品（电影/商品/音乐）
- $r_{ij}$ 是用户 $i$ 对物品 $j$ 的评分（如 1-5 星）
- **问题**：这个矩阵极度稀疏——大多数用户只评过极少物品

### 2.2 核心思想：潜在因子模型

**假设**：评分可以由少数"潜在因子"线性组合生成。

```
用户偏好 × 物品属性 = 预测评分

用户 i 的偏好向量:  p_i = (喜欢动作片?, 喜欢文艺片?, 喜欢老片?, ...)
物品 j 的属性向量: q_j = (是动作片?, 是文艺片?, 是老片?, ...)

预测评分: r̂_ij = p_i · q_j
```

用矩阵表示：
$$R \approx PQ^T$$

其中：
- $P \in \mathbb{R}^{m \times k}$：用户潜在因子矩阵
- $Q \in \mathbb{R}^{n \times k}$：物品潜在因子矩阵
- $k$：潜在因子数量（通常 10-200，远小于 $m$ 和 $n$）

**这就是截断SVD！** 只是名字不同：
- 矩阵分解视角：$R \approx U_k\Sigma_k V_k^T$
- 推荐系统视角：$R \approx PQ^T$

两者等价（$\Sigma$ 可以吸收到 $P$ 或 $Q$ 中）。

### 2.3 为什么 SVD 适用于推荐系统

| 挑战 | SVD 的解决方案 |
|------|---------------|
| 评分矩阵极度稀疏 | 截断SVD用低秩假设填充缺失值 |
| 数据维度极高 | 降维到 $k$ 维潜在空间 |
| 冷启动（新用户无评分） | 用少量已知评分就能推断潜在向量 |
| 可解释性 | 每个潜在因子对应一个"隐含主题"（动作/文艺/年代...） |

> 📖 潜在因子模型与概率的联系：[[Mathematics-Universe/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布.md#23-随机变量函数的分布]]

### 2.4 手写矩阵分解推荐

```python
import torch

def matrix_factorization(R, k=10, lr=0.01, epochs=1000, mask=None):
    """
    基于SVD思想的矩阵分解推荐系统
    R: (m, n) 评分矩阵，0 表示缺失
    k: 潜在因子维度
    mask: (m, n) bool 矩阵，True 表示有评分
    """
    m, n = R.shape
    device = R.device

    # 1. 初始化（小随机值）
    P = torch.randn(m, k, device=device) * 0.1
    Q = torch.randn(n, k, device=device) * 0.1

    # 2. 评分掩码（只看有评分的元素）
    if mask is None:
        mask = R > 0

    # 3. 优化
    for epoch in range(epochs):
        # 预测矩阵
        R_hat = P @ Q.T  # (m, n)

        # 只看有评分的元素计算损失
        error = (R - R_hat) * mask  # 缺失值不参与
        loss = (error ** 2).sum() / mask.sum()

        # L2 正则（防止过拟合）
        loss += 0.01 * (P ** 2).sum() + 0.01 * (Q ** 2).sum()

        # 梯度下降
        P_grad = -2 * error @ Q + 0.02 * P       # (m, k)
        Q_grad = -2 * error.T @ P + 0.02 * Q     # (n, k)

        P -= lr * P_grad
        Q -= lr * Q_grad

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: RMSE = {(loss / mask.sum()).sqrt():.3f}")

    return P, Q

# 测试
torch.manual_seed(42)
m, n, k = 100, 50, 10
R = torch.zeros(m, n)
# 生成模拟评分
true_P = torch.randn(m, k) * 2
true_Q = torch.randn(n, k) * 2
R = (true_P @ true_Q.T + torch.randn(m, n) * 0.5).clamp(1, 5)
mask = torch.rand(m, n) > 0.7  # 30% 有评分（稀疏）

P_learned, Q_learned = matrix_factorization(R, k=10, lr=0.01, epochs=1000, mask=mask)
R_pred = P_learned @ Q_learned.T
```

**关键行解析**：

| 代码行 | 对应的数学操作 | 数学含义 |
|--------|---------------|---------|
| `P @ Q.T` | $\hat{R} = PQ^T$ | 矩阵分解的预测 |
| `(R - R_hat) * mask` | 只看有评分的元素 | 稀疏矩阵的处理 |
| `error @ Q` | $\nabla_P = -2(R - PQ^T)Q$ | 对 $P$ 的梯度 |
| `error.T @ P` | $\nabla_Q = -2(R - PQ^T)^TP$ | 对 $Q$ 的梯度 |
| `0.01 * (P**2).sum()` | $\lambda\|P\|_2^2$ | L2正则（Tikhonov正则化） |

---

## 三、第二次应用：协同过滤

### 3.1 基于用户的协同过滤

**思想**：和你口味相似的人喜欢的物品，你也可能喜欢。

**步骤**：
1. 用评分向量计算用户间的相似度（余弦相似度 = 归一化后的内积）
2. 找 $K$ 个最相似的用户
3. 加权平均他们的评分，预测你对未评分物品的评分

```python
def user_based_cf(R, target_user, target_item, k=10):
    """
    基于用户的协同过滤
    R: (m, n) 评分矩阵
    """
    # 1. 对两个用户都有评分的物品计算余弦相似度
    target_ratings = R[target_user]  # (n,)
    mask = target_ratings > 0  # 目标用户评过的

    similarities = []
    for other_user in range(R.shape[0]):
        if other_user == target_user:
            continue
        other_ratings = R[other_user]
        common = mask & (other_ratings > 0)
        if common.sum() < 2:
            continue

        # 余弦相似度 = 归一化内积
        a = target_ratings[common]
        b = other_ratings[common]
        sim = (a @ b) / (a.norm() * b.norm())  # ← 内积！线代的核心操作
        similarities.append((sim, other_user))

    # 2. 取 top-k 相似用户
    similarities.sort(reverse=True)
    top_k = similarities[:k]

    # 3. 加权平均预测
    pred = sum(sim * R[u, target_item] for sim, u in top_k)
    weight_sum = sum(abs(sim) for sim, _ in top_k)

    return pred / weight_sum if weight_sum > 0 else 0
```

### 3.2 基于物品的协同过滤

**思想**：和你之前喜欢的物品相似的物品，你也可能喜欢。

```
基于用户:  用户 ←→ 用户（相似度）→ 推荐
基于物品:  物品 ←→ 物品（相似度）→ 推荐
```

物品相似度 = 评分向量的余弦相似度 = **内积**。

**矩阵视角**：物品相似度矩阵 $S = R^TR$（未经归一化的余弦相似度）。

这就是为什么 Netflix 大奖赛的获胜方案大量使用了矩阵分解——本质上是在低维空间做相似度计算，避免了 $O(n^2)$ 的全量比较。

### 3.3 内积的深层含义

| 场景 | 内积公式 | 含义 |
|------|---------|------|
| 几何 | $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$ | 方向一致性 |
| PCA | $\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$ | 正交基 |
| 推荐系统 | $p_i \cdot q_j$ | 用户偏好 × 物品属性的匹配度 |
| 注意力 | $QK^T$ | 查询与键的相关性 |
| Transformer | $\text{softmax}(QK^T/\sqrt{d_k})V$ | 加权求和 |

> 📖 内积的完整故事：[[Mathematics-Universe/01-高中数学基础/01-函数与代数/函数与代数详解.md#22-函数复合与反函数]] 和 [[Mathematics-Universe/08-数学联系网络/跨分支深层联系.md#14-Fourier分析-无穷维线性代数]]

---

## 四、第三次应用：SVD 在 NLP 中的角色——词嵌入

### 4.1 词-文档矩阵

构造一个矩阵 $M$：
- 行 = 词汇表中的词
- 列 = 语料库中的文档
- $M_{ij}$ = 词 $i$ 在文档 $j$ 中出现的次数（或 TF-IDF 权重）

对 $M$ 做 SVD：$M \approx U_k\Sigma_k V_k^T$

**词向量** = $U_k\Sigma_k$ 的行（或直接用 $U_k$）

**文档向量** = $V_k$ 的行

这就是**潜在语义分析（LSA）** 的核心。

### 4.2 为什么 SVD 能做词嵌入

- $U_k$ 的行是词的向量表示
- 语义相近的词，在高维空间中方向也相近（因为它们在文档中的出现模式相似）
- 内积 $U_i^T U_j$ = 词 $i$ 和词 $j$ 的语义相似度

**但这和 Word2Vec 有什么关系？**

| | SVD 词嵌入 | Word2Vec |
|--|-----------|----------|
| 矩阵 | 词-文档共现矩阵 | 词-上下文共现矩阵 |
| 分解 | SVD | 神经网络（CBOW/Skip-gram） |
| 向量来源 | $U_k$（SVD的奇异向量） | 嵌入层的权重 |
| 优势 | 可解释、有理论保证 | 捕捉复杂语义、支持增量更新 |
| 劣势 | 高维稀疏、难以增量 | 超参数多、可解释性差 |

**现代深度学习中的等效操作**：

```python
# SVD 词嵌入
U, S, Vt = torch.linalg.svd(M)      # M: (vocab_size, doc_count)
word_embeddings_svd = U[:, :k] @ torch.diag(S[:k])  # (vocab, k)

# Word2Vec（神经网络的等价视角）
# 本质上是在做：给定词 w 预测上下文 c 的概率
# 最优的嵌入矩阵 = 词-上下文共现矩阵的因子分解
# 所以 word2vec ≈ 带噪声目标的 SVD
```

---

## 五、奇异值的"故事"

### 5.1 奇异值谱的意义

$$A = \underbrace{\sigma_1}_{\text{最大方向}} \mathbf{u}_1\mathbf{v}_1^T + \underbrace{\sigma_2}_{\text{第二方向}} \mathbf{u}_2\mathbf{v}_2^T + \cdots$$

每个奇异值 $\sigma_i$ 告诉你：**第 $i$ 个"最重要方向"有多重要**。

**在推荐系统中的解读**：
- $\sigma_1 \gg \sigma_2 \gg \cdots \gg \sigma_k$：用户偏好高度集中在少数几个维度上（比如"动作/文艺"二分就够用了）
- $\sigma_1 \approx \sigma_2 \approx \cdots$：用户偏好分散在很多维度上（需要更多潜在因子）

**这就是"能量集中"**：

```python
def singular_value_energy(S):
    """计算前 k 个奇异值占总能量的比例"""
    total = (S ** 2).sum()
    cumsum = torch.cumsum(S ** 2, dim=0) / total
    return cumsum

# 示例
S = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.5, 0.3, 0.1])
energy = singular_value_energy(S)
for i, e in enumerate(energy):
    print(f"前 {i+1} 个奇异值保留 {e:.1%} 信息")
# 前2个保留 ~88%，前4个保留 ~98%
```

### 5.2 奇异值的数值稳定性

奇异值的大小顺序在数值上非常稳定：
- 即使矩阵有微小的扰动（舍入误差），$\sigma_1 \geq \sigma_2 \geq \cdots$ 的顺序不会改变
- 但特征值可能因为 $A$ 不是对称的而出现复数（当 $A$ 不是方阵时，特征值根本不存在）

**SVD vs 特征值分解的稳定性对比**：

| 特性 | 特征值分解 | SVD |
|------|-----------|-----|
| 矩阵要求 | 方阵（通常对称） | 任意 $m \times n$ |
| 特征值/奇异值 | 可能复数 | 总是非负实数 |
| 排序稳定性 | 特征向量可能翻转 | 奇异值唯一（符号约定下） |
| 条件数 | $\kappa(A) = \|\lambda_{\max}\|/\|\lambda_{\min}\|$ | $\kappa(A) = \sigma_{\max}/\sigma_{\min}$ |

> 📖 数值稳定性背景：[[Mathematics-Universe/06-超纲拓展/数值分析.md]]

---

## 六、完整进化链

```
┌─────────────────────────────────────────────────────────────────┐
│  纯数学                                                          │
│  SVD: A = UΣV^T，对任意矩阵，Σ 对角线非负递减                    │
│  [[Mathematics-Universe/04-线性代数/02-矩阵/矩阵详解.md]]         │
└────────────────────────────┬────────────────────────────────────┘
                             │ 第一次映射：数据降维
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  潜在语义分析（LSA）                                              │
│  词-文档矩阵 SVD → 词向量                                        │
│  早期NLP的核心方法（2000年代主流）                                │
└────────────────────────────┬────────────────────────────────────┘
                             │ 第二次映射：推荐系统
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  矩阵分解推荐系统                                                  │
│  R ≈ PQ^T，用SVD思想填充评分矩阵的缺失值                          │
│  Netflix大奖赛的基础方法                                          │
└────────────────────────────┬────────────────────────────────────┘
                             │ 第三次映射：深度学习
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Neural Collaborative Filtering (NCF)                            │
│  用神经网络替代内积 P @ Q^T                                       │
│  可以学习非线性的用户-物品交互函数                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、直觉总结

**特征值分解 vs SVD**：

```
特征值分解:     A = QΛQ^T          ← A 必须是方阵（通常对称）
                    ↓
                PCA 直接用

SVD:            A = UΣV^T          ← A 可以是任意形状
                    ↓
              更通用：推荐系统、NLP、图像压缩、...
```

**一句话总结**：SVD 是特征值分解的"完全体"——不再要求矩阵是对称的、不再要求矩阵是方阵。这让它成为处理真实世界数据（评分矩阵、词-文档矩阵、图像）的首选工具。推荐系统把 $R \approx U\Sigma V^T$ 重新解释为"用户偏好 × 物品属性 = 评分"，现代深度学习进一步用神经网络替代了其中的线性映射。

---

## 八、延伸阅读

### 论文
- **SVD**：Eckart-Young-Mirsky 定理（1936）— 最优低秩近似的数学证明
- **矩阵分解推荐**：Koren et al. (2009) "Matrix Factorization Techniques for Recommender Systems"
- **神经协同过滤**：He et al. (2017) "Neural Collaborative Filtering" — 用MLP替代内积

### 工具
- [scikit-learn TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) — 稀疏矩阵的截断SVD
- [Surprise](http://surpriselib.com/) — 经典推荐系统库（含矩阵分解）
- [LightFM](https://github.com/lyst/lightfm) — 混合推荐系统（矩阵分解 + 内容特征）

### 关联文章
- [[特征值分解 → PCA → 自编码器]]（轴线A上一篇）
- [[矩阵指数 → Neural ODE → 连续深度网络]]（轴线A第三篇）
- [[QR分解 → 数值稳定的线性求解]]（轴线A后续）

---

## 联系网络

⬆ 上游: [[Mathematics-Universe/04-线性代数/02-矩阵/矩阵详解.md]]（矩阵分解），[[Mathematics-Universe/04-线性代数/05-特征值与特征向量/特征值与特征向量详解.md]]（SVD与特征值的关系）

⬇ 下游: [[矩阵指数 → Neural ODE → 连续深度网络]]（轴线A第三篇）

↔ 横联: [[特征值分解 → PCA → 自编码器]]（轴线A上一篇，SVD是PCA的推广）

🔗 跨域: 推荐系统（Netflix/Amazon/Spotify），NLP（LSA词嵌入），图像处理（图像压缩JPEG本质是SVD），生物信息学（基因表达数据降维）

━━━━━━━━━━━━━━
