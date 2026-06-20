# Bayes 定理 → 贝叶斯神经网络

> Bayes 公式从骰子游戏到神经网络权重分布的哲学跨越——"先验信念 + 数据证据 → 更新后的信念"这一信息流，就是贝叶斯神经网络的完整训练逻辑。

**难度**：[标准]（需要概率论基础 + 了解神经网络前向传播）

## 一、纯数学版

### 1.1 Bayes 公式

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

- $P(A)$：**先验概率**——在看到证据之前对 $A$ 的信念
- $P(B|A)$：**似然**——如果 $A$ 为真，看到 $B$ 的概率
- $P(A|B)$：**后验概率**——看到 $B$ 之后对 $A$ 的更新信念

**全概率公式**（Bayes 公式的分母来源）：

$$P(B) = \sum_i P(B|A_i)P(A_i) \quad (\text{若 } \{A_i\} \text{ 是完备事件组})$$

> 📖 条件概率与全概率公式：[[Mathematics-Universe/05-概率论与数理统计/01-随机事件与概率/随机事件与概率详解.md#12-条件概率与独立性]]

### 1.2 用 Monty Hall 问题直觉化

三扇门，一辆车。你选门1，主持人打开门3（羊）。换不换？

$$P(\text{车在门2}|\text{主持人开3}) = \frac{P(\text{开3}|\text{车在门2}) \cdot P(\text{车在门2})}{P(\text{开3})}$$

- 先验：$P(\text{车在门}i) = 1/3$
- 似然：主持人行为提供了信息（他不会开有车的门）
- 后验：换门赢率 = $2/3$

**直觉**：主持人知道答案，他的行为是"证据"。Bayes 公式教你如何把这个证据更新到你的信念中。

### 1.3 连续形式：贝叶斯推断

对于参数 $\theta$ 和数据 $D$：

$$p(\theta|D) = \frac{p(D|\theta) \cdot p(\theta)}{p(D)}$$

| 符号 | 含义 | 说明 |
|------|------|------|
| $p(\theta)$ | 参数先验 | 训练前对权重的信念 |
| $p(D|\theta)$ | 似然 | 给定权重，看到数据的概率 |
| $p(\theta|D)$ | 参数后验 | 训练后对权重的信念 |
| $p(D)$ | 证据（边际似然） | 对所有可能的 $\theta$ 积分 |

**核心困难**：$p(\theta|D)$ 通常**无法解析计算**——后验分布的维度等于参数个数，对于有百万参数的神经网络，这在高维空间上不可积。

---

## 二、第一次应用：贝叶斯线性回归

### 2.1 从确定性到概率性

**普通线性回归**：$y = w^Tx + b$，$w$ 是一个固定的向量。

**贝叶斯线性回归**：$w$ 是一个**随机变量**，服从某个分布。

```
普通:     y = w*x + b          w 是固定值（通过MLE学到）
贝叶斯:   y = w*x + b          w ~ N(μ, σ²)    ← 分布！
训练前:   w ~ N(0, I)          先验（认为权重接近0）
训练后:   w ~ N(μ*, σ*²)      后验（数据更新了信念）
预测:     p(y|x) = ∫p(y|x,w)p(w|D)dw    ← 积分所有可能的w
```

### 2.2 共轭先验的优雅

当似然 $p(y|w) = \mathcal{N}(w^Tx, \sigma^2)$，先验 $p(w) = \mathcal{N}(0, \tau^2I)$ 时：

**后验也是高斯分布**！这就是**共轭先验**的威力。

$$p(w|D) = \mathcal{N}(\mu_{\text{post}}, \Sigma_{\text{post}})$$

其中：

$$\Sigma_{\text{post}} = \left(\frac{1}{\sigma^2}XX^T + \frac{1}{\tau^2}I\right)^{-1}, \quad \mu_{\text{post}} = \Sigma_{\text{post}} \cdot \frac{1}{\sigma^2}Xy$$

**预测分布**（对 $w$ 积分）：

$$p(y_*|x_*, D) = \mathcal{N}(y_*; \mu_{\text{post}}^T x_*, \sigma^2 + x_*^T \Sigma_{\text{post}} x_*)$$

注意预测方差有两项：
- $\sigma^2$：数据噪声（固有不确定性）
- $x_*^T \Sigma_{\text{post}} x_*$：模型不确定性（权重分布带来的额外不确定性）

**在训练数据附近**，$x_*$ 落在 $X$ 的列空间中 → $\Sigma_{\text{post}}$ 的影响小 → 预测置信高

**在训练数据之外**，$x_*$ 远离 $X$ → $\Sigma_{\text{post}}$ 的影响大 → 预测置信低

这就是贝叶斯方法的**不确定性量化**能力。

```python
import torch

def bayesian_linear_regression(X, y, tau=1.0, sigma=0.1):
    """
    共轭先验的贝叶斯线性回归
    X: (n, d), y: (n,)
    返回后验均值 μ_post 和后验协方差 Σ_post
    """
    # 先验: w ~ N(0, τ²I)
    # 似然: y|w ~ N(Xw, σ²I)
    # 后验: w|D ~ N(μ_post, Σ_post)

    n, d = X.shape
    Sigma_prior_inv = (1 / tau**2) * torch.eye(d)       # 先验精度
    Sigma_likelihood_inv = (1 / sigma**2) * (X.T @ X)    # 似然精度
    Sigma_post = torch.inverse(Sigma_prior_inv + Sigma_likelihood_inv)

    mu_post = Sigma_post @ (1 / sigma**2 * X.T @ y)

    return mu_post, Sigma_post

def predict_bayesian(x_new, mu_post, Sigma_post, sigma=0.1):
    """贝叶斯预测（返回均值和不确定性）"""
    y_mean = x_new @ mu_post
    # 预测方差 = 数据噪声 + 模型不确定性
    y_var = sigma**2 + x_new @ Sigma_post @ x_new.T
    return y_mean, y_var
```

---

## 三、第二次应用：贝叶斯神经网络

### 3.1 为什么不能用共轭先验

在线性回归中，高斯先验 + 高斯似然 = 高斯后验（解析解）。

在神经网络中：
- 参数是 $W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, \ldots$（多层非线性）
- 似然 $p(y|x, \theta)$ 不是高斯（经过ReLU/Softmax）
- **后验 $p(\theta|D)$ 无法解析计算**

这就是为什么神经网络在2010年之前不是"贝叶斯的"——计算不可行。

### 3.2 变分推断：用可学习的分布近似后验

**核心思想**：不用真正的后验 $p(\theta|D)$，而是用一个**简单的分布族** $q_\phi(\theta)$（通常是高斯）去近似它。

$$\theta \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$$

其中 $\mu_\phi$ 和 $\sigma_\phi$ 是神经网络的**输出**（通过softplus保证正性）。

**训练目标**：最小化 KL 散度

$$\min_\phi D_{KL}(q_\phi(\theta) \| p(\theta|D))$$

但 $p(\theta|D)$ 本身不可计算。用Bayes公式重写：

$$D_{KL}(q_\phi(\theta) \| p(\theta|D)) = \mathbb{E}_{q_\phi}[\log q_\phi(\theta)] - \mathbb{E}_{q_\phi}[\log p(D|\theta)] - \log p(D)$$

忽略与 $\phi$ 无关的 $\log p(D)$，得到**证据下界（ELBO）**：

$$\mathcal{L}(\phi) = \underbrace{\mathbb{E}_{q_\phi(\theta)}[\log p(D|\theta)]}_{\text{拟合数据}} - \underbrace{D_{KL}(q_\phi(\theta) \| p(\theta))}_{\text{正则化}}$$

**直觉**：
- 第一项：让神经网络在"可能的权重"上都拟合数据（期望）
- 第二项：让权重分布接近先验（防止过拟合）

### 3.3 重参数化技巧（概率版）

对每个权重参数，不使用随机采样，而是：

$$w_i = \mu_i + \sigma_i \odot \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 1)$$

这样 $w_i$ 是 $\mu_i$ 和 $\sigma_i$ 的**确定性函数**，梯度可以正常反向传播。

```python
class BayesianLinear(nn.Module):
    """贝叶斯线性层——权重是随机变量"""
    def __init__(self, in_features, out_features):
        super().__init__()
        # 每个权重有均值和对数方差两个参数
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.zeros(out_features))

        # 先验：N(0, 1)
        self.prior = torch.distributions.Normal(0, 1)

    def forward(self, x):
        # 重参数化：从 rho 得到 sigma（保证正性）
        weight_sigma = torch.log(1 + torch.exp(self.weight_rho))  # softplus
        bias_sigma = torch.log(1 + torch.exp(self.bias_rho))

        # 采样（重参数化）
        weight_eps = torch.randn_like(self.weight_sigma)
        bias_eps = torch.randn_like(self.bias_sigma)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """KL(q(w) || p(w))"""
        q_weight = Normal(self.weight_mu, torch.log(1 + torch.exp(self.weight_rho)))
        q_bias = Normal(self.bias_mu, torch.log(1 + torch.exp(self.bias_rho)))

        kl_w = kl_divergence(q_weight, self.prior).sum()
        kl_b = kl_divergence(q_bias, self.prior).sum()
        return kl_w + kl_b

class BayesianMLP(nn.Module):
    """贝叶斯神经网络"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def elbo_loss(self, x, y, n_samples=5):
        """
        证据下界（ELBO）损失
        n_samples: 蒙特卡洛采样次数
        """
        log_likelihood = 0
        kl = self.kl_divergence()

        for _ in range(n_samples):
            y_pred = self.forward(x)
            log_likelihood += F.cross_entropy(y_pred, y, reduction='sum')

        log_likelihood /= n_samples

        # ELBO = E[log p(D|θ)] - KL(q||p)
        # 最小化 -ELBO = KL - E[log p(D|θ)]
        return kl / len(x) - log_likelihood / len(x)
```

### 3.4 贝叶斯神经网络 vs 普通神经网络

| 特性 | 普通神经网络 | 贝叶斯神经网络 |
|------|------------|---------------|
| 权重 | 固定值（点估计） | 分布（$w \sim \mathcal{N}(\mu, \sigma^2)$） |
| 预测 | 单一输出 | 多次采样取平均 = 分布 |
| 不确定性 | 无（Softmax 过于自信） | 有（对陌生输入给出低置信度） |
| 参数量 | $N$ | $2N$（均值+方差） |
| 训练速度 | 快（一次前向+反向） | 慢（多次采样 + KL 计算） |
| 正则化 | Dropout / L2（启发式） | KL散度（理论保证） |

---

## 四、第三次应用：MC Dropout — 不精确但实用的近似

### 4.1 深度学习的"朴素贝叶斯"

Gal & Ghahramani (2016) 发现：**Dropout 在推理时保持开启，等价于贝叶斯神经网络的变分推断**。

```
训练时（标准Dropout）:
    h = ReLU(W₁x + b₁)
    h = dropout(h, p=0.5)     ← 随机屏蔽
    out = W₂h + b₂

推理时（MC Dropout）:
    y₁ = model(x, dropout=True)   ← 采样1
    y₂ = model(x, dropout=True)   ← 采样2
    y₃ = model(x, dropout=True)   ← 采样3
    y_mean = (y₁ + y₂ + y₃) / 3   ← 蒙特卡洛平均
    y_std = std(y₁, y₂, y₃)       ← 不确定性估计
```

**数学解释**：
- Dropout = 在权重上施加了伯努利噪声
- 训练时优化的是**集成模型**（无穷多个子网络的均值）
- 推理时多次前向传播 = 从后验分布中采样

**这不需要修改训练代码，只需要在推理时多加几行**：

```python
def mc_dropout_predict(model, x, n_samples=20):
    """
    蒙特卡洛 Dropout 推理
    不需要修改训练——只需要推理时保持 dropout 开启
    """
    model.train()  # 关键：保持 dropout 活跃！
    predictions = []

    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=-1)
            predictions.append(probs)

    predictions = torch.stack(predictions)  # (n_samples, batch, num_classes)

    mean = predictions.mean(dim=0)          # 预测均值
    std = predictions.std(dim=0)            # 预测不确定性
    entropy = -(mean * torch.log(mean + 1e-8)).sum(dim=-1)  # 熵

    return mean, std, entropy
```

### 4.2 不确定性的两种类型

贝叶斯方法天然区分两种不确定性：

| 类型 | 含义 | 在模型中的表现 | 如何处理 |
|------|------|---------------|---------|
| **认知不确定性（Epistemic）** | 模型不知道（数据不足） | 训练数据少的区域 → 高方差 | 收集更多数据 |
| **偶然不确定性（Aleatoric）** | 数据本身有噪声 | 标签重叠/模糊的区域 → 高输出方差 | 改损失函数（回归中预测方差） |

```
认知不确定性：                     偶然不确定性：
    数据少                              标签模糊
    → 模型没学过                        → 噪声不可避免
    → 通过数据缓解                      → 通过建模缓解

    示例：在 MNIST 上给一张模糊的 "3" 写个 "8" → 认知不确定性（训练集中没有类似的）
    示例：两张人脸图片，同一个人在不同光照下 → 偶然不确定性（标签本身就有噪声）
```

---

## 五、Bayes 在深度学习其他地方的影子

### 5.1 正则化 = 先验

| 正则化方法 | Bayes 视角 | 对应的先验 |
|-----------|-----------|-----------|
| L2 正则（权重衰减） | $\|w\|^2$ 惩罚 | $p(w) = \mathcal{N}(0, \lambda^{-1}I)$ |
| L1 正则 | $\|w\|_1$ 惩罚 | $p(w) = \text{Laplace}(0, \lambda^{-1})$ |
| Dropout | 权重的随机子集 | 不是严格的贝叶斯先验 |
| 权重剪枝 | 稀疏先验 | $p(w) = \text{spike-and-slab}$ |

**关键洞见**：你在训练中使用的所有"正则化技巧"，在 Bayes 视角下都是**先验分布**的选择。

### 5.2 训练 = 求后验

| 范式 | 参数 | 目标 | 解 |
|------|------|------|-----|
| 频率学派（MLE） | 固定值 | $\max_\theta p(D|\theta)$ | 优化（梯度下降） |
| 贝叶斯学派 | 分布 | $p(\theta|D) \propto p(D|\theta)p(\theta)$ | 推断（变分/采样） |
| 贝叶斯深度学习 | 分布 | $\min_\phi D_{KL}(q_\phi(\theta) \| p(\theta|D))$ | 变分推断 |

---

## 六、直觉总结

**Bayes 公式的深度学习翻译**：

```
纯数学:          P(假设|证据) = P(证据|假设) × P(假设) / P(证据)
                    ↓
贝叶斯线性回归:   p(w|数据) ∝ p(数据|w) × p(w)
                    ↓
贝叶斯神经网络:   q(w|D) ≈ N(μ_φ, σ_φ)    ← 用变分推断近似
                    ↓
MC Dropout:      多次前向传播采样 = 从后验采样
                    ↓
                    本质一句话：
                    权重不是学到一个固定值，
                    而是学到一个" plausible 的范围"。
                    这个范围量化了模型的不确定性。
```

**一句话总结**：Bayes 公式教我们"用新证据更新信念"。在深度学习中，这意味着权重不是学到一个固定值，而是学到一个分布——这个分布本身就能告诉模型"什么我不知道"。

---

## 七、延伸阅读

### 论文
- **贝叶斯深度学习综述**：Gal (2016) "Uncertainty in Deep Learning" — PhD thesis, UCL
- **MC Dropout**：Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
- **贝叶斯神经网络**：Blundell et al. (2015) "Weight Uncertainty in Neural Networks" — 重参数化技巧的首次应用
- **Dirichlet扩散模型**：Wang et al. (2023) "Probabilistic Deep Learning" — 不确定性校准

### 课程
- [UCL Bayesian Deep Learning](https://www.bayes-deep-learning.net/) — Gal 的课程讲义
- [CS229T: Trustworthy ML](https://cs229t.stanford.edu/) — Stanford 不确定性量化专题

### 关联文章
- [[MLE → 交叉熵损失]]（轴线B下一篇）
- [[正态分布 → Xavier/He 初始化]]（轴线B第三篇）
- [[重参数化 → VAE]]（轴线A已覆盖）

---

## 联系网络

⬆ 上游: [[Mathematics-Universe/05-概率论与数理统计/01-随机事件与概率/随机事件与概率详解.md]]（Bayes公式与全概率），[[Mathematics-Universe/05-概率论与数理统计/06-数理统计基础/数理统计基础详解.md]]（参数估计·MLE）

⬇ 下游: [[MLE → 交叉熵损失]]（轴线B下一篇），[[重参数化 → VAE]]

↔ 横联: [[04-正则化]]（贝叶斯视角下的正则化 = 先验），[[01-特征值分解]]（贝叶斯线性回归的共轭推导）

🔗 跨域: 自动驾驶（不确定性量化用于安全决策），医疗AI（模型自信度评估），主动学习（用不确定性指导采样）

━━━━━━━━━━━━━━
