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

> 📖 条件概率与全概率公式：[随机事件与概率详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/01-随机事件与概率/随机事件与概率详解.md#12-条件概率链)

### 1.2 用 Monty Hall 问题直觉化

三扇门，一辆车。你选门1，主持人打开门3（羊）。换不换？

$$P(\text{车在门2}|\text{主持人开3}) = \frac{P(\text{开3}|\text{车在门2}) \cdot P(\text{车在门2})}{P(\text{开3})}$$

- 先验：$P(\text{车在门}i) = 1/3$
- 似然：主持人行为提供了信息（他不会开有车的门）
- 后验：若主持人在有两扇羊门可开时等概率选择，给定开门 3 后换门赢率为 $2/3$；无论这种选择偏好，预先决定总是换门的总体赢率均为 $2/3$

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

**核心困难**：$p(\theta|D)$ 通常**无法解析计算**——后验分布的维度等于参数个数，对于有百万参数的神经网络，这个高维积分通常难以精确求值（并非数学上不可积）。

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

$$\Sigma_{\text{post}} = \left(\frac{1}{\sigma^2}X^TX + \frac{1}{\tau^2}I\right)^{-1}, \quad \mu_{\text{post}} = \Sigma_{\text{post}} \cdot \frac{1}{\sigma^2}X^Ty$$

**预测分布**（对 $w$ 积分）：

$$p(y_*|x_*, D) = \mathcal{N}(y_*; \mu_{\text{post}}^T x_*, \sigma^2 + x_*^T \Sigma_{\text{post}} x_*)$$

注意预测方差有两项：
- $\sigma^2$：数据噪声（固有不确定性）
- $x_*^T \Sigma_{\text{post}} x_*$：模型不确定性（权重分布带来的额外不确定性）

**模型方差由 $x_*^T\Sigma_{\text{post}}x_*$ 决定**：被训练设计矩阵充分约束的方向方差较小，缺少观测的方向较大。它还随输入的长度变化，不能只按与训练点的欧氏距离或数据密度判断置信度。

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
    Sigma_prior_inv = (1 / tau**2) * torch.eye(d, device=X.device, dtype=X.dtype)       # 先验精度
    Sigma_likelihood_inv = (1 / sigma**2) * (X.T @ X)    # 似然精度
    Sigma_post = torch.inverse(Sigma_prior_inv + Sigma_likelihood_inv)

    mu_post = Sigma_post @ (1 / sigma**2 * X.T @ y)

    return mu_post, Sigma_post

def predict_bayesian(x_new, mu_post, Sigma_post, sigma=0.1):
    """贝叶斯预测（返回均值和不确定性）"""
    y_mean = x_new @ mu_post
    # 预测方差 = 数据噪声 + 模型不确定性
    y_var = sigma**2 + ((x_new @ Sigma_post) * x_new).sum(dim=-1)  # 每个输入的边缘方差
    return y_mean, y_var
```

---

## 三、第二次应用：贝叶斯神经网络

### 3.1 为什么不能用共轭先验

在线性回归中，高斯先验 + 高斯似然 = 高斯后验（解析解）。

在神经网络中：
- 参数是 $W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}, \ldots$（多层非线性）
- 即便回归似然仍为高斯，其均值对权重也通常是非线性的，因而失去高斯共轭结构
- **后验 $p(\theta|D)$ 无法解析计算**

贝叶斯神经网络在 1990 年代已有研究；大规模近似推断的计算成本一直是挑战。

### 3.2 变分推断：用可学习的分布近似后验

**核心思想**：不用真正的后验 $p(\theta|D)$，而是用一个**简单的分布族** $q_\phi(\theta)$（通常是高斯）去近似它。

$$\theta \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$$

其中 $\mu_\phi$ 与 $\rho_\phi$ 通常直接作为变分参数学习，令 $\sigma_\phi=\mathop{\mathrm{softplus}}\nolimits(\rho_\phi)$ 保证尺度正性。

**训练目标**：最小化 KL 散度

$$\min_\phi D_{KL}(q_\phi(\theta) \| p(\theta|D))$$

但 $p(\theta|D)$ 本身不可计算。用Bayes公式重写：

$$D_{KL}(q_\phi(\theta) \| p(\theta|D)) = \mathbb{E}_{q_\phi}[\log q_\phi(\theta)] - \mathbb{E}_{q_\phi}[\log p(D|\theta)] - \mathbb{E}_{q_\phi}[\log p(\theta)] + \log p(D)$$

忽略与 $\phi$ 无关的 $\log p(D)$，得到**证据下界（ELBO）**：

$$\mathcal{L}(\phi) = \underbrace{\mathbb{E}_{q_\phi(\theta)}[\log p(D|\theta)]}_{\text{拟合数据}} - \underbrace{D_{KL}(q_\phi(\theta) \| p(\theta))}_{\text{正则化}}$$

**直觉**：
- 第一项：让神经网络在"可能的权重"上都拟合数据（期望）
- 第二项：让权重分布接近先验（防止过拟合）

### 3.3 重参数化技巧（概率版）

对每个权重参数，将随机采样重写为：

$$w_i = \mu_i + \sigma_i \odot \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, 1)$$

这样 $w_i$ 是 $\mu_i$ 和 $\sigma_i$ 的**确定性函数**，梯度可以正常反向传播。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho)
        bias_sigma = F.softplus(self.bias_rho)
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        total = self.weight_mu.new_zeros(())
        for mu, rho in [(self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)]:
            q = Normal(mu, F.softplus(rho))
            prior = Normal(torch.zeros_like(mu), torch.ones_like(mu))
            total = total + kl_divergence(q, prior).sum()
        return total

class BayesianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

    def kl_divergence(self):
        return self.fc1.kl_divergence() + self.fc2.kl_divergence()

    def elbo_loss(self, x, y, dataset_size=None, n_samples=5):
        # 平均 NLL + KL / 全数据集大小；mini-batch 不能用批大小代替 N。
        N = len(x) if dataset_size is None else dataset_size
        if n_samples < 1 or N < len(x):
            raise ValueError("invalid sample count or dataset size")
        nll = torch.stack([F.cross_entropy(self(x), y) for _ in range(n_samples)]).mean()
        return nll + self.kl_divergence() / N

model = BayesianMLP(4, 8, 3)
x, y = torch.randn(6, 4), torch.randint(0, 3, (6,))
loss = model.elbo_loss(x, y, dataset_size=60)
loss.backward()
```

### 3.4 贝叶斯神经网络 vs 普通神经网络

| 特性 | 普通神经网络 | 贝叶斯神经网络 |
|------|------------|---------------|
| 权重 | 固定值（点估计） | 分布（$w \sim \mathcal{N}(\mu, \sigma^2)$） |
| 预测 | 单一输出 | 多次采样取平均 = 分布 |
| 不确定性 | 可用输出概率/校准或集成估计，但不自动表示参数后验 | 可估计（是否可靠仍取决于先验、近似质量和校准） |
| 参数量 | $N$ | $2N$（均值+方差） |
| 训练速度 | 快（一次前向+反向） | 慢（多次采样 + KL 计算） |
| 正则化 | Dropout / L2（启发式） | KL 项来自 ELBO；不保证泛化或校准 |

---

## 四、第三次应用：MC Dropout — 不精确但实用的近似

### 4.1 深度学习的"朴素贝叶斯"

Gal & Ghahramani (2016) 在特定模型、正则化与变分族假设下，将 Dropout 解释为近似贝叶斯推断。推理时重复采样掩码可估计预测波动，但并非任意带 Dropout 网络都等价于精确后验采样。

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
- 训练时优化的是**集成模型**（有限个掩码子网络的共享参数近似集成）
- 推理时多次前向传播 = 从掩码诱导的近似分布中采样

**这不需要修改训练代码，只需要在推理时多加几行**：

```python
def mc_dropout_predict(model, x, n_samples=20):
    """仅启用 Dropout；成功或失败都恢复各子模块原有 train/eval 状态。"""
    if not isinstance(n_samples, int) or n_samples < 2:
        raise ValueError("n_samples must be an integer >= 2")
    original_modes = {m: m.training for m in model.modules()}
    predictions = []
    try:
        model.eval()  # 保持 BatchNorm 的运行统计不变
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                m.train()
        with torch.no_grad():
            for _ in range(n_samples):
                predictions.append(F.softmax(model(x), dim=-1))
        predictions = torch.stack(predictions)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        entropy = -(mean * torch.log(mean + 1e-8)).sum(dim=-1)
        return mean, std, entropy
    finally:
        for m, training in original_modes.items():
            m.training = training

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

    示例：测试出现训练未覆盖的书写风格 → 可能有认知不确定性
    示例：输入分辨率过低，使同一可观测图像既可能是 3 也可能是 8 → 条件标签存在偶然不确定性
    光照变化本身不等于标签噪声；两类不确定性取决于可观测信息与模型假设。
```

---

## 五、Bayes 在深度学习其他地方的影子

### 5.1 正则化 = 先验

下表先验对应**总负对数似然**加表中完整惩罚项；若改用平均损失，系数还需乘样本数。

| 正则化方法 | Bayes 视角 | 对应的先验 |
|-----------|-----------|-----------|
| L2 正则（普通 SGD 下可写成权重衰减） | $\frac\lambda2\|w\|^2$ 惩罚 | $p(w) = \mathcal{N}(0, \lambda^{-1}I)$ |
| L1 正则 | $\lambda\|w\|_1$ 惩罚 | $p(w) = \text{Laplace}(0, \lambda^{-1})$ |
| Dropout | 权重的随机子集 | 不是严格的贝叶斯先验 |
| 某些贝叶斯稀疏模型 | 可用 spike-and-slab 先验 | 普通幅值剪枝不自动等价于此先验 |

**关键洞见**：显式参数惩罚在合适归一化下可对应 MAP 先验；Dropout、数据增强等其他机制不能一概等同于先验选择。

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
MC Dropout:      多次掩码前向传播 = 近似预测采样
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
- **贝叶斯神经网络**：[Blundell et al. (2015), Weight Uncertainty in Neural Networks](https://proceedings.mlr.press/v37/blundell15.html) — 重参数化技巧首次系统用于 BNN 权重（Bayes by Backprop；技巧本身更早出自 Kingma & Welling 2013 / Rezende et al. 2014）

### 课程
- [Bayesian Deep Learning Workshop](https://bayesiandeeplearning.org/) — 贝叶斯深度学习专题报告与论文
- [CS229T: Trustworthy ML](https://cs229t.stanford.edu/) — Stanford 不确定性量化专题

### 关联文章
- [MLE → 交叉熵损失](02-MLE→交叉熵损失.md)（轴线B下一篇）
- [He 初始化](03-正态分布→Xavier-He初始化.md)（轴线B第三篇）
- [重参数化 → VAE](../轴线E-信息论/02-KL散度→信息瓶颈.md)（轴线A已覆盖）

---

## 联系网络

⬆ 上游: [随机事件与概率详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/01-随机事件与概率/随机事件与概率详解.md)（Bayes公式与全概率），[数理统计基础详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/06-数理统计基础/数理统计基础详解.md)（参数估计·MLE）

⬇ 下游: [MLE → 交叉熵损失](02-MLE→交叉熵损失.md)（轴线B下一篇），[重参数化 → VAE](../轴线E-信息论/02-KL散度→信息瓶颈.md)

↔ 横联: [04-正则化](../../PART-02-深度学习核心/04-正则化.md)（贝叶斯视角下的正则化 = 先验），[01-特征值分解](../轴线A-矩阵分解/01-特征值分解→PCA→自编码器.md)（贝叶斯线性回归的共轭推导）

🔗 跨域: 自动驾驶（不确定性量化用于安全决策），医疗AI（模型自信度评估），主动学习（用不确定性指导采样）

━━━━━━━━━━━━━━
