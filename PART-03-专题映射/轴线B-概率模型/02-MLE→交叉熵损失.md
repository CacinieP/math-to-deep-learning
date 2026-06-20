# MLE → 交叉熵损失

> 分类任务用交叉熵损失不是"约定俗成"，而是最大似然估计的直接结果。如果你从 MLE 出发推导，会**必然**得到交叉熵——这就是数学的必然性。

**难度**：[标准]（需要概率论 + 了解softmax）

## 一、纯数学版

### 1.1 最大似然估计（MLE）

给定数据集 $D = \{(x_1, y_1), \ldots, (x_n, y_n)\}$ 和参数化模型 $p(y|x; \theta)$：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \prod_{i=1}^n p(y_i|x_i; \theta)$$

取对数（乘积变求和，不影响极值点位置）：

$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(y_i|x_i; \theta)$$

取负号（最大化 → 最小化）：

$$\hat{\theta}_{\text{MLE}} = \arg\min_\theta \underbrace{-\sum_{i=1}^n \log p(y_i|x_i; \theta)}_{\text{负对数似然 (NLL)}}$$

> 📖 MLE 的完整推导与性质：[[Mathematics-Universe/05-概率论与数理统计/06-数理统计基础/数理统计基础详解.md#43-参数估计]]

### 1.2 MLE 的核心思想

**让模型分配给"真实标签"的概率尽可能大。**

- 如果模型对真实标签的预测概率是 0.99 → $\log(0.99) \approx -0.01$（损失很小）
- 如果模型对真实标签的预测概率是 0.01 → $\log(0.01) = -4.6$（损失很大）
- 如果模型对真实标签的预测概率是 0 → $\log(0) = -\infty$（无穷大损失）

**MLE 等价于"惩罚模型把概率给错标签"**。

---

## 二、从 MLE 到交叉熵

### 2.1 分类问题的似然

对于 $C$ 类分类，模型输出一个概率分布（通过 softmax）：

$$p(y=c|x; \theta) = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{j=1}^C e^{z_j}}$$

其中 $z = Wx + b$ 是 logits（未归一化的分数）。

**MLE 目标**：
$$\mathcal{L}_{\text{MLE}} = -\sum_{i=1}^n \log p(y_i|x_i; \theta)$$

### 2.2 交叉熵的定义

两个概率分布 $p$（真实）和 $q$（预测）的**交叉熵**：

$$H(p, q) = -\sum_c p(c) \log q(c)$$

在分类问题中：
- $p$ 是**one-hot 编码**的真实标签（$p_c = 1$ 对真实类别，否则 0）
- $q$ 是模型的 softmax 输出

$$H(p, q) = -\sum_{c=1}^C p_c \log q_c = -\log q_{y_{\text{true}}}$$

**这正是 MLE 的负对数似然！**

```
MLE:       -Σ log p(y_i | x_i; θ)
                    ↓
交叉熵:   -Σ_c p_c log q_c    (p = one-hot, q = softmax输出)
                    ↓
                    完全等价
```

### 2.3 分类损失的完整推导

```python
import torch
import torch.nn.functional as F

# Step 1: 模型输出 logits（未归一化分数）
logits = model(x)  # (batch, C)

# Step 2: softmax → 概率分布
probs = F.softmax(logits, dim=-1)  # (batch, C)

# Step 3: 取真实类别的概率
true_class_probs = probs[range(batch_size), y]  # (batch,)

# Step 4: 负对数似然 = 交叉熵
nll = -torch.log(true_class_probs)  # (batch,)

# Step 5: 取平均
loss = nll.mean()
```

**这就是 `F.cross_entropy` 在内部做的事**——而且它把 softmax + NLL 合并成了一步（数值更稳定）。

### 2.4 为什么不用 MSE

一个常见的新手问题：**"分类问题为什么不用 MSE？"**

```
MSE:   loss = (y_pred - y_true)^2
交叉熵: loss = -log(p(y_true|x))
```

假设真实标签是类别 1，模型输出（softmax后）：
- 类别 1 概率 = 0.9
- MSE 损失（one-hot [0,1,0] vs [0.1, 0.9, 0]）= $(0-0.1)^2 + (1-0.9)^2 + (0-0)^2 = 0.02$
- 交叉熵损失 = $-\log(0.9) \approx 0.105$

模型输出：
- 类别 1 概率 = 0.99
- MSE = 0.002
- 交叉熵 = $-\log(0.99) \approx 0.010$

**MSE 对"已经很确定"的改进不敏感**：
- 从 0.9 → 0.99，MSE 减少了 90%（0.02 → 0.002）
- 但从 0.9 → 0.99，交叉熵只减少了约 52%（0.105 → 0.010）

**等等，这不是说 MSE 更好吗？** 不——问题在于梯度：

```python
# 交叉熵 + softmax 的梯度（对 logits）
# dL/dz_c = softmax(z)_c - 1_{c=y_true}
grad_ce = probs - one_hot  # (batch, C)

# MSE 的梯度（对 logits，经过 softmax 的 chain rule）
# dL/dz = (softmax(z) - one_hot) ⊙ softmax(z) ⊙ (1 - softmax(z))
grad_mse = (probs - one_hot) * probs * (1 - probs)  # (batch, C)
```

**关键差异**：MSE 的梯度被 $\text{softmax}(z) \odot (1-\text{softmax}(z))$ 缩放。

- 当模型已经很确定（$p \approx 1$）时，$\text{softmax} \odot (1-\text{softmax}) \approx 0$
- **梯度消失！** 模型在"已经做对"的区域几乎不学习

**交叉熵没有这个问题**——梯度始终是 $p_c - \mathbb{1}_{c=y}$，当 $p_c$ 接近 1 时梯度接近 0（这是对的——已经很好了，不需要再学）。

---

## 三、第三次应用：信息论解释

### 3.1 交叉熵 = 编码一个事件所需的比特数

Shannon 信息论中，事件 $c$ 的自信息：
$$I(c) = -\log p(c)$$

编码 $c$ 所需的最优比特数 = $-\log p(c)$。

**交叉熵** $H(p, q) = -\sum p(c)\log q(c)$ 的含义：
- 如果真实分布是 $p$，但你用 $q$ 来编码
- 平均每个事件需要的比特数 = $H(p, q)$
- 当 $q = p$ 时，$H(p, p) = H(p) = -\sum p(c)\log p(c)$ = 熵（最优编码）

**在分类中的含义**：
- 真实标签的 one-hot 编码 = $p$（我们不知道真实概率，就用 one-hot 近似）
- 模型的 softmax 输出 = $q$（模型的"编码方案"）
- 交叉熵 = 用模型的编码方案来表示真实标签的"信息代价"

**当模型预测很确定但预测错误时**（$p_c = 0.99$，真实类是别的）：
- $-\log(0.01) \approx 4.6$（很大的损失）
- 意味着模型需要"很多比特"才能表达这个标签——编码效率极低

### 3.2 交叉熵 vs KL 散度

$$D_{KL}(p \| q) = \sum_c p(c) \log \frac{p(c)}{q(c)} = H(p, q) - H(p)$$

在分类问题中，$p$ 是 one-hot 的固定分布，所以 $H(p)$ 是常数（与模型无关）。

$$\min_\theta H(p, q_\theta) = \min_\theta \left[D_{KL}(p \| q_\theta) + H(p)\right]$$

**最小化交叉熵 = 最小化 KL 散度**（因为 $H(p)$ 是常数）。

**直觉**：让模型的预测分布 $q_\theta$ 尽可能接近真实分布 $p$。

### 3.3 从 BCE 到 BCEWithLogits

二分类是 $C=2$ 的特例。用 $y \in \{0, 1\}$ 表示真实标签，$p$ 表示模型预测的正类概率：

$$H = -y\log p - (1-y)\log(1-p)$$

这就是**二元交叉熵（BCE）**。

但直接用 softmax/sigmoid + BCE 有**数值不稳定**的问题（$\log(0) = -\infty$）。

**解决方案**：把 sigmoid + BCE 合并成一个操作，直接在 logits 上计算：

$$\text{BCEWithLogits}(z, y) = \max(z, 0) - yz + \log(1 + e^{-|z|})$$

这是一个**数值稳定**的等价形式（没有显式的 softmax 和 log）。

```python
# 不推荐（数值不稳定）
loss = F.binary_cross_entropy(torch.sigmoid(logits), targets)

# 推荐（数值稳定，PyTorch 内部自动合并）
loss = F.binary_cross_entropy_with_logits(logits, targets)
```

---

## 四、完整映射图

```
┌─────────────────────────────────────────────────────────────────┐
│  纯数学                                                          │
│  MLE: θ* = argmax Σ log p(y_i | x_i; θ)                        │
│  [[Mathematics-Universe/05-概率论与数理统计/06-数理统计基础/...]] │
└────────────────────────────┬────────────────────────────────────┘
                             │ 取对数 + 取负
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  负对数似然 (NLL)                                                │
│  L(θ) = -Σ log p(y_i | x_i; θ)                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ p(y|x) = softmax(logits)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  交叉熵损失                                                       │
│  L = -Σ_c y_c log(p_c) = -log(softmax(z)_y_true)                │
│  F.cross_entropy(logits, targets)                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ 等价关系
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  KL 散度最小化                                                    │
│  min_θ KL(one_hot || softmax(z_θ))                               │
│  "让模型预测分布贴近真实标签分布"                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、直觉总结

**MLE → 交叉熵 的推导只有三步**：

```
1. MLE 目标: max_θ Σ log p(y_i | x_i; θ)
                         ↓
2. 分类模型: p(y=c|x) = softmax(z)_c
                         ↓
3. 展开: -Σ log softmax(z_c) where c = y_true
                         ↓
                 = CrossEntropy(logits, labels)
```

**为什么这个推导很重要**：

- 如果你是从"实践"入门的，你学到的是"分类用交叉熵"
- 如果你是从"理论"入门的，你学到的是"MLE"
- **两者是同一个公式**——这让你不再需要记忆"分类用什么损失"，而是从第一性原理推导出来

**当你有新的任务时（比如标签是分布的而不是one-hot的）**，不需要问"用什么损失"，只需要：
1. 写出似然 $p(y|x; \theta)$
2. 取负对数 → 这就是你的损失函数

这就是 MLE 框架的威力——**任何概率模型都能自动给出损失函数**。

---

## 六、延伸阅读

### 论文
- **交叉熵与分类**：最早的信息论应用到分类
- **标签平滑**：Szegedy et al. (2016) — 用 soften 的标签（而非 one-hot）减少过拟合，本质是修改 $p(y)$ 使其不那么极端

### 代码
- [PyTorch `F.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html) — 合并 softmax + NLL + label smoothing
- [PyTorch `F.cross_entropy_with_logits`](https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html) — 二分类的数值稳定版本

### 关联文章
- [[Bayes 定理 → 贝叶斯神经网络]]（轴线B上一篇）
- [[正态分布 → Xavier/He 初始化]]（轴线B第三篇）
- [[KL 散度 → VAEs]]（轴线A已覆盖）

---

## 联系网络

⬆ 上游: [[Mathematics-Universe/05-概率论与数理统计/06-数理统计基础/数理统计基础详解.md]]（MLE 的数学推导），[[Mathematics-Universe/05-概率论与数理统计/01-随机事件与概率/随机事件与概率详解.md]]（Bayes 公式 → KL 散度），[[Mathematics-Universe/04-信息论]]（熵与交叉熵的定义）

⬇ 下游: [[正态分布 → Xavier/He 初始化]]（轴线B第三篇——交叉熵 + softmax 的初始化）

↔ 横联: [[01-Bayes定理→贝叶斯神经网络]]（MLE vs 贝叶斯推断的对偶关系），[[04-损失函数]]（PART-02中的损失函数全景）

🔗 跨域: NLP（语言建模的交叉熵），推荐系统（排序损失），强化学习（策略梯度的对数似然）

━━━━━━━━━━━━━━
