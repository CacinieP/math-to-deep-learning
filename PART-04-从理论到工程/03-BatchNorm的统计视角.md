# BatchNorm 的统计视角

> BatchNorm 把每个特征在 mini-batch 上**去均值、除标准差**——这是把激活分布**白化**（whitening）的简化版。统计学上，它在做"分布对齐"：让每层激活的均值方差稳定，缓解内部协变量偏移，让深网络能用大学习率训练。

**难度**：[进阶]（需要均值/方差/协方差 + 归一化直觉）

## 一、纯数学：白化与归一化

### 1.1 标准化的统计意义

随机变量 $X$ 的标准化：

$$\tilde X = \frac{X - \mathbb{E}[X]}{\sqrt{\text{Var}(X)}}$$

标准化后 $\mathbb{E}[\tilde X] = 0$，$\text{Var}(\tilde X) = 1$。**消除尺度和偏移差异**，让不同特征在同一量纲下比较。

> 📖 期望与方差：[[Mathematics-Universe/05-概率论与数理统计/04-数字特征/数字特征详解.md]]

### 1.2 白化（多元情形）

向量 $\mathbf{X}$ 的白化：

$$\tilde{\mathbf{X}} = \Sigma^{-1/2}(\mathbf{X} - \boldsymbol\mu)$$

其中 $\Sigma$ 是协方差矩阵。白化后各分量**不相关且方差为 1**。

**完整白化的代价**：要算协方差矩阵的逆平方根 $\Sigma^{-1/2}$，对高维特征计算量巨大。BatchNorm 是其**简化版**。

### 1.3 BatchNorm = 逐维标准化

BN 不算完整协方差，只对每个特征维度**独立**标准化：

$$\tilde x_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$$

其中 $\mu_j, \sigma_j^2$ 是 batch 内第 $j$ 维特征的均值方差。**忽略特征间相关性**，只对齐边缘分布——计算廉价。

---

## 二、内部协变量偏移（ICS）

### 2.1 问题陈述

深网络里，每层激活的分布在训练中不断变化（因为前层权重在变）。后层要不断适应新分布——这叫**内部协变量偏移（Internal Covariate Shift）**。

**症状**：学习率必须小（否则震荡），训练慢，深网络难收敛。

### 2.2 BN 的缓解

BN 强制每层激活的均值方差稳定（固定为 0、1，再由可学参数 $\gamma, \beta$ 微调）。后层看到的分布不再剧烈漂移，可以用更大学习率。

### 2.3 争议：BN 真的是因为 ICS 吗

后续研究（Santurkar et al., 2018）发现：BN 提升训练的真正原因可能不是"消除 ICS"，而是**让损失面更平滑**（梯度更稳定、Lipschitz 常数更小）。但无论机制，BN 确实有效。

---

## 三、BN 的完整公式

### 3.1 训练时（用 batch 统计）

对 batch $B = \{x_1, \ldots, x_m\}$，第 $j$ 维：

$$\mu_j = \frac{1}{m}\sum_{i=1}^m x_{i,j}, \quad \sigma_j^2 = \frac{1}{m}\sum_{i=1}^m (x_{i,j} - \mu_j)^2$$

$$\hat x_{i,j} = \frac{x_{i,j} - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}$$

$$y_{i,j} = \gamma_j \hat x_{i,j} + \beta_j$$

$\gamma_j, \beta_j$ 是**可学参数**，让网络能恢复任意均值方差（不强制标准化）。

### 3.2 推理时（用总体统计）

测试时 batch 可能很小（甚至 1），不能用 batch 统计。BN 用训练时累积的**移动平均** $\bar\mu, \bar\sigma^2$：

$$\hat x = \frac{x - \bar\mu}{\sqrt{\bar\sigma^2 + \epsilon}}$$

```python
# model.train() 用 batch 统计; model.eval() 用移动平均
model.train()  # 训练模式
model.eval()   # 推理模式
```

### 3.3 小 batch 的问题

batch 太小（如 2）时，batch 统计 $\mu, \sigma^2$ 估计不准，BN 失效。**这是 LayerNorm / GroupNorm 兴起的原因**——它们不依赖 batch 维度。

---

## 四、归一化家族对比

| 方法 | 归一化维度 | 依赖 batch | 典型场景 |
|------|-----------|-----------|---------|
| BatchNorm | batch + 空间 | 是 | CNN 图像 |
| LayerNorm | 特征维 | 否 | Transformer/RNN |
| InstanceNorm | 空间维 | 否 | 风格迁移 |
| GroupNorm | 特征分组 | 否 | 小 batch CNN |

```
        batch  特征  空间
BN:       ■      □    ■     (在 batch+空间 上归一化)
LN:       □      ■    □     (在特征 上归一化)
```

> 📖 LayerNorm 在 Transformer：[[轴线D/03-正交基→残差与变换]]

---

## 五、工程实现层

### 5.1 一行使用

```python
import torch.nn as nn

bn = nn.BatchNorm1d(128)     # 对 128 维特征做 BN
ln = nn.LayerNorm(512)       # Transformer 标配
```

### 5.2 手写 BN（理解内部）

```python
import torch

class BatchNorm1d:
    def __init__(self, dim, momentum=0.1, eps=1e-5):
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.momentum, self.eps = momentum, eps
    def __call__(self, x, training=True):
        if training:
            mean = x.mean(0)
            var = x.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean
                self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_hat + self.beta
```

### 5.3 train/eval 必须切换

```python
model.train()   # 启用 dropout + BN 用 batch 统计
# 训练...
model.eval()    # 关闭 dropout + BN 用移动平均
# 推理...  ← 忘记 eval() 是常见 bug
```

---

## 六、BN 的副作用：轻微正则

BN 的 batch 统计引入了**随机性**（每个 mini-batch 的 $\mu, \sigma$ 不同），这有轻微正则效果——类似 Dropout。所以用 BN 的网络往往可以减少或去掉 Dropout。

> 📖 正则化：[[PART-02/04-正则化]]

---

## 七、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  标准化: (X-μ)/σ;  白化: Σ^{-1/2}(X-μ)                     │
│  协方差对角化 ⇒ 去相关                                      │
│  [[Mathematics-Universe/05-概率论与数理统计/04-数字特征/...]] │
└──────────────────────────┬───────────────────────────────┘
                           │ 逐维标准化(简化白化)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  BatchNorm                                                  │
│  训练用 batch 统计, 推理用移动平均                          │
│  缓解 ICS, 让损失面平滑, 大学习率可行                       │
│  可学 γ,β 恢复任意分布; 副作用: 轻微正则                    │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  nn.BatchNorm1d/2d; nn.LayerNorm; train()/eval() 切换      │
└──────────────────────────────────────────────────────────┘
```

---

## 八、直觉总结

```
BN = 每个 mini-batch 把激活去均值除标准差
   = 简化版白化(忽略特征间相关性)
   = 让每层分布稳定, 后层好学

副作用: batch 随机性带来轻微正则
陷阱:   batch 太小失效; 推理必须 eval()
```

**BN 是分布对齐的简化工程实现。** 统计上是标准化，工程上是稳定训练的关键组件。

---

## 九、延伸阅读

### 论文
- **Ioffe & Szegedy (2015)** "Batch Normalization"——BN 原文
- **Santurkar et al. (2018)** "How Does Batch Normalization Help Optimization?"——重新解释 BN
- **Ba et al. (2016)** "Layer Normalization"

### 关联文章
- [[01-数值稳定性]]，[[02-梯度消失爆炸的数学根源]]（BN 缓解这些问题）
- [[PART-02/04-正则化]]（BN 的正则副作用）

---

## 联系网络

⬆ 上游：[[Mathematics-Universe/05-概率论与数理统计/04-数字特征/数字特征详解.md]]（均值方差协方差）

⬇ 下游：[[04-注意力机制的线性代数本质]]，[[05-Transformer的谱分析]]（Transformer 里 LayerNorm）

↔ 横联：[[轴线A/01-特征值分解→PCA→自编码器]]（PCA 是完整白化）

🔗 跨域：信号处理（自动增益控制）、统计学（z-score 标准化）

━━━━━━━━━━━━━━
