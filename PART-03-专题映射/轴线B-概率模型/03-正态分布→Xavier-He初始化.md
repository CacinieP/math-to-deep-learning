# 正态分布 → Xavier/He 初始化

> 神经网络的权重初始化不是"随便设个小随机数"，而是用正态分布和方差传播定理推导出来的。Xavier 和 He 初始化的差异本质上是一个微积分的导数计算。

**难度**：[标准]（需要概率论 + 了解前向传播 + 一点随机过程）

## 一、纯数学版

### 1.1 正态分布的三个角色

$$\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**角色一：概率模型** — 描述连续随机变量

$$\text{身高} \sim \mathcal{N}(170, 9^2) \quad (\text{cm})$$

**角色二：误差模型** — 描述测量噪声

$$y_{\text{obs}} = y_{\text{true}} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

**角色三：初始化分布** — 描述神经网络训练前的权重

$$w_{ij} \sim \mathcal{N}(0, \sigma_w^2)$$

在深度学习中的角色三，核心问题变成了：**$\sigma_w$ 应该设多大？**

> 📖 正态分布的完整性质：[[Mathematics-Universe/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布.md#23-正态分布最重要的分布]]

### 1.2 方差传播（Delta Method）

对于函数 $y = f(x)$，其中 $x \sim \mathcal{N}(\mu_x, \sigma_x^2)$：

$$\sigma_y^2 \approx \left(\frac{\partial f}{\partial x}\bigg|_{\mu_x}\right)^2 \sigma_x^2$$

这是**一阶Taylor展开**的应用——在均值附近用线性近似。

**多维版本**：若 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$，$\mathbf{y} = A\mathbf{x} + \mathbf{b}$：

$$\Sigma_y = A\Sigma A^T$$

**这是神经网络初始化分析的核心工具。**

> 📖 Taylor展开的数学基础：[[Mathematics-Universe/03-高等数学/02-一元微分学/一元微分学详解.md#24-Taylor展开万能近似]]

---

## 二、神经网络的信号传播

### 2.1 前向传播的方差

考虑一个简单网络：

$$h = \text{ReLU}(Wx + b), \quad y = Vh + c$$

假设：
- $x$ 已经归一化：$\mathbb{E}[x] = 0, \text{Var}(x) = 1$
- 权重初始化：$W_{ij} \sim \mathcal{N}(0, \sigma_w^2)$, $V_{ij} \sim \mathcal{N}(0, \sigma_v^2)$
- 偏置初始化为 0

**第一层线性变换的方差**：

$$\text{Var}(Wx) = \text{Var}\left(\sum_j W_{ij}x_j\right)$$

由于 $W_{ij}$ 和 $x_j$ 独立：

$$= \sum_j \mathbb{E}[W_{ij}^2] \cdot \mathbb{E}[x_j^2] = \sum_j \sigma_w^2 \cdot 1 = n_{in} \sigma_w^2$$

其中 $n_{in}$ 是输入维度。

**关键结果**：
$$\text{Var}(Wx) = n_{in} \cdot \sigma_w^2$$

**这意味着**：如果 $\sigma_w$ 固定，输入维度越大，输出的方差越大——信号会越来越大（或越来越小）。

### 2.2 信号消失和爆炸

```
前向传播:
    Var(h) = n_in * σ_w^2 * Var(x)

如果 σ_w = 1/sqrt(n_in):
    Var(h) = 1  ← 信号保持稳定！

如果 σ_w 太大:
    Var(h) >> 1  ← 信号爆炸（饱和激活函数→梯度为0）

如果 σ_w 太小:
    Var(h) << 1  ← 信号消失（接近0→梯度消失）
```

**这就是 Xavier 初始化的来源。**

---

## 三、Xavier 初始化（Glorot & Bengio, 2010）

### 3.1 推导

目标是让**前向传播和反向传播的方差都保持为 1**。

**前向**（上面已推导）：
$$\text{Var}(h) = n_{in} \cdot \sigma_w^2 \cdot \text{Var}(x)$$

设 $\text{Var}(h) = \text{Var}(x) = 1$：
$$\sigma_w = \frac{1}{\sqrt{n_{in}}}$$

**反向传播**（对称推导）：
$$\text{Var}(\delta_{l-1}) = n_{out} \cdot \sigma_w^2 \cdot \text{Var}(\delta_l)$$

设 $\text{Var}(\delta_{l-1}) = \text{Var}(\delta_l) = 1$：
$$\sigma_w = \frac{1}{\sqrt{n_{out}}}$$

**同时满足前向和反向**：
$$\sigma_w = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

这就是 **Xavier 初始化**（也称 Glorot 初始化）的方差。

### 3.2 均匀分布版本

为了实际采样，用均匀分布替代正态分布（方差相同）：

$$W \sim U\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

均匀分布 $U[-a, a]$ 的方差 = $a^2/3$：

$$\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}} \Rightarrow a = \sqrt{\frac{6}{n_{in} + n_{out}}}$$

### 3.3 代码

```python
import torch
import math

def xavier_init_(tensor):
    """Xavier 均匀初始化"""
    fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
    a = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-a, a)

def xavier_normal_(tensor):
    """Xavier 正态初始化"""
    fan_in, fan_out = tensor.shape[-2], tensor.shape[-1]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.normal_(0, std)

# PyTorch 内置
W = torch.empty(256, 128)
torch.nn.init.xavier_uniform_(W)
# 或
torch.nn.init.xavier_normal_(W)
```

**Xavier 初始化适用于**：tanh、sigmoid、softsign（S 型激活函数）

---

## 四、He 初始化（He et al., 2015）

### 4.1 为什么需要新的初始化

Xavier 是为 **tanh/sigmoid** 设计的。但现代深度学习大量使用 **ReLU**：

$$\text{ReLU}(x) = \max(0, x)$$

**ReLU 的问题**：一半的输入被置为 0。

```
输入 x ~ N(0, 1):
    ReLU(x) = 0    (概率 50%, 当 x < 0)
    ReLU(x) = x    (概率 50%, 当 x > 0)

    E[ReLU(x)] = 0.5 * 0 + 0.5 * E[x | x > 0] = 0.5 * sqrt(2/π) ≈ 0.4
    Var(ReLU(x)) = E[ReLU(x)^2] - E[ReLU(x)]^2 = 0.5 - 0.16 = 0.34
```

**ReLU 把方差从 1 降到了约 0.5**——信号衰减了一半。

### 4.2 推导

在 ReLU 下，前向传播的方差：

$$\text{Var}(h) = \frac{1}{2} n_{in} \sigma_w^2 \text{Var}(x)$$

（系数 $1/2$ 来自 ReLU 只让一半信号通过）

设 $\text{Var}(h) = \text{Var}(x) = 1$：
$$\sigma_w = \sqrt{\frac{2}{n_{in}}}$$

这就是 **He 初始化**。

```python
def he_init_(tensor):
    """He 初始化（适用于 ReLU）"""
    fan_in = tensor.shape[-2]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0, std)

# PyTorch 内置
W = torch.empty(256, 128)
torch.nn.init.kaiming_normal_(W)  # He 正态
torch.nn.init.kaiming_uniform_(W)  # He 均匀
```

### 4.3 不同激活函数对应的初始化

| 激活函数 | 方差因子 | 初始化方差 | 公式 |
|---------|---------|-----------|------|
| tanh | 1 | $2/(n_{in}+n_{out})$ | Xavier |
| sigmoid | 1 | $2/(n_{in}+n_{out})$ | Xavier |
| ReLU | 1/2 | $2/n_{in}$ | He |
| LeakyReLU($\alpha$) | $(1+\alpha^2)/2$ | $2/((1+\alpha^2)n_{in})$ | He 变体 |
| Swish/SiLU | ~1.78 | $1.78/n_{in}$ | 推导更复杂 |

---

## 五、反向传播的方差

### 5.1 梯度消失的数学根源

考虑 loss $L$ 对 $W^{(1)}$ 的梯度：

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdots \frac{\partial h^{(1)}}{\partial W^{(1)}}$$

**链式法则的乘积效应**：

$$\text{Var}\left(\frac{\partial L}{\partial W^{(1)}}\right) \approx \prod_{l=1}^L \text{Var}\left(\frac{\partial h^{(l)}}{\partial h^{(l-1)}}\right) \cdot \text{Var}\left(\frac{\partial h^{(1)}}{\partial W^{(1)}}\right)$$

- 如果每层的 $\text{Var}(\text{Jacobian}) < 1$：深层网络 → 梯度指数衰减 → **梯度消失**
- 如果每层的 $\text{Var}(\text{Jacobian}) > 1$：深层网络 → 梯度指数增长 → **梯度爆炸**

**Xavier/He 初始化通过控制权重方差，把每层的 Jacobian 方差控制在 1 附近，防止梯度在传播中指数衰减或增长。**

### 5.2 为什么 ReLU 比 sigmoid 好（初始化角度）

```
Sigmoid:  σ'(x) = σ(x)(1-σ(x))  ≤ 1/4  （最大在 x=0 处）
ReLU:     ReLU'(x) = 1 (x>0) 或 0 (x<0)

Sigmoid 的 Jacobian:  每项 ≤ 0.25，100层后: 0.25^100 ≈ 10^-61
ReLU 的 Jacobian:     每项 = 1（或0），100层后: 1^100 = 1

这是 ReLU 解决梯度消失问题的原因之一（不是全部原因）
```

---

## 六、实验验证

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_with_init(init_fn, activation, n_layers=50, hidden_dim=256):
    """测试不同初始化下的信号传播"""
    torch.manual_seed(42)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            for _ in range(n_layers):
                layer = nn.Linear(hidden_dim, hidden_dim)
                init_fn(layer.weight)
                nn.init.zeros_(layer.bias)
                layers.append(layer)
                layers.append(activation())
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            return self.layers(x)

    model = MLP()

    # 测试信号传播
    x = torch.randn(1, hidden_dim)
    vars = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                vars.append(x.var().item())
            else:
                x = layer(x)
                vars.append(x.var().item())

    return vars

# 对比三种初始化
results = {
    'Xavier (tanh)': train_with_init(
        nn.init.xavier_uniform_, nn.Tanh),
    'He (ReLU)': train_with_init(
        nn.init.kaiming_normal_, nn.ReLU),
    'Random (σ=0.01)': train_with_init(
        lambda w: nn.init.normal_(w, 0, 0.01), nn.ReLU),
}

for name, vars in results.items():
    plt.plot(vars, label=name, alpha=0.7)

plt.xlabel('Layer')
plt.ylabel('Activation Variance')
plt.yscale('log')
plt.legend()
plt.title('信号方差随层数的传播')
plt.savefig('init_comparison.png', dpi=150)
```

**预期结果**：
- Xavier + ReLU：方差逐渐衰减（ReLU 损失一半信号，Xavier 没有补偿）
- He + ReLU：方差稳定在 1 附近
- 随机小初始化 + ReLU：方差指数衰减到接近 0（梯度消失）

---

## 七、完整映射图

```
┌─────────────────────────────────────────────────────────────────┐
│  纯数学                                                          │
│  正态分布 N(0, σ²)  +  Taylor展开（一阶近似）                    │
│  [[Mathematics-Universe/05-概率论与数理统计/02-随机变量/...]]     │
│  [[Mathematics-Universe/03-高等数学/02-一元微分学/...]]          │
└────────────────────────────┬────────────────────────────────────┘
                             │ 方差传播
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  信号传播分析                                                      │
│  Var(Wx) = n_in * σ_w^2 * Var(x)                                │
│  设 Var = 1 → σ_w = 1/sqrt(n_in)                                │
└────────────────────────────┬────────────────────────────────────┘
                             │ 前向 + 反向
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Xavier 初始化 (2010)                                             │
│  σ = sqrt(2/(n_in + n_out))                                      │
│  适用于 tanh / sigmoid                                           │
└────────────────────────────┬────────────────────────────────────┘
                             │ ReLU 修正
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  He 初始化 (2015)                                                 │
│  σ = sqrt(2/n_in)                                                │
│  适用于 ReLU / LeakyReLU                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、直觉总结

**初始化 = 给信号传输线设定正确的阻抗**：

```
信号从输入到输出的传播：

  x₀ →[W₁]→ h₁ →[W₂]→ h₂ → ... →[W_L]→ h_L → loss

  每层: h_l = σ(W_l h_{l-1} + b_l)

  信号方差: Var(h_l) = Var(h_{l-1}) × (Jacobian的方差) × Var(W)

  初始化就是设定 Var(W) 使得:
  - 前向传播: Var(h_L) ≈ Var(x)   （信号不爆炸不消失）
  - 反向传播: Var(∇_W L) ≈ 1     （梯度稳定）

  Xavier: σ² = 2/(n_in + n_out)   ← tanh/sigmoid
  He:     σ² = 2/n_in             ← ReLU（考虑ReLU损失一半信号）
```

**一句话总结**：Xavier 和 He 初始化不是经验公式——它们来自"前向传播方差为1 + 反向传播方差为1"这两个约束条件的数学推导。选择哪一个取决于你的激活函数（因为不同激活函数对信号的缩放因子不同）。

---

## 九、延伸阅读

### 论文
- **Xavier**：Glorot & Bengio (2010) "Understanding the difficulty of training deep feedforward neural networks"
- **He**：He et al. (2015) "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
- **信号传播分析**：Saxe et al. (2013) "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"

### 工具
- [PyTorch `torch.nn.init`](https://pytorch.org/docs/stable/nn.init.html) — 所有初始化方法的官方实现
- [torchinfo](https://github.com/tyleryep/torchinfo) — 查看每层的输入输出维度，帮助你计算 fan_in/fan_out

### 关联文章
- [[MLE → 交叉熵损失]]（轴线B上一篇）
- [[正态分布 → 权重初始化]] ← 你正在读的
- [[梯度消失/爆炸的数学根源]]（PART-04）

---

## 联系网络

⬆ 上游: [[Mathematics-Universe/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布详解.md]]（正态分布的性质），[[Mathematics-Universe/03-高等数学/02-一元微分学/一元微分学详解.md]]（Taylor展开用于方差传播），[[Mathematics-Universe/06-超纲拓展/数值分析.md]]（数值稳定性）

⬇ 下游: [[PART-04/梯度消失爆炸的数学根源]]

↔ 横联: [[02-MLE→交叉熵损失]]（MLE + 正态假设 = MSE损失），[[01-Bayes定理→贝叶斯神经网络]]（权重先验的选择）

🔗 跨域: 计算机视觉（CNN的初始化），NLP（Transformer的初始化），强化学习（策略网络的初始化）

━━━━━━━━━━━━━━
