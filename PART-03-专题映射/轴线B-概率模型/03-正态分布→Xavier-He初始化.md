# 正态分布 → Xavier/He 初始化

> 神经网络的权重初始化可以用信号的方差与二阶矩传播来分析。Xavier 在近线性激活下折中前向与反向传播，He 则计入 ReLU 对二阶矩的影响；这些推导依赖初始化时的统计假设。

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

> 📖 正态分布的完整性质：[随机变量及其分布](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布详解.md#23-正态分布-nmusigma2)

### 1.2 方差传播（Delta Method）

对于函数 $y = f(x)$，其中 $x \sim \mathcal{N}(\mu_x, \sigma_x^2)$：

$$\sigma_y^2 \approx \left(\frac{\partial f}{\partial x}\bigg|_{\mu_x}\right)^2 \sigma_x^2$$

这是**一阶Taylor展开**的应用——在均值附近用线性近似。

**多维版本**：若 $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$，$\mathbf{y} = A\mathbf{x} + \mathbf{b}$：

$$\Sigma_y = A\Sigma A^T$$

**这是神经网络初始化分析的核心工具。**

> 📖 Taylor展开的数学基础：[一元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/02-一元微分学/一元微分学详解.md#三taylor公式)

---

## 二、神经网络的信号传播

### 2.1 前向传播的方差

考虑一个简单网络：

$$a = Wx + b, \quad h = \text{ReLU}(a), \quad y = Vh + c$$

假设：
- $x$ 已经归一化：$\mathbb{E}[x] = 0, \text{Var}(x) = 1$
- 权重独立初始化并与输入独立：$W_{ij} \sim \mathcal{N}(0, \sigma_w^2)$, $V_{ij} \sim \mathcal{N}(0, \sigma_v^2)$
- 偏置初始化为 0

**第一层线性变换的方差**：

$$\text{Var}(Wx) = \text{Var}\left(\sum_j W_{ij}x_j\right)$$

由于 $W_{ij}$ 和 $x_j$ 独立：

$$= \sum_j \mathbb{E}[W_{ij}^2] \cdot \mathbb{E}[x_j^2] = \sum_j \sigma_w^2 \cdot 1 = n_{in} \sigma_w^2$$

其中 $n_{in}$ 是输入维度；这里的方差对随机初始化与输入共同取平均。$a$ 是预激活，$h$ 是激活输出，两者统计量不能混用。

**关键结果**：
$$\text{Var}(Wx) = n_{in} \cdot \sigma_w^2$$

**这意味着**：如果 $\sigma_w$ 固定，输入维度越大，输出的方差越大——信号会越来越大（或越来越小）。

### 2.2 信号消失和爆炸

```
线性预激活（输入零均值）:
    Var(a) = n_in * σ_w^2 * Var(x)

如果 σ_w = 1/sqrt(n_in):
    Var(a) = 1  ← 保持输入方差为 1 时的预激活尺度

如果 σ_w 太大:
    Var(a) >> 1  ← 尺度过大，可能使饱和激活的导数接近 0

如果 σ_w 太小:
    Var(a) << 1  ← 尺度过小，多层连乘可能使信号和梯度衰减
```

这是 Xavier 的线性传播分析起点。若 $a\sim\mathcal N(0,q)$，则 ReLU 后 $\mathbb E[h^2]=q/2$，而 $\mathop{\mathrm{Var}}\nolimits(h)=q(1/2-1/(2\pi))$；对应修正在第四节推导。

---

## 三、Xavier 初始化（Glorot & Bengio, 2010）

### 3.1 推导

目标是尽量保持**前向信号与反向梯度的尺度**。本节假设激活近似零中心且导数为 1（例如 tanh 的原点附近），故 $h\approx a$；反向还采用初始化时梯度与权重近似独立的假设。

**前向**（上面已推导）：
$$\text{Var}(h) \approx \text{Var}(a) = n_{in} \cdot \sigma_w^2 \cdot \text{Var}(x)$$

设 $\text{Var}(h) = \text{Var}(x) = 1$：
$$\sigma_w = \frac{1}{\sqrt{n_{in}}}$$

**反向传播**（对称推导）：
$$\text{Var}(\delta_{l-1}) \approx n_{out} \cdot \sigma_w^2 \cdot \text{Var}(\delta_l)$$

设 $\text{Var}(\delta_{l-1}) = \text{Var}(\delta_l) = 1$：
$$\sigma_w = \frac{1}{\sqrt{n_{out}}}$$

**在前向和反向之间折中**（扇入与扇出不同时无法同时精确满足）：
$$\sigma_w = \sqrt{\frac{2}{n_{in} + n_{out}}}$$

这是 **Xavier 初始化**（也称 Glorot 初始化）的标准差；方差是它的平方。

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
    """仅用于二维 nn.Linear 权重的 Xavier 均匀初始化。"""
    fan_in, fan_out = tensor.shape[-1], tensor.shape[-2]  # nn.Linear 权重形状为 (out, in)
    a = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.uniform_(-a, a)

def xavier_normal_(tensor):
    """仅用于二维 nn.Linear 权重的 Xavier 正态初始化。"""
    fan_in, fan_out = tensor.shape[-1], tensor.shape[-2]  # nn.Linear 权重形状为 (out, in)
    std = math.sqrt(2.0 / (fan_in + fan_out))
    with torch.no_grad():
        tensor.normal_(0, std)

# PyTorch 内置
W = torch.empty(256, 128)
torch.nn.init.xavier_uniform_(W)
# 或
torch.nn.init.xavier_normal_(W)
```

**Xavier 的推导近似激活处于零中心、斜率约为 1 的线性区**，例如原点附近的 tanh。不能把 sigmoid 的缩放因子当作 1：logistic sigmoid 在原点导数为 $1/4$ 且输出不居中，深层仍易饱和。参见 [Glorot 与 Bengio 原论文](https://proceedings.mlr.press/v9/glorot10a.html)。

---

## 四、He 初始化（He et al., 2015）

### 4.1 为什么需要新的初始化

Xavier 基于近似线性、零中心激活的方差传播分析。但现代深度学习大量使用 **ReLU**：

$$\text{ReLU}(x) = \max(0, x)$$

**ReLU 的问题**：一半的输入被置为 0。

```
输入 x ~ N(0, 1):
    ReLU(x) = 0    (概率 50%, 当 x < 0)
    ReLU(x) = x    (概率 50%, 当 x > 0)

    E[ReLU(x)]   = 0.5 * 0 + 0.5 * E[x | x > 0] = 0.5 * sqrt(2/π) ≈ 0.4   ← 均值不再为 0
    E[ReLU(x)^2] = 0.5 * E[x^2] = 0.5                                    ← 二阶矩减半
    Var(ReLU(x)) = E[ReLU(x)^2] - E[ReLU(x)]^2 = 0.5 - 0.16 ≈ 0.34
```

注意区分**二阶矩**与**方差**：ReLU 把二阶矩 E[h²] 从 1 降到 0.5，而 Var(h) = E[h²] − (E[h])² ≈ 0.34。He 的前向推导保持激活的**二阶矩**；ReLU 输出均值非零，不能将它说成保持激活方差为 1。随后独立、零均值权重使下一层预激活的方差取决于这个二阶矩。

### 4.2 推导

在零均值对称的预激活与初始化独立性假设下，ReLU 输出的二阶矩：

$$\mathbb{E}[h^2] = \frac{1}{2} n_{in} \sigma_w^2 \, \mathbb{E}[x^2]$$

（系数 $1/2$ 来自 ReLU 只让一半信号通过；He 推导用的是二阶矩而非方差）

设 $\mathbb{E}[h^2] = \mathbb{E}[x^2] = 1$：
$$\sigma_w = \sqrt{\frac{2}{n_{in}}}$$

这就是 **He 初始化**。

```python
def he_init_(tensor):
    """仅用于二维 nn.Linear 权重的 He 初始化（ReLU）。"""
    fan_in = tensor.shape[-1]  # nn.Linear 权重形状为 (out, in)，fan_in 取第 1 维
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        tensor.normal_(0, std)

# PyTorch 内置
W = torch.empty(256, 128)
torch.nn.init.kaiming_normal_(W)  # He 正态
torch.nn.init.kaiming_uniform_(W)  # He 均匀
```

### 4.3 不同激活函数对应的初始化

| 激活函数 | 二阶矩因子或局部线性化说明 | 初始化方差 | 公式 |
|---------|---------|-----------|------|
| tanh | 原点线性化约为 1；实际依赖输入尺度 | $2/(n_{in}+n_{out})$ | 基础 Xavier（可另设 gain） |
| sigmoid | 原点导数平方为 $1/16$；输出非零均值 | 无简单方差守恒通式 | 基础 Xavier 不保证深层稳定 |
| ReLU | 1/2 | $2/n_{in}$ | He |
| LeakyReLU($\alpha$) | $(1+\alpha^2)/2$ | $2/((1+\alpha^2)n_{in})$ | He 变体 |
| Swish/SiLU | 标准正态输入下二阶矩约 0.356；随尺度变化 | 需解二阶矩固定点或实测 | 不能直接取 $1/0.356$ 的增益平方 |

SiLU 不像 ReLU 那样正齐次。令输入二阶矩为 1、权重方差为 $v/n_{in}$，则预激活近似 $\sqrt v Z$（$Z\sim\mathcal N(0,1)$），应求 $\mathbb E[\mathop{\mathrm{SiLU}}\nolimits(\sqrt v Z)^2]=1$。数值积分给 $v\approx2.4297$；直接用 $v=1/0.356\approx2.81$ 会得到约 1.181 的输出二阶矩。这个固定点仍只是前向统计近似，不保证反向梯度稳定。

```python
import numpy as np

def silu_second_moment(variance, points=80):
    """Gauss-Hermite 积分：预激活 N(0, variance) 的 SiLU 输出二阶矩。"""
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    z = np.sqrt(2 * variance) * nodes
    h = z / (1 + np.exp(-z))
    return float(weights @ (h * h) / np.sqrt(np.pi))

print(silu_second_moment(1.0))                  # 约 0.355776
print(silu_second_moment(1 / silu_second_moment(1.0)))  # 约 1.181106，而非 1
print(silu_second_moment(2.4297325))            # 约 1
```

---

## 五、反向传播的方差

### 5.1 梯度消失的数学根源

考虑 loss $L$ 对 $W^{(1)}$ 的梯度：

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdots \frac{\partial h^{(1)}}{\partial W^{(1)}}$$

**反向信号的统计近似**：在初始化的独立性、零均值等近似下，单坐标梯度方差满足

$$\mathop{\mathrm{Var}}\nolimits(\delta_{l-1})\approx n_{\rm out}\sigma_w^2\,\mathbb E[\sigma'(z_l)^2]\mathop{\mathrm{Var}}\nolimits(\delta_l).$$

这里包含扇出数量、权重方差与激活导数的二阶矩；不能直接写成“各 Jacobian 元素方差的乘积”。Xavier/He 旨在改善这些统计缩放，并不保证每个样本或每个梯度方向稳定。

### 5.2 为什么 ReLU 比 sigmoid 好（初始化角度）

```
Sigmoid:  σ'(x) = σ(x)(1-σ(x))  ≤ 1/4  （最大在 x=0 处）
ReLU:     ReLU'(x) = 1 (x>0) 或 0 (x<0)

Sigmoid 激活导数 ≤ 0.25；完整 Jacobian 还包含权重矩阵
ReLU 激活导数为 1 或 0；权重与门控连乘仍可使梯度爆炸或消失

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
- Xavier + tanh：激活方差可能逐层衰减；本实验没有绘制 Xavier + ReLU
- He + ReLU：二阶矩通常保持同一量级，激活方差不必等于 1，有限宽深网络仍可能漂移
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
│  近线性、零中心激活（如原点附近的 tanh）                         │
└────────────────────────────┬────────────────────────────────────┘
                             │ ReLU 修正
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  He 初始化 (2015)                                                 │
│  σ = sqrt(2/n_in)                                                │
│  ReLU；LeakyReLU 另除以 (1 + α²)                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、直觉总结

**初始化 = 给信号传输线设定正确的阻抗**：

```
信号从输入到输出的传播：

  x₀ →[W₁]→ h₁ →[W₂]→ h₂ → ... →[W_L]→ h_L → loss

  每层: h_l = σ(W_l h_{l-1} + b_l)

  线性层二阶矩由 fan_in × Var(W) 缩放，非线性再改变其统计

  初始化根据统计近似设定 Var(W)，目标是:
  - 前向传播: 保持近线性激活的方差或 ReLU 激活的二阶矩尺度
  - 反向传播: 避免各层激活梯度尺度持续放大或缩小

  Xavier: σ² = 2/(n_in + n_out)   ← 近线性、零中心激活
  He:     σ² = 2/n_in             ← ReLU（二阶矩减半）
```

**总结**：Xavier 与 He 都来自初始化时的统计传播分析。Xavier 折中 fan_in 与 fan_out 的方差要求，He 的 fan_in 版本保持 ReLU 前向二阶矩；它们都不保证任意网络的前向与反向方差同时等于 1。

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
- [MLE → 交叉熵损失](02-MLE→交叉熵损失.md)（轴线B上一篇）
- [正态分布 → 权重初始化](03-正态分布→Xavier-He初始化.md) ← 你正在读的
- [爆炸的数学根源](../../PART-04-从理论到工程/02-梯度消失爆炸的数学根源.md)（PART-04）

---

## 联系网络

⬆ 上游: [随机变量及其分布详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布详解.md)（正态分布的性质），[一元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/02-一元微分学/一元微分学详解.md)（Taylor展开用于方差传播），[数值分析](https://github.com/CacinieP/Mathematics-Universe/blob/main/06-超纲拓展/数值分析.md)（数值稳定性）

⬇ 下游: [梯度消失爆炸的数学根源](../../PART-04-从理论到工程/02-梯度消失爆炸的数学根源.md)

↔ 横联: [02-MLE→交叉熵损失](02-MLE→交叉熵损失.md)（MLE + 正态假设 = MSE损失），[01-Bayes定理→贝叶斯神经网络](01-Bayes定理→贝叶斯神经网络.md)（权重先验的选择）

🔗 跨域: 计算机视觉（CNN的初始化），NLP（Transformer的初始化），强化学习（策略网络的初始化）

━━━━━━━━━━━━━━
