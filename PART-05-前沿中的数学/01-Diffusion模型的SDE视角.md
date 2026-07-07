# Diffusion 模型的 SDE 视角

> Diffusion 模型的数学骨架是**随机微分方程（SDE）**：前向过程用 SDE 把数据逐步加噪变成纯噪声，反向过程用另一个 SDE 从噪声还原数据。这条 SDE 的解是 **Ornstein-Uhlenbeck 过程**的变种——概率论里最经典的扩散过程。

**难度**：[前沿]（需要随机过程 + SDE + 概率论）

## 一、纯数学：随机微分方程

### 1.1 从 ODE 到 SDE

常微分方程（ODE）：$dx/dt = f(x, t)$——确定性轨迹。

随机微分方程（SDE）加入布朗运动 $W_t$：

$$dx = f(x, t)\,dt + g(x, t)\,dW_t$$

- $f$：漂移项（确定性趋势）
- $g$：扩散项（随机扰动）
- $W_t$：标准布朗运动（$dW_t \sim \mathcal{N}(0, dt)$）

> 📖 随机过程基础：[[Mathematics-Universe/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布详解.md]]

### 1.2 Ornstein-Uhlenbeck 过程

经典 OU 过程：

$$dx = -\theta x\,dt + \sigma\,dW_t$$

**性质**：把任意初值拉向 0（漂移项），同时被噪声扰动（扩散项）。**平稳分布是高斯** $\mathcal{N}(0, \sigma^2/(2\theta))$。

**意义**：Diffusion 模型的前向过程就是 OU 的推广——把数据分布平滑地变成标准高斯。

---

## 二、Diffusion 的前向 SDE

### 2.1 逐渐加噪

给定数据 $x_0 \sim q_{\text{data}}$，前向过程定义连续加噪：

$$dx = -\frac{1}{2}\beta(t) x\,dt + \sqrt{\beta(t)}\,dW_t$$

其中 $\beta(t)$ 是噪声调度（控制加噪速度）。

**性质**：
- 漂移项 $-\frac{1}{2}\beta(t) x$ 把信号拉向 0
- 扩散项 $\sqrt{\beta(t)}$ 加噪声
- $t \to \infty$ 时 $x_t \sim \mathcal{N}(0, I)$（纯噪声）

### 2.2 离散版本（DDPM）

离散化得到 DDPM 的转移：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

**闭式跳步**：可以直接从 $x_0$ 采样任意 $x_t$：

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

其中 $\bar\alpha_t = \prod_{s\leq t}(1-\beta_s)$。**这是训练时算损失的关键**。

---

## 三、反向 SDE：从噪声生成数据

### 3.1 反向扩散定理

Anderson (1982) 证明：每个 SDE 都对应一个**反向 SDE**。前向 SDE 的反向：

$$dx = \left[-f(x,t) + g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)\,d\bar W_t$$

**关键项**：$\nabla_x \log p_t(x)$——**分数函数**（score function），指向概率密度上升方向。

### 3.2 用神经网络学分数

反向 SDE 需要分数函数，但解析形式未知。用网络 $s_\theta(x, t) \approx \nabla_x \log p_t(x)$ 逼近。

**训练目标**（分数匹配）：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[\|s_\theta(x_t, t) - \nabla_{x_t}\log p(x_t|x_0)\|^2\right]$$

**化简**：由于 $p(x_t|x_0)$ 是高斯，分数有闭式：

$$\nabla_{x_t}\log p(x_t|x_0) = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$$

**这就是为什么 Diffusion 训练就是"预测噪声 $\epsilon$"**。

### 3.3 DDPM 的等价形式

DDPM 实际训练网络 $\epsilon_\theta$ 预测噪声：

$$\mathcal{L}_{\text{DDPM}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)\|^2\right]$$

**与分数匹配等价**——同一个数学的两种写法。

---

## 四、ODE 视角：概率流

### 4.1 概率流 ODE

Fokker-Planck 方程把 SDE 转成密度的演化。去掉随机项，得到**等价的确定性 ODE**：

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

**意义**：Diffusion 既能用 SDE 采样（随机、快），也能用 ODE 采样（确定性、可逆、似然可算）。

### 4.2 与 Normalizing Flow 的统一

概率流 ODE 是**连续深度的 Normalizing Flow**——可逆、雅可比行列式可算。这把 Diffusion 和流模型统一在同一框架。

> 📖 流模型：[[02-流模型与微分同胚]]

---

## 五、工程实现层

### 5.1 DDPM 训练循环

```python
import torch
import torch.nn as nn

class Diffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise, noise

    def loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.size(0),))
        xt, noise = self.q_sample(x0, t)
        pred = model(xt, t)
        return nn.functional.mse_loss(pred, noise)   # 预测噪声
```

### 5.2 DDPM 采样（反向 SDE 离散化）

```python
@torch.no_grad()
def sample(self, model, shape):
    x = torch.randn(shape)  # 从纯噪声开始
    for t in reversed(range(self.T)):
        z = torch.randn(shape) if t > 0 else 0
        eps = model(x, torch.full((shape[0],), t))
        # 反向一步: x_{t-1} = 均值 + 噪声
        mean = (1/self.alphas[t]).sqrt() * (x - (1-self.alphas[t]).sqrt()/(1-self.alpha_bars[t]).sqrt() * eps)
        x = mean + self.betas[t].sqrt() * z
    return x
```

---

## 六、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  SDE: dx = f dt + g dW;  OU 过程;  Fokker-Planck           │
│  Anderson 反向定理: 反向 SDE 需要分数函数 ∇log p            │
│  [[Mathematics-Universe/05-概率论与数理统计/...]]           │
└──────────────────────────┬───────────────────────────────┘
                           │ 离散化 + 学分数
                           ▼
┌──────────────────────────────────────────────────────────┐
│  Diffusion 模型                                             │
│  前向: 数据→噪声(OU 变种);  反向: 噪声→数据                 │
│  训练: 预测噪声 ε ≡ 学分数函数                              │
│  采样: SDE(随机快) 或 ODE(确定性, 可算似然)                 │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  q_sample 闭式加噪; mse_loss(ε, ε_θ); 反向循环采样          │
└──────────────────────────────────────────────────────────┘
```

---

## 七、直觉总结

```
前向 SDE: 数据 → 噪声(OU 拉向高斯)
反向 SDE: 噪声 → 数据, 需要分数函数 ∇log p
训练:     学网络逼近分数 ≡ 预测噪声 ε
采样:     解反向 SDE(随机) 或 概率流 ODE(确定)

Diffusion = 用神经网络参数化的随机微分方程
```

**Diffusion 模型的本质是参数化的 SDE。** 数学是随机过程，工程是噪声预测网络。

---

## 八、延伸阅读

### 论文
- **Song et al. (2021)** "Score-Based Generative Modeling Through SDEs"——SDE 视角的奠基
- **Ho et al. (2020)** "Denoising Diffusion Probabilistic Models"（DDPM）
- **Anderson (1982)** "Reverse-time Diffusion Equation"——反向 SDE 理论

### 关联文章
- [[02-流模型与微分同胚]]（概率流 ODE 统一两者）
- [[轴线B/01-Bayes定理→贝叶斯神经网络]]（贝叶斯视角的 Diffusion）

---

## 联系网络

⬆ 上游：[[Mathematics-Universe/05-概率论与数理统计/02-随机变量及其分布/随机变量及其分布详解.md]]（高斯过程），[[Mathematics-Universe/03-高等数学/07-常微分方程/常微分方程详解.md]]（ODE）

⬇ 下游：[[02-流模型与微分同胚]]（ODE 视角的统一）

↔ 横联：[[轴线B/01-Bayes定理→贝叶斯神经网络]]（变分视角）

🔗 跨域：统计物理（朗之万动力学）、金融（期权定价 SDE）、化学（分子动力学）

━━━━━━━━━━━━━━
