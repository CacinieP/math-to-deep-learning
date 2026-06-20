# Lagrange 乘数 → 约束优化

> 从"在曲面上找极值"到"GAN中判别器的作用"再到"强化学习的奖励 shaping"，Lagrange 乘数把"约束"统一成了一种视角——几乎所有深度学习中的约束问题都可以用这套语言描述。

**难度**：[进阶]（需要多元微分 + 了解 GAN/RL 基本概念）

## 一、纯数学版

### 1.1 问题定义

**无约束优化**（你已经很熟悉了）：
$$\min_{\mathbf{x}} f(\mathbf{x})$$

解法：令 $\nabla f(\mathbf{x}) = 0$。

**约束优化**（更现实）：
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g(\mathbf{x}) = 0$$

其中 s.t. = subject to（受限于）。

**几何直觉**：不是在整个空间找最低点，而是在**曲面** $g(\mathbf{x}) = 0$ 上找最低点。

```
无约束:                            约束:
                                                                
        ╭──╮                          ╭──╮
      ╭─┤  ├─╮   在整个空间       ╭──┤  ├──╮ 在曲面上找
      │ │  │ │   找最低点         │  │   │  │ 最低点
    ╭─┤─┤  ├─┤─╮              ╭──┤──┤───├──┤──╮
    │ │ │  │ │ │ │              │  │  │   │  │  │
      ╰─┤  ├─╯                  ╰──┤──┤───├──┤──╯
        ╰──╯                        ╰──╯
        ▽ 全局最小                    ▽ 约束下的最小
        (可能不在约束上)              (在约束与等高线相切处)
```

> 📖 Lagrange 乘数法的完整推导：[[Mathematics-Universe/03-高等数学/04-多元微分学/多元微分学详解.md#45-条件极值与Lagrange乘数法]]

### 1.2 Lagrange 乘数法

**构造 Lagrange 函数**：
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

其中 $\lambda$ 是 **Lagrange 乘数**。

**求解**：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0$$

展开：
$$\nabla f(\mathbf{x}^*) = \lambda \nabla g(\mathbf{x}^*)$$

**几何含义**：在约束的极值点处，目标函数的梯度**平行于**约束的梯度。

$$\nabla f \parallel \nabla g \quad \Leftrightarrow \quad \text{等高线与约束曲面相切}$$

### 1.3 不等式约束：KKT 条件

对于更一般的约束优化：
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$$

**KKT（Karush-Kuhn-Tucker）条件**：
1. **可行性**：$g_i(\mathbf{x}^*) \leq 0$, $h_j(\mathbf{x}^*) = 0$
2. **Lagrange 函数梯度为 0**：$\nabla f + \sum \lambda_i \nabla g_i + \sum \mu_j \nabla h_j = 0$
3. **对偶可行性**：$\lambda_i \geq 0$
4. **互补松弛**：$\lambda_i g_i(\mathbf{x}^*) = 0$

**互补松弛的直觉**：如果约束 $g_i$ 是**松弛的**（$g_i < 0$，未触及边界），则对应的 $\lambda_i = 0$（该约束不起作用）。只有**起作用的约束**（$g_i = 0$，在边界上）才有 $\lambda_i > 0$。

---

## 二、第一次应用：GAN 中的判别器

### 2.1 GAN 的极小极大博弈

GAN（Generative Adversarial Network）的目标：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

这是一个**博弈论**的极小极大问题——$G$ 想最小化 $V$，$D$ 想最大化 $V$。

**从 Lagrange 乘数的视角看**：

生成器的训练目标是让判别器输出 $D(G(z)) \to 1$（骗过判别器）。这等价于：

$$\min_G \left[ -\mathbb{E}_z[\log D(G(z))] \right]$$

而判别器的目标是：
$$\max_D \left[ \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))] \right]$$

### 2.2 最优判别器

固定 $G$，求使 $V$ 最大的 $D^*$：

$$\frac{\partial V}{\partial D} = \frac{\mathbb{E}_x}{D(x)} - \frac{\mathbb{E}_z}{1 - D(G(z))} = 0$$

解得：
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$$

这是**最优判别器的闭式解**。

### 2.3 最优生成器

代入 $D^*$ 到 $V$，得到仅关于 $G$ 的目标：

$$V(G, D^*) = \mathbb{E}_x\left[\log\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}\right] + \mathbb{E}_z\left[\log\frac{p_G(x)}{p_{\text{data}}(x) + p_G(x)}\right]$$

这等价于：
$$V(G, D^*) = -2\log 2 + D_{JS}(p_{\text{data}} \| p_G)$$

其中 $D_{JS}$ 是 **Jensen-Shannon 散度**。

**全局最优**：当 $p_G = p_{\text{data}}$ 时，$D_{JS} = 0$，$V = -2\log 2$。

**这就是 GAN 的理论保证**——如果两个网络都有足够的容量且训练充分，最终生成分布会收敛到真实数据分布。

```python
# 简化版 GAN 训练（展示 Lagrange 视角）
class GAN:
    def __init__(self, generator, discriminator):
        self.G = generator
        self.D = discriminator

    def train_step(self, real_data, noise):
        # D 的训练：最大化 V
        # 等价于最小化 -V
        d_optim.zero_grad()
        d_real = self.D(real_data)
        d_fake = self.D(self.G(noise).detach())  # 不更新 G
        d_loss = -(
            torch.log(d_real + 1e-8).mean() +      # E[log D(x)]
            torch.log(1 - d_fake + 1e-8).mean()    # E[log(1-D(G(z)))]
        )
        d_loss.backward()
        d_optim.step()

        # G 的训练：最小化 V
        # 等价于最大化 E[log D(G(z))]（或最小化 -log D(G(z))）
        g_optim.zero_grad()
        d_fake_for_g = self.D(self.G(noise))
        g_loss = -torch.log(d_fake_for_g + 1e-8).mean()  # -E[log D(G(z))]
        g_loss.backward()
        g_optim.step()
```

---

## 三、第二次应用：强化学习中的约束

### 3.1 带约束的 RL 问题

标准的 RL 目标是最大化期望回报：
$$\max_\pi \mathbb{E}_{\tau \sim \pi}\left[\sum_t r(s_t, a_t)\right]$$

但实际中常有约束：
- **安全约束**：不能进入危险区域（如自动驾驶不能撞墙）
- **资源约束**：计算资源不能超过 $C$
- **公平性约束**：不同群体的准确率差距不超过阈值

$$\max_\pi \mathbb{E}[R(\pi)] \quad \text{s.t.} \quad \mathbb{E}[C_i(\pi)] \leq b_i$$

### 3.2 Lagrange 方法在 RL 中的应用

构造 Lagrange 函数：
$$\mathcal{L}(\pi, \lambda) = \mathbb{E}[R(\pi)] + \sum_i \lambda_i(b_i - \mathbb{E}[C_i(\pi)])$$

**内层优化策略 $\pi$**，**外层调整 Lagrange 乘数 $\lambda$**：

```
外循环: λ ← λ + η(b - E[C(π)])   （更新乘数，惩罚超约束）
    ↓
内循环: π ← argmax E[R(π)] - Σ λᵢE[Cᵢ(π)]   （带惩罚的RL优化）
    ↓
        λ 是"价格"——约束的违反程度乘以价格
        π 在"奖励 - 价格×代价"的框架下学习
```

这就是 **Constrained Policy Optimization (CPO)** 和 **Lagrange 方法**的核心。

```python
# 概念：用 Lagrange 乘数处理 RL 约束
class LagrangianRL:
    def __init__(self, n_constraints):
        self.lambda_ = torch.zeros(n_constraints)  # Lagrange 乘数
        self.lambda_lr = 0.01

    def lagrangian_loss(self, returns, costs):
        """
        returns: (batch,) 奖励
        costs: (batch, n_constraints) 各约束代价
        lambda_: (n_constraints,) Lagrange 乘数
        """
        # 惩罚项: λ · cost
        penalty = (self.lambda_ * costs).sum(dim=-1)
        # 总目标: 奖励 - 惩罚
        loss = -(returns - penalty).mean()
        return loss

    def update_lambda(self, costs, bounds):
        """
        λ ← max(0, λ + η(cost - bound))
        互补松弛: λ > 0 ⟺ 约束被违反
        """
        violation = costs.mean(dim=0) - bounds  # 超了多少
        self.lambda_ = torch.clamp(
            self.lambda_ + self.lambda_lr * violation,
            min=0.0  # λ ≥ 0
        )
```

---

## 四、第三次应用：对比学习中的 InfoNCE 损失

### 4.1 互信息最大化 = 约束优化

对比学习的目标是让正样本对的表示接近，负样本对的表示远离。

**InfoNCE 损失**：
$$\mathcal{L} = -\mathbb{E}\left[\log\frac{e^{f(x_i)^T f(x_j)/\tau}}{\sum_{k=1}^N e^{f(x_i)^T f(x_k)/\tau}}\right]$$

这等价于最小化：
$$\mathcal{L} = \mathbb{E}[\log N] - \mathbb{E}[\log e^{f(x_i)^T f(x_j)/\tau}]$$

从信息论角度：
$$\mathcal{L} = \log N - I(x; z)$$

其中 $I(x; z)$ 是输入和表示之间的互信息。

**最大化互信息 $I(x; z)$ 等价于最小化 $\mathcal{L}$**。

**从 Lagrange 视角**：
- 我们想最大化 $I(x; z)$
- 但 $I(x; z)$ 无法直接计算
- InfoNCE 是互信息的**下界估计**（NCE = Noise Contrastive Estimation）
- 最大化下界 = 在 Lagrange 框架下的凸近似

### 4.2 对比损失中的"约束"

对比学习的负样本对实际上是一个**对偶约束**：

```
正样本对:  (x_i, x_j)  →  鼓励 f(x_i)^T f(x_j) 大
                               ↓
                        f(x_i) · f(x_j) ≥ C  （某个阈值）

负样本对:  (x_i, x_k)  →  鼓励 f(x_i)^T f(x_k) 小
                               ↓
                        f(x_i) · f(x_k) ≤ c  （另一个阈值）
```

用 Lagrange 乘数表示：
$$\mathcal{L} = \mathbb{E}[\max(0, C - f(x_i)^T f(x_j))] + \mathbb{E}[\max(0, f(x_i)^T f(x_k) - c)]$$

这就是 **Triplet Loss** 的形式！

```python
def triplet_loss(anchor, positive, negative, margin=0.2):
    """
    Triplet Loss = 约束优化的特例
    约束: d(anchor, positive) ≤ d(anchor, negative) - margin
    违反约束时惩罚: max(0, d(ap) - d(an) + margin)
    """
    d_ap = (anchor - positive).pow(2).sum(dim=-1)  # 正样本距离
    d_an = (anchor - negative).pow(2).sum(dim=-1)  # 负样本距离

    # Lagrange 乘数视角: 违反约束的惩罚
    loss = torch.clamp(d_ap - d_an + margin, min=0.0)
    return loss.mean()
```

---

## 五、完整映射图

```
┌─────────────────────────────────────────────────────────────────┐
│  纯数学                                                          │
│  Lagrange 函数: L(x,λ) = f(x) - λ·g(x)                         │
│  KKT 条件: 可行性 + 梯度=0 + 对偶可行 + 互补松弛                  │
│  [[Mathematics-Universe/03-高等数学/04-多元微分学/...]]          │
└────────────────────────────┬────────────────────────────────────┘
                             │ 推广
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  约束优化通用框架                                                  │
│  min_θ L(θ) s.t. C_i(θ) ≤ b_i                                   │
│  L(θ,λ) = L_task(θ) + Σ λᵢ(bᵢ - Cᵢ(θ))                        │
│  交替优化: θ（梯度下降）+ λ（乘数更新）                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ GAN      │  │ RL 约束  │  │ 对比学习  │
        │ D = arg  │  │ 安全RL   │  │ Triplet  │
        │ max_D V  │  │ CPO      │  │ Margin   │
        └──────────┘  └──────────┘  └──────────┘
```

---

## 六、直觉总结

**Lagrange 乘数的本质**：当你不能直接优化目标函数（因为有约束），就把"约束违反量"当作一个**惩罚项**加进目标函数，然后用一个**乘数**控制惩罚的力度。

```
无约束:     min f(x)
                ↓ 加约束
约束优化:   min f(x) s.t. g(x) = 0
                ↓ Lagrange 变换
Lagrange:   min f(x) - λ·g(x)
                ↓ 交替优化
交替:       x ← 最小化 L(x, λ)
            λ ← 最大化 L(x, λ)  （增大对约束违反的惩罚）
```

**在深度学习中的应用模式**：
1. **GAN**：判别器 = Lagrange 乘数（它衡量生成分布和真实分布的"差距"）
2. **约束 RL**：$\lambda$ = 约束违反的"价格"，自动调整
3. **对比学习**：Triplet Loss = 距离约束的 Lagrange 惩罚
4. **注意力**：Mask = 硬约束（不允许 attending 到某些位置）

**一句话总结**：Lagrange 乘数把"硬约束"（不允许违反）转化为"软约束"（违反就罚钱），乘数 $\lambda$ 是罚金的自动调节旋钮——罚轻了约束不起作用，罚重了主目标被牺牲，调到恰到好处就是最优解。

---

## 七、延伸阅读

### 论文
- **GAN 理论**：Goodfellow et al. (2014) — 原始 GAN 论文的定理 1（全局最优）
- **Constrained RL**：Achiam et al. (2017) "Constrained Policy Optimization"
- **对比学习**：Chen et al. (2020) "A Simple Framework for Contrastive Learning"
- **Lagrangian RL 综述**：Ray et al. (2022) "Reinforcement Learning with Constraints"

### 课程
- [CS229: Constrained Optimization](https://cs229.stanford.edu/) — Lagrange / KKT 条件
- [Stanford EE178: Lagrange Duality](https://web.stanford.edu/class/ee178/) — 对偶性的严格推导

### 关联文章
- [[梯度 → 反向传播]]（轴线C上一篇）
- [[Taylor展开 → 二阶优化]]（轴线C第二篇）
- [[PART-05/RLHF 的博弈论视角]]

---

## 联系网络

⬆ 上游: [[Mathematics-Universe/03-高等数学/04-多元微分学/多元微分学详解.md]]（Lagrange 乘数法），[[Mathematics-Universe/08-数学联系网络/跨分支深层联系.md]]（对偶思想）

⬇ 下游: [[PART-05/RLHF的博弈论视角]]

↔ 横联: [[02-Taylor展开→二阶优化]]（二阶条件），[[01-梯度→反向传播]]（梯度是约束优化的工具）

🔗 跨域: 博弈论（极小极大 = Lagrange 对偶），控制理论（约束最优控制），经济学（效用最大化·预算约束）

━━━━━━━━━━━━━━
