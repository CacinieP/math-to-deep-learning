# Lagrange 乘数 → 约束优化

> 从"在曲面上找极值"到"GAN中判别器的作用"再到"强化学习的奖励 shaping"，Lagrange 乘数把"约束"统一成了一种视角——几乎所有深度学习中的约束问题都可以用这套语言描述。

**难度**：[进阶]（需要多元微分 + 了解 GAN/RL 基本概念）

## 一、纯数学版

### 1.1 问题定义

**无约束优化**（你已经很熟悉了）：
$$\min_{\mathbf{x}} f(\mathbf{x})$$

内部可微极小点须满足 $\nabla f(\mathbf{x})=0$，但这只是必要条件，还需判断极值、边界和存在性。

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

> 📖 Lagrange 乘数法的完整推导：[多元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/04-多元微分学/多元微分学详解.md#32-条件极值lagrange乘数法)

### 1.2 Lagrange 乘数法

**构造 Lagrange 函数**（全文统一采用 $\mathcal{L} = f + \lambda g$ 的符号约定）：
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

其中 $\lambda$ 是 **Lagrange 乘数**。

**求解**：
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = 0, \quad \frac{\partial \mathcal{L}}{\partial \lambda} = 0$$

展开：
$$\nabla f(\mathbf{x}^*) = -\lambda \nabla g(\mathbf{x}^*)$$

**前提**：等式约束需满足正则性（单约束常用 $\nabla g
e0$）。**几何含义**：在这样的约束极值点处，目标函数的梯度**平行于**约束的梯度。

$$\nabla f \parallel \nabla g \quad \Leftrightarrow \quad \text{等高线与约束曲面相切}$$

### 1.3 不等式约束：KKT 条件

对于更一般的约束优化：
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0$$

**KKT（Karush-Kuhn-Tucker）条件**：满足约束资格时是局部最优的必要条件；凸目标、凸不等式约束与仿射等式约束下也是充分条件。
1. **可行性**：$g_i(\mathbf{x}^*) \leq 0$, $h_j(\mathbf{x}^*) = 0$
2. **Lagrange 函数梯度为 0**：$\nabla f + \sum \lambda_i \nabla g_i + \sum \mu_j \nabla h_j = 0$
3. **对偶可行性**：$\lambda_i \geq 0$
4. **互补松弛**：$\lambda_i g_i(\mathbf{x}^*) = 0$

**互补松弛的直觉**：如果约束 $g_i$ 是**松弛的**（$g_i < 0$，未触及边界），则对应的 $\lambda_i = 0$（该约束不起作用）。只有在边界上的约束才**可能**有 $\lambda_i>0$，但活跃约束也可以对应零乘数。

---

## 二、第一次应用：GAN 中的判别器

### 2.1 GAN 的极小极大博弈

GAN（Generative Adversarial Network）的目标：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

这是一个**博弈论**的极小极大问题——$G$ 想最小化 $V$，$D$ 想最大化 $V$。

**这是极小极大问题，与 Lagrange 鞍点问题形式相似，但判别器不是 Lagrange 乘数**：

生成器的训练目标是让判别器输出 $D(G(z)) \to 1$（骗过判别器）。常用的非饱和生成器替代目标是：

$$\min_G \left[ -\mathbb{E}_z[\log D(G(z))] \right]$$

而判别器的目标是：
$$\max_D \left[ \mathbb{E}_x[\log D(x)] + \mathbb{E}_z[\log(1 - D(G(z)))] \right]$$

### 2.2 最优判别器

固定 $G$，求使 $V$ 最大的 $D^*$：

$$\frac{\partial V}{\partial D} = \frac{p_{\text{data}}(x)}{D(x)} - \frac{p_G(x)}{1 - D(x)} = 0$$

解得：
$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}$$

这是**最优判别器的闭式解**。

### 2.3 最优生成器

代入 $D^*$ 到 $V$，得到仅关于 $G$ 的目标：

$$V(G, D^*) = \mathbb{E}_x\left[\log\frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}\right] + \mathbb{E}_z\left[\log\frac{p_G(x)}{p_{\text{data}}(x) + p_G(x)}\right]$$

常用的非饱和生成器替代目标是：
$$V(G, D^*) = -2\log 2 + 2\, D_{JS}(p_{\text{data}} \| p_G)$$

其中 $D_{JS}$ 是 **Jensen-Shannon 散度**（按 $\frac{1}{2}D_{KL}(p\|m) + \frac{1}{2}D_{KL}(q\|m)$、$m=\frac{p+q}{2}$ 的 ½ 加权标准定义，系数 2 由此而来）。

**全局最优**：当 $p_G = p_{\text{data}}$ 时，$D_{JS} = 0$，$V = -2\log 2$。

这个结论刻画理想目标的全局最优；不保证有限神经网络交替梯度训练收敛，实际还可能振荡或模式坍缩。

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
        # 使用非饱和替代损失 -log D(G(z))，不等于原 minimax 梯度
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

将目标改写为最小化 $-\mathbb{E}[R(\pi)]$，约束写成 $g_i(\pi) = \mathbb{E}[C_i(\pi)] - b_i \leq 0$（与上文 $\mathcal{L} = f + \lambda g$ 约定一致），构造 Lagrange 函数：

$$\mathcal{L}(\pi, \lambda) = -\mathbb{E}[R(\pi)] + \sum_i \lambda_i \left(\mathbb{E}[C_i(\pi)] - b_i\right)$$

**内层优化策略 $\pi$**，**外层调整 Lagrange 乘数 $\lambda$**：

```
外循环: λ ← λ + η(E[C(π)] − b)，并截断 λ ≥ 0   （约束被违反时 λ 增大）
    ↓
内循环: π ← argmax E[R(π)] − Σ λᵢE[Cᵢ(π)]   （带惩罚的RL优化，等价于最小化上式）
    ↓
        λ 是"价格"——约束的违反程度乘以价格
        π 在"奖励 - 价格×代价"的框架下学习
```

这是约束 RL 的 Lagrange 方法；CPO 采用信赖域约束更新，是相关但不同的算法。

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
        训练中违反约束会增大 λ；最优点 λ > 0 意味着约束活跃，不是仍被违反
        """
        violation = costs.mean(dim=0) - bounds  # 超了多少
        self.lambda_ = torch.clamp(
            self.lambda_ + self.lambda_lr * violation,
            min=0.0  # λ ≥ 0
        )
```

---

## 四、第三次应用：对比学习中的 InfoNCE 损失

### 4.1 InfoNCE 是互信息下界代理

设一个正样本来自联合分布，另 $N-1$ 个负样本独立来自边缘分布，总候选数为 $N$，则

$$\mathcal L_{\rm NCE}=-\mathbb E\log\frac{e^{s(x,z^+)}}{\sum_{j=1}^{N}e^{s(x,z_j)}},\qquad I(X;Z)\geq\log N-\mathcal L_{\rm NCE}.$$

分母的 log-sum-exp 不能直接替换为 $\log N$；即便打分函数最优，有限负样本下也不一般得到等号。最小化 InfoNCE 最大化这个代理下界，不等于直接最大化真实互信息，也不是 Lagrange 对偶推导。

### 4.2 Triplet Loss 是间隔惩罚

约束 $d(a,p)+m\leq d(a,n)$ 可用 hinge 惩罚

$$\max\{0,d(a,p)-d(a,n)+m\}$$

构成三元组损失。它把违反排序间隔的程度加入目标；固定惩罚系数不保证硬约束最终满足，也不等价于上面的 InfoNCE。

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
│  Lagrange 函数: L(x,λ) = f(x) + λ·g(x)                         │
│  KKT 条件: 可行性 + 梯度=0 + 对偶可行 + 互补松弛                  │
│  [[Mathematics-Universe/03-高等数学/04-多元微分学/...]]          │
└────────────────────────────┬────────────────────────────────────┘
                             │ 推广
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  约束优化通用框架                                                  │
│  min_θ L(θ) s.t. C_i(θ) ≤ b_i                                   │
│  L(θ,λ) = L_task(θ) + Σ λᵢ(Cᵢ(θ) − bᵢ)                        │
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
Lagrange:   min f(x) + λ·g(x)
                ↓ 交替优化
交替:       x ← 最小化 L(x, λ)
            λ ← 最大化 L(x, λ)  （增大对约束违反的惩罚）
```

**在深度学习中的应用模式**：
1. **GAN**：具有极小极大结构；判别器一般不等于 Lagrange 乘数
2. **约束 RL**：$\lambda$ = 约束违反的"价格"，自动调整
3. **对比学习**：Triplet Loss 是排序间隔的 hinge 惩罚，不是直接求解 Lagrange 对偶
4. **注意力**：Mask = 硬约束（不允许 attending 到某些位置）

**一句话总结**：Lagrange 乘数把"硬约束"（不允许违反）转化为"软约束"（违反就罚钱），乘数 $\lambda$ 是罚金的自动调节旋钮——罚轻了约束不起作用，罚重了主目标被牺牲，调到恰到好处就是最优解。

---

## 七、延伸阅读

### 论文
- **GAN 理论**：Goodfellow et al. (2014) — 原始 GAN 论文的定理 1（全局最优）
- **Constrained RL**：Achiam et al. (2017) "Constrained Policy Optimization"
- **对比学习**：Chen et al. (2020) "A Simple Framework for Contrastive Learning"
- **安全 RL 综述**：García & Fernández (2015) "A Comprehensive Survey on Safe Reinforcement Learning"

### 课程
- [CS229: Constrained Optimization](https://cs229.stanford.edu/) — Lagrange / KKT 条件
- [Stanford EE178: Lagrange Duality](https://web.stanford.edu/class/ee178/) — 对偶性的严格推导

### 关联文章
- [梯度 → 反向传播](01-梯度→反向传播.md)（轴线C上一篇）
- [Taylor展开 → 二阶优化](02-Taylor展开→二阶优化.md)（轴线C第二篇）
- [RLHF 的博弈论视角](../../PART-05-前沿中的数学/05-RLHF的博弈论视角.md)

---

## 联系网络

⬆ 上游: [多元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/04-多元微分学/多元微分学详解.md)（Lagrange 乘数法），[跨分支深层联系](https://github.com/CacinieP/Mathematics-Universe/blob/main/08-数学联系网络/跨分支深层联系.md)（对偶思想）

⬇ 下游: [RLHF的博弈论视角](../../PART-05-前沿中的数学/05-RLHF的博弈论视角.md)

↔ 横联: [02-Taylor展开→二阶优化](02-Taylor展开→二阶优化.md)（二阶条件），[01-梯度→反向传播](01-梯度→反向传播.md)（梯度是约束优化的工具）

🔗 跨域: 博弈论（极小极大 = Lagrange 对偶），控制理论（约束最优控制），经济学（效用最大化·预算约束）

━━━━━━━━━━━━━━
