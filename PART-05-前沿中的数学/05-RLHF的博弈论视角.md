# RLHF 的博弈论视角

> RLHF 不只是"用人类反馈训练"。博弈论里，它是一个**双人博弈**：人类（或奖励模型）设定目标，策略（语言模型）最大化奖励。问题在于——如果纯最大化奖励，策略会**reward hacking**（钻空子）。KL 约束把博弈变成"潜在博弈"，确保收敛到**纳什均衡**而非退化策略。

**难度**：[前沿]（需要博弈论 + 强化学习 + KL 散度）

## 一、纯数学：博弈论基础

### 1.1 标准式博弈

博弈 $(\mathcal{A}, \mathcal{B}, u_A, u_B)$：玩家、策略集、效用函数。每个玩家选策略，效用由所有玩家的选择共同决定。

### 1.2 纳什均衡

策略组合 $(\pi_A^*, \pi_B^*)$ 是**纳什均衡**，若无人能单方面偏离获得更高效用：

$$u_A(\pi_A^*, \pi_B^*) \geq u_A(\pi_A, \pi_B^*) \quad \forall \pi_A$$

**意义**：纳什均衡是"谁都不想先变"的稳定状态。

> 📖 博弈论基础：[[Mathematics-Universe/06-超纲拓展/数值分析.md]]

### 1.3 潜在博弈

博弈是**潜在博弈**，若存在潜在函数 $\Phi$ 使任意玩家偏离时，自己效用的变化 = $\Phi$ 的变化：

$$u_i(\pi_i', \pi_{-i}) - u_i(\pi_i, \pi_{-i}) = \Phi(\pi_i', \pi_{-i}) - \Phi(\pi_i, \pi_{-i})$$

**性质**：潜在博弈保证**梯度/更新收敛到纳什均衡**（最大化 $\Phi$）。这是 RLHF 训练稳定性的理论根基。

---

## 二、RLHF 的博弈结构

### 2.1 两个玩家

- **奖励模型 RM**：代表人类偏好，给出标量奖励 $r(x, y)$
- **策略 $\pi$**：语言模型，生成回答 $y$，目标 $\max \mathbb{E}[r(x, y)]$

### 2.2 不加约束的灾难

若策略纯最大化奖励：

$$\max_\pi \mathbb{E}_{x, y\sim\pi}[r(x, y)]$$

**reward hacking**：策略会找到 RM 的盲点——如重复"好评"词、生成长篇废话刷分。**RM 是有缺陷的近似**，纯最大化会放大缺陷。

### 2.3 KL 约束的博弈改造

加入 KL 惩罚：

$$\max_\pi \mathbb{E}_{x, y\sim\pi}[r(x, y)] - \beta\,D_{\text{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

**博弈论解读**：
- 第一项：策略想最大化 RM 奖励
- 第二项：策略不能离参考模型 $\pi_{\text{ref}}$ 太远（保持语言能力）

**这构成一个潜在博弈**——目标函数单调，梯度更新收敛。

---

## 三、RLHF 的收敛性

### 3.1 潜在博弈保证收敛

带 KL 约束的 RLHF 目标是**强凹**的（KL 项强凸），策略梯度/PPO 收敛到唯一最优解：

$$\pi^* = \arg\max_\pi \mathbb{E}[r] - \beta\,D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

### 3.2 闭式解

$$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x, y)/\beta)$$

**这是 DPO 的理论基础**——直接偏好优化利用此闭式关系，绕过 RL。

> 📖 DPO 与互信息：[[轴线E/03-互信息→RLHF]]

### 3.3 β 的权衡

- $\beta \to \infty$：策略 = 参考模型（不学偏好，但安全）
- $\beta \to 0$：纯最大化奖励（学偏好，但 reward hacking）

**实践**：$\beta \in [0.01, 0.5]$，在偏好对齐与稳定性间平衡。

---

## 四、多智能体与对齐难题

### 4.1 多目标博弈

真实对齐不止"好/坏"二元。多个偏好维度（有用、无害、诚实）构成**多目标博弈**：

$$\max_\pi \sum_k w_k \mathbb{E}[r_k(x,y)] - \beta\,D_{\text{KL}}(\pi\|\pi_{\text{ref}})$$

不同目标可能冲突（"无害" vs "有用"），权重 $w_k$ 调节权衡。

### 4.2 对手的攻防

红队测试、对抗性提示把 RLHF 看成**攻防博弈**：
- 攻击者：找提示绕过对齐（越狱）
- 防御者：RM 标注对抗样本，重训

这是动态博弈，纳什均衡可能不存在或不稳定。

### 4.3 自我对弈（Constitutional AI）

Anthropic 的 Constitutional AI：模型**自己生成偏好数据**（按"宪法"自评）。这把 RLHF 变成**自我博弈**——类似 AlphaGo 的自我对弈，但有"宪法"作仲裁。

---

## 五、工程实现层

### 5.1 PPO 训练循环

```python
import torch
import torch.nn.functional as F

def ppo_step(policy, ref_policy, rm, prompts, beta=0.1, clip=0.2):
    # 1. 生成回答
    responses, old_logp = policy.generate(prompts, return_logp=True)
    # 2. 算奖励
    rewards = rm(prompts, responses)
    # 3. 算 KL 惩罚(蒙特卡洛估计)
    with torch.no_grad():
        ref_logp = ref_policy.logp(prompts, responses)
    kl = (old_logp - ref_logp).mean()
    penalized = rewards - beta * kl
    # 4. PPO 更新(带 clip 防止策略大跳)
    new_logp = policy.logp(prompts, responses)
    ratio = (new_logp - old_logp).exp()
    loss = -torch.min(ratio * penalized, 
                      ratio.clamp(1-clip, 1+clip) * penalized).mean()
    return loss
```

### 5.2 DPO 的简化

```python
def dpo_loss(policy, ref, x, y_win, y_lose, beta=0.1):
    logp_w = policy.logp(x, y_win)
    logp_l = policy.logp(x, y_lose)
    with torch.no_grad():
        logp_w_ref = ref.logp(x, y_win)
        logp_l_ref = ref.logp(x, y_lose)
    logits = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
    return -F.logsigmoid(logits).mean()
```

---

## 六、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  博弈论: 纳什均衡, 潜在博弈(收敛保证)                       │
│  KL 约束: 构成潜在博弈 ⇒ 收敛                               │
│  闭式解: π* ∝ π_ref·exp(r/β) ⇒ DPO 理论基础               │
│  [[Mathematics-Universe/06-超纲拓展/...]]                  │
└──────────────────────────┬───────────────────────────────┘
                           │ 偏好博弈 + KL 约束
                           ▼
┌──────────────────────────────────────────────────────────┐
│  RLHF 的博弈结构                                            │
│  RM(人类代理) vs 策略(语言模型)                             │
│  无约束 ⇒ reward hacking; KL ⇒ 潜在博弈收敛                │
│  DPO 用闭式解绕过 RL                                        │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  PPO: 生成→奖励→KL惩罚→clip 更新                           │
│  DPO: logsigmoid(β·logratio 差)                            │
└──────────────────────────────────────────────────────────┘
```

---

## 七、直觉总结

```
RLHF = 奖励模型 vs 策略 的博弈
无约束: 策略钻 RM 空子(reward hacking)
加 KL: 策略不能离参考模型太远 ⇒ 潜在博弈 ⇒ 收敛
闭式解 π* ∝ π_ref·exp(r/β) ⇒ DPO 绕过 RL
β 调节: 大=安全不学, 小=学但易 hacking
```

**RLHF 是带约束的博弈。** 数学是博弈论 + KL，工程是 PPO/DPO 的实现。

---

## 八、延伸阅读

### 论文
- **Christiano et al. (2017)** "Deep RL from Human Preferences"——RLHF 起源
- **Ouyang et al. (2022)** "Training language models to follow instructions"（InstructGPT）
- **Rafailov et al. (2023)** "Direct Preference Optimization"（DPO）
- **Monderer & Shapley (1996)** "Potential Games"——潜在博弈理论

### 关联文章
- [[轴线E/03-互信息→RLHF]]（信息论视角的 RLHF）
- [[轴线E/02-KL散度→信息瓶颈]]（KL 约束的信息论意义）

---

## 联系网络

⬆ 上游：[[Mathematics-Universe/06-超纲拓展/数值分析.md]]（博弈论基础），[[轴线E/03-互信息→RLHF]]（信息论视角）

⬇ 下游：大模型对齐、安全训练

↔ 横联：[[轴线E/03-互信息→RLHF]]（互信息与偏好建模）

🔗 跨域：经济学（机制设计）、演化生物学（进化博弈）、控制论（约束优化）

━━━━━━━━━━━━━━
