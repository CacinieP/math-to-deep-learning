# RLHF 的博弈论视角

> RLHF 不只是"用人类反馈训练"。博弈论里，可借用多方目标冲突的视角讨论人类偏好与策略优化。过度最大化有缺陷的奖励模型可能导致 **reward hacking**（钻空子）。但固定奖励下，KL 正则的 RLHF 是**单智能体正则化优化**（目标函数自身即可视为平凡的势函数）；潜在博弈框架在此不提供额外收敛保证，PPO 的收敛性需另行分析。

**难度**：[前沿]（需要博弈论 + 强化学习 + KL 散度）

## 一、纯数学：博弈论基础

### 1.1 标准式博弈

博弈 $(\mathcal{A}, \mathcal{B}, u_A, u_B)$：玩家、策略集、效用函数。每个玩家选策略，效用由所有玩家的选择共同决定。

### 1.2 纳什均衡

策略组合 $(\pi_A^*, \pi_B^*)$ 是**纳什均衡**，若无人能单方面偏离获得更高效用：

$$u_A(\pi_A^*, \pi_B^*) \geq u_A(\pi_A, \pi_B^*) \quad \forall \pi_A$$

对玩家 B 也须满足相应的不等式。**意义**：纳什均衡是无人能靠单方面偏离改善效用的状态，不等于学习动态必然收敛。

> 📖 博弈论基础：纳什均衡、势博弈——Monderer & Shapley 1996

### 1.3 潜在博弈

博弈是**潜在博弈**，若存在潜在函数 $\Phi$ 使任意玩家偏离时，自己效用的变化 = $\Phi$ 的变化：

$$u_i(\pi_i', \pi_{-i}) - u_i(\pi_i, \pi_{-i}) = \Phi(\pi_i', \pi_{-i}) - \Phi(\pi_i, \pi_{-i})$$

**性质**：有限势博弈的严格单方改进路径会终止于纯策略纳什均衡；连续策略、任意步长的梯度更新或 PPO 不自动满足这一条件。势函数只能给出特定更新规则下的结论，不能作为标准 RLHF 的通用收敛保证。

---

## 二、RLHF 的博弈结构

### 2.1 奖励模型与策略的角色

- **奖励模型 RM**：代表人类偏好，给出标量奖励 $r(x, y)$
- **策略 $\pi$**：语言模型，生成回答 $y$，目标 $\max \mathbb{E}[r(x, y)]$

RM 由 **Bradley-Terry 模型**训练：$p(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$——它是奖励模型训练与 DPO 推导的起点。

### 2.2 不加约束的灾难

若策略纯最大化奖励：

$$\max_\pi \mathbb{E}_{x, y\sim\pi}[r(x, y)]$$

**reward hacking**：策略会找到 RM 的盲点——如重复"好评"词、生成长篇废话刷分。**RM 是有缺陷的近似**，纯最大化会放大缺陷。

### 2.3 KL 约束的博弈改造

加入 KL 惩罚：

$$\max_\pi\mathbb E_{x\sim\mathcal D}\left[\mathbb E_{y\sim\pi(\cdot\mid x)}r(x,y)-\beta D_{\mathrm{KL}}(\pi(\cdot\mid x)\|\pi_{\mathrm{ref}}(\cdot\mid x))\right]$$

**优化解读**（奖励模型固定时只有策略在优化，并非双玩家同时行动）：
- 第一项：策略想最大化 RM 奖励
- 第二项：策略不能离参考模型 $\pi_{\text{ref}}$ 太远（保持语言能力）

**固定奖励下，KL 正则的 RLHF 是单智能体正则化优化**（目标函数自身即可视为平凡的势函数）；潜在博弈框架在此不提供额外收敛保证，PPO 的收敛性需另行分析。

---

## 三、RLHF 的收敛性

### 3.1 KL 目标的收敛性

在有限表格式策略的概率单纯形上，KL 正则可使目标严格凹，从而有唯一最优分布；这不等于用任意优化算法都会收敛

$$\pi^* = \arg\max_\pi \mathbb{E}[r] - \beta\,D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

以上需 $\beta>0$、有限动作集、有限奖励且参考策略在可选动作上为正。对**参数化语言模型与 PPO 无此收敛定理保证**。

### 3.2 闭式解

$$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp(r(x, y)/\beta)$$

**这是 DPO 的理论基础**——直接偏好优化利用此闭式关系，绕过 RL。

> 📖 DPO 与互信息：[03-互信息→RLHF](../PART-03-专题映射/轴线E-信息论/03-互信息→RLHF.md)

### 3.3 β 的权衡

- $\beta \to \infty$：策略 = 参考模型（不学偏好，但仅保持参考行为，不保证安全）
- $\beta \to 0^+$：更接近纯奖励最大化，增加过度优化奖励模型的风险；并非必然发生 reward hacking

**实践**：$\beta$ 的量级取决于奖励尺度、序列长度以及 KL 按 token 还是序列归一化，不能给出脱离这些约定的通用范围。应结合奖励、KL 与独立偏好评估调参。

---

## 四、多智能体与对齐难题

### 4.1 多目标博弈

真实对齐不止"好/坏"二元。多个偏好维度（有用、无害、诚实）可构成**多目标优化**；固定权重求加权和仍是单策略优化：

$$\max_\pi \sum_k w_k \mathbb{E}[r_k(x,y)] - \beta\,D_{\text{KL}}(\pi\|\pi_{\text{ref}})$$

不同目标可能冲突（"无害" vs "有用"），权重 $w_k$ 调节权衡。

### 4.2 对手的攻防

红队测试、对抗性提示把 RLHF 看成**攻防博弈**：
- 攻击者：找提示绕过对齐（越狱）
- 防御者：RM 标注对抗样本，重训

只有指定双方策略集、信息与收益后，才能分析均衡存在性及学习动态；红队流程本身不足以推出均衡不存在或不稳定。

### 4.3 自我对弈（Constitutional AI）

Constitutional AI 使用原则指导模型批评、修订及生成偏好数据，属于 AI 反馈训练；这不等于 AlphaGo 式自我对弈，也不自动形成博弈论均衡。

---

## 五、工程实现层

### 5.1 PPO 裁剪代理目标（序列级教学接口）

下面的包装器约定 `logp` 返回每条完整回答的对数概率之和。省略了价值网络、GAE、token 掩码和优化器，只演示一次固定 rollout 的裁剪目标；完整 PPO 还需要这些组件。`old_logp`、参考策略与奖励模型都不参与梯度，KL 惩罚逐样本加入 advantage。

```python
import torch
import torch.nn.functional as F

@torch.no_grad()
def collect_rollout(policy, ref_policy, rm, prompts, beta=0.1):
    responses, old_logp = policy.generate(prompts, return_logp=True)
    ref_logp = ref_policy.logp(prompts, responses)
    rewards = rm(prompts, responses)
    advantages = rewards - beta * (old_logp - ref_logp)  # 零 baseline 的简化
    return responses, old_logp.detach(), advantages.detach()

def ppo_step(policy, ref_policy, rm, prompts, beta=0.1, clip=0.2, rollout=None):
    # 多轮更新时，显式传入同一个 collect_rollout(...) 返回值。
    if rollout is None:
        rollout = collect_rollout(policy, ref_policy, rm, prompts, beta)
    responses, old_logp, advantages = rollout
    new_logp = policy.logp(prompts, responses)
    ratio = (new_logp - old_logp.detach()).exp()
    advantages = advantages.detach()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1 - clip, 1 + clip) * advantages,
    )
    return -surrogate.mean()

```

多轮更新先调用一次 `rollout = collect_rollout(policy, ref_policy, rm, prompts)`，随后各次 `ppo_step(..., rollout=rollout)` 复用它。只有第一次更新前 ratio 为 1；参数改变后比例才偏离 1，裁剪才起作用。不能在每轮更新前重新采样并重定义旧策略。

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
│  博弈论: 纳什均衡、势博弈及其适用条件                       │
│  KL 正则: 固定奖励下的单策略优化                               │
│  闭式解: π* ∝ π_ref·exp(r/β) ⇒ DPO 理论基础               │
│  [[Mathematics-Universe/06-超纲拓展/...]]                  │
└──────────────────────────┬───────────────────────────────┘
                           │ 偏好博弈 + KL 约束
                           ▼
┌──────────────────────────────────────────────────────────┐
│  RLHF 的博弈结构                                            │
│  RM(人类代理) vs 策略(语言模型)                             │
│  过度优化可能 reward hacking；KL 限制策略漂移                │
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
标准 RLHF: 固定奖励模型下优化策略
无约束: 策略钻 RM 空子(reward hacking)
加 KL: 惩罚策略偏离参考；不保证 PPO 收敛或安全
闭式解 π* ∝ π_ref·exp(r/β) ⇒ DPO 绕过 RL
β 调节: 大=更接近参考策略，小=更重视奖励；均不保证安全
```

**固定奖励模型的 RLHF 是带 KL 正则的策略优化。** 博弈论适用于明确定义多方策略与更新的扩展场景，不能替代 PPO/DPO 的具体分析。

---

## 八、延伸阅读

### 论文
- **Christiano et al. (2017)** "Deep RL from Human Preferences"——RLHF 起源
- **Ouyang et al. (2022)** "Training language models to follow instructions"（InstructGPT）
- [Rafailov et al. (2023), Direct Preference Optimization](https://arxiv.org/abs/2305.18290)（DPO）
- **Monderer & Shapley (1996)** "Potential Games"——潜在博弈理论

### 关联文章
- [03-互信息→RLHF](../PART-03-专题映射/轴线E-信息论/03-互信息→RLHF.md)（信息论视角的 RLHF）
- [02-KL散度→信息瓶颈](../PART-03-专题映射/轴线E-信息论/02-KL散度→信息瓶颈.md)（KL 约束的信息论意义）

---

## 联系网络

⬆ 上游：（博弈论基础：纳什均衡、势博弈——Monderer & Shapley 1996），[03-互信息→RLHF](../PART-03-专题映射/轴线E-信息论/03-互信息→RLHF.md)（信息论视角）

⬇ 下游：大模型对齐、安全训练

↔ 横联：[03-互信息→RLHF](../PART-03-专题映射/轴线E-信息论/03-互信息→RLHF.md)（互信息与偏好建模）

🔗 跨域：经济学（机制设计）、演化生物学（进化博弈）、控制论（约束优化）

━━━━━━━━━━━━━━
