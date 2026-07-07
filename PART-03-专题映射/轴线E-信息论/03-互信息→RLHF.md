# 互信息 → RLHF 与表示学习

> 互信息 $I(X;Y)$ 度量"两个变量共享多少信息"。深度学习里有两条主线把它推到极致：一是**表示学习**——最大化表示与标签的互信息（对比学习的 InfoNCE、SimCLR 的实质）；二是 **RLHF**——人类反馈强化学习里，奖励模型在捕捉"人类偏好"这一隐变量的信息，策略优化本质是在偏好分布与策略分布间做 KL 约束下的信息逼近。

**难度**：[前沿]（需要互信息 + 强化学习 + 对比学习直觉）

## 一、纯数学版：互信息的三张面孔

### 1.1 经典定义

$$I(X; Y) = \mathbb{E}\!\left[\log \frac{p(X, Y)}{p(X)p(Y)}\right] = H(Y) - H(Y|X)$$

**解读**：知道 $X$ 后 $Y$ 的不确定性减少了多少。

### 1.2 几何视角：互信息与 KL 的等价

$$I(X; Y) = D_{\text{KL}}(p_{X,Y} \| p_X \otimes p_Y)$$

**几何**：互信息度量**联合分布偏离"独立分布"多远**。完全独立则 $I=0$。

### 1.3 数据处理不等式（关键约束）

若 $X \to Z \to Y$ 是马尔可夫链（$Z$ 由 $X$ 处理得到）：

$$I(X; Y) \geq I(Z; Y)$$

**深度学习含义**：网络每一层处理都不会"创造"新信息，只会丢失或保留。表示学习的目标是**保留预测相关信息**。

> 📖 互信息基础：[[02-KL散度→信息瓶颈]]

---

## 二、对比学习：互信息的神经估计

### 2.1 核心目标

学一个表示 $Z = f(X)$，使**同一对象的不同视角**（如同一图片的两个增强）互信息大：

$$\max_f I(Z_1; Z_2), \quad Z_1 = f(\text{aug}_1(X)), \ Z_2 = f(\text{aug}_2(X))$$

### 2.2 InfoNCE：互信息的下界

直接算 $I$ 不可行（需要联合分布）。对比学习用**噪声对比估计**给互信息下界：

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}\!\left[\log \frac{e^{s(z_1, z_2^+)/\tau}}{\sum_{z^-} e^{s(z_1, z^-)/\tau}}\right]$$

- $z_2^+$：正样本（同图另一增强）
- $z^-$：负样本（其他图）
- $s$：相似度（余弦）

**定理（van den Oord, 2018）**：最小化 NCE 等价于最大化互信息的下界，且该下界与负样本数 $K$ 的对数相关：

$$I(Z_1; Z_2) \geq \log K - \mathcal{L}_{\text{NCE}}$$

### 2.3 SimCLR / CLIP 的实质

- **SimCLR**：图像两个增强的表示互信息最大化
- **CLIP**：图像与对应文本的互信息最大化

两者都是"**用对比损失逼近互信息**"的实例。

---

## 三、RLHF：偏好作为隐变量

### 3.1 RLHF 三阶段

1. **SFT**：监督微调，学基础语言能力
2. **奖励模型 RM**：学人类偏好的标量奖励 $r(x, y)$
3. **PPO**：用 RL 优化策略 $\pi$ 最大化奖励，同时用 KL 约束防止偏离太远

### 3.2 奖励模型：Bradley-Terry 偏好模型

给定人类对 $(y_w \succ y_l)$（$y_w$ 比 $y_l$ 好）的偏好，假设偏好概率服从 Bradley-Terry：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

训练 RM 即最大化该似然（等价于二元交叉熵）。**RM 在捕捉"人类偏好"这一隐变量的信息**。

### 3.3 PPO 阶段：KL 约束的策略优化

目标：

$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta \cdot D_{\text{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

- 第一项：最大化奖励（让回答更符合人类偏好）
- 第二项：**KL 约束**——策略不要偏离参考模型 $\pi_{\text{ref}}$（SFT 模型）太远

**信息论解读**：KL 项限制策略"信息偏离量"。没有它，模型会为了刷高奖励而胡言乱语（reward hacking）；有它，模型在"**偏好信息增益**"与"**语言先验保持**"间权衡。

### 3.4 DPO：绕过 RL 的简化

直接偏好优化（DPO, 2023）利用了奖励与策略的闭式关系，把 RL 目标转成纯监督损失：

$$\mathcal{L}_{\text{DPO}} = -\log\sigma\!\left(\beta\log\frac{\pi(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**优势**：不需要训 RM、不需要 PPO 采样，纯交叉熵训练。本质仍是在偏好分布与策略分布间做 KL 约束的优化。

---

## 四、工程实现层

### 4.1 InfoNCE（对比学习）

```python
import torch
import torch.nn.functional as F

def info_nce(z1, z2, temperature=0.1):
    # z1, z2: (B, d) 同 batch 的两个增强视图
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature        # (B, B) 相似度矩阵
    labels = torch.arange(z1.size(0))
    # 对角线是正样本, 其余是负样本
    return F.cross_entropy(logits, labels)
```

### 4.2 奖励模型训练（Bradley-Terry）

```python
def rm_loss(rm, x, y_win, y_lose):
    r_w = rm(x, y_win)       # 标量奖励
    r_l = rm(x, y_lose)
    return -F.logsigmoid(r_w - r_l).mean()   # 二元交叉熵
```

### 4.3 PPO 的 KL 惩罚

```python
def ppo_loss(policy, ref_policy, rm, x, beta=0.1):
    y, logp = policy.generate(x, return_logp=True)
    with torch.no_grad():
        logp_ref = ref_policy.logp(x, y)
    reward = rm(x, y)
    kl = (logp - logp_ref).mean()   # KL 的蒙特卡洛估计
    return -(reward - beta * kl).mean()
```

### 4.4 DPO 损失

```python
def dpo_loss(policy, ref_policy, x, y_w, y_l, beta=0.1):
    logp_w = policy.logp(x, y_w)
    logp_l = policy.logp(x, y_l)
    with torch.no_grad():
        logp_w_ref = ref_policy.logp(x, y_w)
        logp_l_ref = ref_policy.logp(x, y_l)
    logits = beta * ((logp_w - logp_w_ref) - (logp_l - logp_l_ref))
    return -F.logsigmoid(logits).mean()
```

---

## 五、互信息视角能解释的现象

### 5.1 为什么对比学习有效

对比损失最大化同一对象不同视图的互信息。**表示被迫保留"增强不变"的信息**（语义），丢弃"增强敏感"的信息（细节）——这正是信息瓶颈原则的实例。

### 5.2 为什么 RLHF 需要 KL 约束

没有 KL，策略会为最大化奖励而退化（如重复"好评"词）。KL 项保持策略接近语言先验，相当于**信息论正则化**——只让偏好信息"有控制地"流入策略。

### 5.3 大模型的能力 vs 对齐

预训练最大化"下一词"互信息 $I(\text{上下文}; \text{下一词})$（即最小化交叉熵）。RLHF 在此基础上**对齐人类偏好**——两个阶段分别处理"能力"和"方向"。

---

## 六、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  互信息 I(X;Y) = D(p_XY ‖ p_X p_Y)                        │
│  数据处理不等式: 处理只丢不创信息                            │
│  [[Mathematics-Universe/05-概率论与数理统计/04-数字特征/...]] │
└──────────────────────────┬───────────────────────────────┘
                           │ 互信息的神经估计与优化
                           ▼
┌──────────────────────────────────────────────────────────┐
│  对比学习 & RLHF                                            │
│  对比: max I(视图1; 视图2), 用 InfoNCE 估下界              │
│  RLHF: 偏好隐变量 + PPO(KL约束) / DPO(闭式简化)            │
│  共同: 用 KL/互信息 做信息论正则                            │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  InfoNCE: 相似度矩阵 + cross_entropy                      │
│  RM: logsigmoid(r_w - r_l)                                │
│  PPO: reward - β·KL;  DPO: logsigmoid(β·logratio 差)      │
└──────────────────────────────────────────────────────────┘
```

---

## 七、直觉总结

```
互信息: 两变量共享多少信息

对比学习: 同对象的两视图应该高互信息(语义不变)
          InfoNCE = 互信息下界的最大化

RLHF:    奖励 = 偏好隐变量的信息
         PPO: max 奖励 - β·KL(策略‖参考)
              ↑ 信息论正则, 防止为刷奖励而胡言
         DPO: 把 RL 目标闭式化成监督损失
```

**互信息是连接表示学习与对齐的数学主线。** 对比学习用它学表示，RLHF 用它（经 KL）约束对齐。

---

## 八、延伸阅读

### 论文
- **van den Oord et al. (2018)** "Representation Learning with Contrastive Predictive Coding"——InfoNCE
- **Chen et al. (2020)** "SimCLR"——对比学习代表
- **Radford et al. (2021)** "CLIP"——图文互信息
- **Christiano et al. (2017)** "Deep RL from Human Preferences"——RLHF 奠基
- **Rafailov et al. (2023)** "Direct Preference Optimization"——DPO

### 关联文章
- [[01-熵→交叉熵损失]]、[[02-KL散度→信息瓶颈]]（信息论基础）
- [[PART-05/05-RLHF的博弈论视角]]（RLHF 的博弈论分析）

---

## 联系网络

⬆ 上游：[[02-KL散度→信息瓶颈]]（互信息 = KL 的期望），[[Mathematics-Universe/05-概率论与数理统计/04-数字特征/数字特征详解.md]]

⬇ 下游：[[PART-05/05-RLHF的博弈论视角]]（博弈论下的 RLHF）

↔ 横联：[[01-熵→交叉熵损失]]、[[PART-02/05-损失函数]]（对比/偏好损失在损失家族中的位置）

🔗 跨域：神经科学（互信息与神经编码）、经济学（偏好建模）、信息检索（学习排序）

━━━━━━━━━━━━━━
