# Weierstrass 逼近 → 通用近似定理

> 纯数学说：任何连续函数都能被多项式一致逼近（Weierstrass, 1885）。深度学习说：只要有一层隐单元足够宽的非线性网络，能逼近紧集上任何连续函数（Cybenko, 1989）。两者是同一条"函数逼近"思想在两个时代的回响——前者用多项式做基，后者用 sigmoid 做基。

**难度**：[标准]（需要连续性 + 一致收敛 + 一点泛函直觉）

## 一、纯数学版：Weierstrass 逼近定理

### 1.1 定理陈述

**Weierstrass 第一逼近定理（1885）**：设 $f \in C[a,b]$（闭区间上连续函数）。则对任意 $\varepsilon > 0$，存在多项式 $p$ 使：

$$\|f - p\|_\infty = \sup_{x \in [a,b]} |f(x) - p(x)| < \varepsilon$$

**含义**：多项式在连续函数空间中**稠密**。再"难看"的连续函数，都能被多项式一致逼近到任意精度。

> 📖 连续函数空间：[极限与连续详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/01-极限与连续/极限与连续详解.md)

### 1.2 Bernstein 多项式构造性证明

对 $f \in C[0,1]$，定义 Bernstein 多项式：

$$B_n(f)(x) = \sum_{k=0}^n f\left(\frac{k}{n}\right) \binom{n}{k} x^k (1-x)^{n-k}$$

**这其实是二项分布期望的形式**：$B_n(f)(x) = \mathbb{E}[f(K/n)]$，$K \sim \text{Bin}(n, x)$。

**证明思路**：
- $K/n \to x$ 依概率（大数定律）
- $f$ 连续 $\Rightarrow f(K/n) \to f(x)$
- 一致收敛来自 $f$ 的一致连续性 + Chebyshev 不等式

**意义**：不仅证了存在，还给了**显式构造**——概率论成了分析学的工具。

> 📖 大数定律：[大数定律与中心极限定理详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/05-大数定律与中心极限定理/大数定律与中心极限定理详解.md)

### 1.3 函数空间的视角

把 $C[a,b]$ 看作**赋范线性空间**（上确界范数）。Weierstrass 定理说：

$$\overline{\text{span}\{1, x, x^2, \ldots\}} = C[a,b]$$

多项式张成的子空间在 $C[a,b]$ 中**稠密**。这是泛函分析里"基"概念的雏形——多项式族线性张成稠密，但稠密性不等于 Schauder 基；并非每个连续函数都能以唯一的单项式级数一致展开。

> 📖 函数空间：[数值分析](https://github.com/CacinieP/Mathematics-Universe/blob/main/06-超纲拓展/数值分析.md)

---

## 二、从多项式到神经网络：换一组"基"

### 2.1 通用近似定理（Cybenko, 1989）

设 $\sigma$ 是**连续** sigmoidal 函数，例如 logistic sigmoid。则函数族：

$$\mathcal{F} = \left\{ g(x) = \sum_{i=1}^N \alpha_i \sigma(w_i^\top x + b_i) : N \in \mathbb{N}, \alpha_i \in \mathbb{R}, w_i \in \mathbb{R}^d, b_i \in \mathbb{R} \right\}$$

在 $C(K)$（$K \subset \mathbb{R}^d$ 紧）中**稠密**。

**对比 Weierstrass**：
- Weierstrass 的"基"：$\{1, x, x^2, \ldots\}$
- Cybenko 的"基"：$\{\sigma(w^\top x + b) : w, b\}$

两者都说：**某种简单函数族的线性组合能逼近任何连续函数**。

### 2.2 Hahn-Banach 定理的妙用（证明核心）

Cybenko 的证明用了**泛函分析的对偶论证**：

1. 假设 $\mathcal{F}$ 在 $C(K)$ 中**不**稠密
2. 由 Hahn-Banach 定理，存在非零有界线性泛函 $T$ 湮灭 $\mathcal{F}$：$T(g) = 0, \forall g \in \mathcal{F}$
3. 由 Riesz 表示定理，$T(f) = \int f\,d\mu$（某个测度 $\mu$）
4. 但可以证明（用 $\sigma$ 的性质 + 唯一性定理）：$T$ 必然为零泛函——矛盾

**意义**：纯泛函分析的工具给出了神经网络表达能力的根基。

### 2.3 多元与多层的推广

- **Hornik (1991)**：在有限测度的 $L^p$ 等适当意义下可推广到可测函数；不代表对不连续函数也能用连续网络一致逼近
- **深度分离**：对特定构造的函数族与误差度量，深层网络可比浅层网络节省指数级单元；不是对每个目标函数都成立

---

## 三、宽度 vs 深度：逼近效率的鸿沟

### 3.1 单层要多少神经元

网格插值的直觉是：在固定大小的 $d$ 维区域上，Lipschitz 常数为 $L$ 的函数要控制误差 $\varepsilon$，网格单元数通常随下面的量级增长：

$$N = O\left(\left(\frac{L}{\varepsilon}\right)^d\right)$$

**维数灾难**：一般函数的这种网格构造随维度很昂贵；这不是不带条件的单隐层网络最优宽度定理，结构化函数可能容易得多。

### 3.2 深度的指数加速

某些函数（如 parity 奇偶函数、波的叠加），**深度 $L$ 的网络用 $O(L)$ 个神经元**能做到的，**单层需要 $O(2^L)$ 个**。

```
单层(宽):  ▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢▢   指数宽
深度(窄):  ▢→▢→▢→▢→▢→▢→▢→▢        线性深
```

**这就是深度学习"深"的数学理由**：深度换宽度，指数级节省参数。

---

## 四、工程实现层：可视化通用近似

### 4.1 用 ReLU 网络逼近 sin

```python
import torch
import torch.nn as nn
import numpy as np

x = torch.linspace(-np.pi, np.pi, 200).unsqueeze(1)
y = torch.sin(x)

model = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(),
    nn.Linear(64, 1)
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(3000):
    loss = ((model(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

# 64 个 ReLU 单元就能把 sin 逼近得很好
```

### 4.2 观察宽度的效果

```python
import torch
import torch.nn as nn

def fit_sine(width, steps=3000):
    model = nn.Sequential(nn.Linear(1, width), nn.ReLU(), nn.Linear(width, 1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.linspace(-torch.pi, torch.pi, 200).unsqueeze(1)
    y = torch.sin(x)
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()

for width in [4, 16, 64, 256]:
    _, mse = fit_sine(width)
    print(f"width={width:3d}  final MSE={mse:.5f}")
# width=4:  棱角分明, 欠拟合（MSE 明显偏大）
# width=256: 几乎与 sin 重合（MSE 接近 0）
```

### 4.3 深度 vs 宽度的实验

```python
# 浅宽 vs 深窄（注意：两者参数量并不相同）
shallow = nn.Sequential(nn.Linear(1, 256), nn.ReLU(), nn.Linear(256, 1))   # 2 层, 256 宽, 769 参数
deep    = nn.Sequential(nn.Linear(1, 32), nn.ReLU(),
                        nn.Linear(32, 32), nn.ReLU(),
                        nn.Linear(32, 32), nn.ReLU(),
                        nn.Linear(32, 32), nn.ReLU(),
                        nn.Linear(32, 1))                                   # 5 层, 32 宽, 3265 参数
# 浅宽 769 参数 vs 深窄 3265 参数——这里只展示架构，不能用这组不同参数预算的示例证明深度省参数
```

---

## 五、定理的边界：能逼近 ≠ 能学到

通用近似定理只说**存在性**，不保证：

1. **可学性**：梯度下降能否找到那个好的逼近？现实中常陷局部极小
2. **效率**：所需宽度可能大到不可行
3. **泛化**：逼近训练数据 ≠ 逼近真实函数（过拟合）

**这就是为什么需要深度、正则化、好的优化器**——定理给的是"理论上能"，工程要解决"实际上能"。

---

## 六、完整映射图

```
┌──────────────────────────────────────────────────────────┐
│  纯数学                                                    │
│  Weierstrass: 多项式在 C[a,b] 稠密                         │
│  Bernstein: 概率构造性证明                                 │
│  泛函: 函数空间 + 稠密性 + 范数                            │
│  [[Mathematics-Universe/03-高等数学/01-极限与连续/...]]     │
└──────────────────────────┬───────────────────────────────┘
                           │ 换一组基: 多项式 → 神经元
                           ▼
┌──────────────────────────────────────────────────────────┐
│  通用近似定理 (Cybenko 1989)                               │
│  单层 sigmoid 网络在 C(K) 稠密                             │
│  证明工具: Hahn-Banach + Riesz + 测度论                    │
│  深度 > 宽度: 指数级节省参数                                │
└──────────────────────────┬───────────────────────────────┘
                           │ 工程实现
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PyTorch                                                   │
│  ReLU 网络逼近任意函数(宽度足够)                           │
│  深网络用更少参数拟合复杂函数                              │
│  ⚠ 定理只保证存在性, 不保证能学到                          │
└──────────────────────────────────────────────────────────┘
```

---

## 七、直觉总结

```
Weierstrass(1885): 任何连续函数 ≈ 多项式(基: 1, x, x², ...)
Cybenko  (1989):  任何连续函数 ≈ Σαᵢσ(wᵢ·x+bᵢ)(基: 神经元)

神经网络 = 用"神经元"做基的函数逼近器
深度 = 用更少基函数达到同样精度 (指数加速)
⚠ 能逼近 ≠ 能用梯度下降学到
```

**神经网络是个万能函数逼近器，这是它能工作的理论基础。** 但"理论上能"和"实际学得到"之间，横亘着优化、泛化、深度的全部工程难题。

---

## 八、延伸阅读

### 论文
- **Weierstrass (1885)** "Über die analytische Darstellbarkeit sogenannter willkürlicher Functionen"
- **Cybenko (1989)** "Approximation by Superpositions of a Sigmoidal Function"——通用近似定理
- **Hornik (1991)** "Approximation Capabilities of Multilayer Feedforward Networks"
- **Telgarsky (2016)** "Benefits of Depth in Neural Networks"——深度指数优势

### 关联文章
- [02-Fourier分析→注意力机制](02-Fourier分析→注意力机制.md)（另一组基：三角函数）
- [03-正交基→残差与变换](03-正交基→残差与变换.md)（正交基与残差网络）
- [01-神经网络基础](../../PART-02-深度学习核心/01-神经网络基础.md)（被逼近的函数结构）

---

## 联系网络

⬆ 上游：[极限与连续详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/01-极限与连续/极限与连续详解.md)（连续性与一致收敛），[大数定律与中心极限定理详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/05-概率论与数理统计/05-大数定律与中心极限定理/大数定律与中心极限定理详解.md)（Bernstein 证明）

⬇ 下游：[02-Fourier分析→注意力机制](02-Fourier分析→注意力机制.md)，[03-正交基→残差与变换](03-正交基→残差与变换.md)

↔ 横联：[01-神经网络基础](../../PART-02-深度学习核心/01-神经网络基础.md)（网络结构），[02-梯度消失爆炸的数学根源](../../PART-04-从理论到工程/02-梯度消失爆炸的数学根源.md)（深度的代价）

🔗 跨域：数值分析（多项式插值、样条）、信号处理（滤波器设计）

━━━━━━━━━━━━━━
