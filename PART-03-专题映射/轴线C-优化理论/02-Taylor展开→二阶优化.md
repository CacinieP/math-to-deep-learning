# Taylor 展开 → 二阶优化

> 梯度下降只用到了目标函数的**一阶信息**（梯度），二阶优化利用**二阶信息**（Hessian矩阵）。两者之间的差距，就是Taylor展开告诉我们"曲率"在局部的近似有多精确。

**难度**：[进阶]（需要多元微积分 + 熟悉梯度下降）

## 一、纯数学版

### 1.1 多元 Taylor 展开

若 $f$ 在邻域内二阶可微且 Hessian 局部 Lipschitz（例如三阶连续可微），在 $\mathbf{x}_0$ 处的 Taylor 展开：

$$f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T(\mathbf{x} - \mathbf{x}_0) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_0)^T H(\mathbf{x}_0)(\mathbf{x} - \mathbf{x}_0) + O(\|\mathbf{x} - \mathbf{x}_0\|^3)$$

其中：
- $\nabla f(\mathbf{x}_0)$：梯度（一阶）
- $H(\mathbf{x}_0)$：Hessian 矩阵（二阶）
- $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

> 📖 多元 Taylor 展开：[一元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/02-一元微分学/一元微分学详解.md#三taylor公式)

### 1.2 为什么二阶信息重要

在一阶近似下（梯度下降），目标函数被近似为：

$$f(\mathbf{x}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^T(\mathbf{x} - \mathbf{x}_0)$$

这是一个**线性函数**——等高线是平行直线。梯度下降沿着最陡的方向走。

但真实的函数可能是这样的：

```
一阶近似（线性）:
    等高线:     ∇f
              ╱
            ╱   ← 最陡下降
          ╱

二阶近似（二次）:
         ╭──╮
       ╭─┤  ├─╮        ← Hessian 描述的曲率
       │ │  │ │
     ╭─┤─┤  ├─┤─╮
     │ │ │  │ │ │
       ╰─┤  ├─╯
         ╰──╯
        ▽         ← 真正的极小值（不在最陡方向）
```

**二阶项告诉我们"最陡方向"在局部是有误导的**——因为曲率在不同方向上不同。

### 1.3 驻点处的二阶信息

在驻点 $\nabla f(\mathbf{x}^*) = 0$ 处，Taylor 展开简化为：

$$f(\mathbf{x}) \approx f(\mathbf{x}^*) + \frac{1}{2}(\mathbf{x} - \mathbf{x}^*)^T H(\mathbf{x}^*)(\mathbf{x} - \mathbf{x}^*)$$

| Hessian 性质 | 极值类型 |
|-------------|---------|
| 所有特征值 > 0（正定） | 局部极小 |
| 所有特征值 < 0（负定） | 局部极大 |
| 特征值有正有负（不定） | 鞍点 |
| 半正定/半负定且有零特征值 | 二阶判别不充分，需进一步分析；若同时有正负特征值，仍已可判为鞍点 |

> 📖 正定性判别：[二次型详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/04-线性代数/06-二次型/二次型详解.md#31-正定判别)

---

## 二、Newton 法：二阶优化的原型

### 2.1 用二次模型近似

在每一步，用当前点的二阶 Taylor 展开构造一个**二次模型**：

$$m_k(\mathbf{p}) = f(\mathbf{x}_k) + \nabla f(\mathbf{x}_k)^T \mathbf{p} + \frac{1}{2}\mathbf{p}^T H(\mathbf{x}_k) \mathbf{p}$$

**当 Hessian 正定时**，Newton 步最小化这个二次模型；Hessian 不定时驻点不是最小值：

$$\mathbf{p}_k = \arg\min_{\mathbf{p}} m_k(\mathbf{p})$$

求导 = 0：
$$\nabla f(\mathbf{x}_k) + H(\mathbf{x}_k)\mathbf{p}_k = 0$$

$$\mathbf{p}_k = -H(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k)$$

**迭代**：
$$\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{p}_k = \mathbf{x}_k - H(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k)$$

### 2.2 Newton 法 vs 梯度下降

| | 梯度下降 | Newton 法 |
|--|---------|----------|
| 步长 | $-\eta \nabla f$ | $-H^{-1} \nabla f$ |
| 方向 | 最陡下降方向 | 考虑了曲率的方向 |
| 步长大小 | 需要手动选择 $\eta$ | 自动（由 Hessian 决定） |
| 收敛速度 | 线性（误差每步乘常数，$e_{k+1} \leq C e_k$） | 二次（误差每步平方，$e_{k+1} \leq C e_k^2$） |
| 每步计算 | 一次梯度求值 + $O(n)$ 更新 | Hessian 构造 + 稠密线性求解 $O(n^3)$ |
| 内存 | $O(n)$ | $O(n^2)$（存储 Hessian） |

这些速度需要条件：梯度下降在光滑强凸目标及适当步长下线性收敛；Newton 在非退化极小点附近、Hessian 足够光滑且初值足够近时二次收敛。一般凸或非凸函数不具备这些保证。

**代价**：存储和求逆 $H$ 需要 $O(n^2)$ 内存和 $O(n^3)$ 计算。对于百万参数网络，这是不可行的。

### 2.3 手动实现 Newton 法

```python
import torch

def newton_step(f_value, x):
    """
    一步 Newton 法
    f_value: 已在 x 处计算的标量张量
    x: 参数向量 (n,)
    """
    # 一阶：梯度
    grad = torch.autograd.grad(f_value, x, create_graph=True)[0]  # (n,)

    # 二阶：Hessian（通过逐元素二阶导数）
    n = x.shape[0]
    H = x.new_zeros(n, n)
    for i in range(n):
        H[i] = torch.autograd.grad(grad[i], x, retain_graph=True)[0]

    # Newton 步: p = -H^(-1) * grad
    # 实际用线性求解（更稳定）
    p = torch.linalg.solve(H, -grad)

    # 阻尼并不自动保证下降；正式算法需线搜索或信赖域
    # 如果 f(x + p) > f(x)，说明步长太大，缩小
    return p

# 测试：优化 Rosenbrock 函数
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 初始点 (远离极小值)
x_init = torch.tensor([-1.0, 1.0], requires_grad=True)

for i in range(10):
    f = rosenbrock(x_init[0], x_init[1])
    p = newton_step(f, x_init)
    with torch.no_grad():
        x_init += p * 0.1  # 阻尼因子
    print(f"Step {i}: f = {f.item():.6f}, x = ({x_init[0]:.4f}, {x_init[1]:.4f})")
```

---

## 三、从 Newton 法到深度学习优化器

### 3.1 为什么不用 Newton 法训练神经网络

**问题一：Hessian 太大**
- 对于有 $n$ 个参数的模型，$H$ 是 $n \times n$ 矩阵
- ResNet-50：约 2500 万参数 → Hessian 需要 $2500万 \times 2500万$ 个元素 = $6.25 \times 10^{14}$ 个元素 ≈ 2.5 PB 内存

**问题二：Hessian 难以精确计算**
- $H_{ij} = \frac{\partial^2 L}{\partial w_i \partial w_j}$ 需要两遍自动微分
- 即使只算 Hessian-vector product $Hv$，也需要保留计算图

**问题三：非凸损失 landscape**
- 深度网络的损失函数有大量鞍点和局部极小
- Hessian 在鞍点处有负特征值 → Newton 步方向不可靠

### 3.2 折中方案：拟 Newton 法

**核心思想**：不计算精确的 Hessian，而是用**易于更新的近似矩阵** $B_k \approx H(\mathbf{x}_k)$。

**BFGS**：通过秩2更新维护 $B_k$ 的近似：

$$B_{k+1} = B_k + \frac{y_k y_k^T}{y_k^T s_k} - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}$$

其中 $s_k = \mathbf{x}_{k+1} - \mathbf{x}_k$, $y_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$。

当 $B_k\succ0$ 且 $y_k^Ts_k>0$ 时上述 BFGS 更新保持正定；通常用 Wolfe 线搜索或阻尼维护曲率条件。

**L-BFGS**：BFGS 的有限内存版本，只存储最近 $m$ 个 $(s, y)$ 对（通常 $m=10$），内存从 $O(n^2)$ 降到 $O(mn)$。

```python
# PyTorch 内置 L-BFGS
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, history_size=10)

# L-BFGS 需要重新定义闭包（因为需要多次函数求值）
def closure():
    optimizer.zero_grad()
    output = model(input)
    loss = loss_fn(output, target)
    loss.backward()
    return loss

optimizer.step(closure)
```

### 3.3 现代优化器的二阶视角

| 优化器 | 核心思想 | 二阶近似 | 适用 |
|--------|---------|---------|------|
| SGD | 一阶梯度 + 动量 | 无 | 大规模训练 |
| **Adam** | 自适应学习率 | 梯度二阶矩预条件（不等于 Hessian） | 大多数场景 |
| **AdaGrad** | 累积梯度平方 | 梯度二阶矩预条件 | 稀疏特征 |
| **AdamW** | Adam + 解耦权重衰减 | 同上 | 大模型训练 |
| L-BFGS | 秩2更新的拟 Newton | 满秩 Hessian 近似 | 小模型/微调 |

**Adam 的"二阶"解释**：
- Adam 维护每个参数的梯度的一阶矩（动量）和二阶矩（梯度平方的移动平均）
- $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
- 更新：$w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$
- 分母 $\sqrt{\hat{v}_t}$ 按历史梯度尺度预条件。梯度的二阶矩不是损失的二阶导数，Adam 仍是一阶方法，不能一般视为 Hessian 的对角近似

---

## 四、二阶优化在深度学习中的实际应用

### 4.1 学习率调度的二阶解释

考虑一维情况（单参数），Newton 步：

$$w_{k+1} = w_k - \frac{f'(w_k)}{f''(w_k)}$$

- $f''(w_k) > 0$（曲率大，山谷陡）：步长小（保守）
- $f''(w_k) > 0$ 且数值小（曲率小，山谷平缓）：步长大（大胆）——自适应步长的论证只在这一情形（正曲率）下成立
- $f''(w_k) < 0$（负曲率，局部上凸）：二次模型没有极小值，Newton 步指向"上山"方向、方向不可靠——这是经典危险情形，需要信赖域（trust region）或 Levenberg-Marquardt 式正则化来兜底

**这就是自适应学习率的直觉来源**——不同参数有不同的"曲率"，应该用不同的步长。

### 4.2 二阶信息在 NLP 中的使用

**K-FAC**（Kronecker-Factored Approximate Curvature）：利用神经网络的全连接层和卷积层的结构，将 **Fisher 信息矩阵**（自然梯度方法中的曲率度量，而非 Hessian 本身）近似为 Kronecker 积：

$$F_{\text{Fisher}} \approx A \otimes B$$

这是 **Kronecker 因子分解**（而非低秩分解）：每层只需两个小矩阵 $A$、$B$（分别来自该层输入激活和输出梯度的二阶统计），存储和求逆它们远比处理满矩阵便宜。

```python
# 概念：K-FAC 对单层的近似
# 对于 y = Wx (W: [out, in], x: [in])
# Jacobian = W (对输入的导数)
# 列优先向量化权重，令 δ 为对该层预激活的对数似然梯度：
# Fisher 块 F = E[(x x^T) ⊗ (δ δ^T)] ≈ E[x x^T] ⊗ E[δ δ^T]
# 可以分解为两个小矩阵的 Kronecker 积
# 求逆: (A⊗B)^(-1) = A^(-1) ⊗ B^(-1)
# 这样 n×n 的求逆变成了两个小矩阵的求逆
```

**K-FAC 在大型语言模型训练中的应用**：
- K-FAC 是按层近似 Fisher 的方法，原始研究报告了特定网络上的优化收益。
- 是否加速某个 Transformer 必须比较总运行时间、统计估计与通信成本，不能由曲率近似直接推出。参见 [Martens 与 Grosse 原论文](https://proceedings.mlr.press/v37/martens15.html)。

### 4.3 Newton-CG 方法

**Newton-CG**（Newton Conjugate Gradient）：用共轭梯度法求解 Newton 方程 $Hp = -g$，**不需要显式构建 Hessian**。标准 CG 要求系数矩阵对称正定；非凸问题使用截断 CG 检测负曲率，或配合阻尼/信赖域。

```
Newton-CG 迭代:
    1. 用 CG 求解 Hp = -g（只需要 H*v，不需要 H 本身）
    2. 做线性搜索
    3. 更新参数

    对于 n 维参数:
    - 存储: O(n)（不存 H）
    - 每步 CG 迭代: 一次 H*v 计算 = 两次反向传播
```

---

## 五、Hessian 的几何意义

### 5.1 Hessian 的特征值分解

$$H = Q\Lambda Q^T$$

- **特征值 $\lambda_i$**：沿第 $i$ 个特征值方向的**曲率**
  - $\lambda_i > 0$：该方向是"山谷"（向上凸）
  - $\lambda_i < 0$：该方向是"山脊"（向下凸）
  - $|\lambda_i|$ 越大：曲率越陡
- **特征向量 $\mathbf{q}_i$**：**主曲率方向**

**正定 Hessian 的条件数** $\kappa_2(H) = \lambda_{\max}/\lambda_{\min}$（不定情形要用奇异值，奇异时条件数无穷）：
- $\kappa \approx 1$：接近球面（所有方向曲率相同），优化容易
- $\kappa \gg 1$：像细长的椭球（某些方向极陡，某些极平），优化困难

### 5.2 深度网络的 Hessian 谱

研究表明深度网络的 Hessian 矩阵有一个**特殊结构**：

```
特征值分布:
                    ╭╮
                  ╭─┤─╮
              ╭───┤ │ ├───╮
  ───────────┼───┤ │ ├───┼───────────
              ╰───┤ │ ├───╯
                  ╰─┤─╯
                    ╰╯

  大部分特征值集中在 0 附近（"平坦区域"）
  少数特征值很大（"尖锐区域"）
```

- **零空间**：大部分特征值 $\approx 0$ 意味着 Hessian 接近奇异——参数空间中有很多"平坦方向"
- **少数大特征值**：对应损失 landscape 中的"尖锐极小"
- **Hessian 的迹 / 有效秩**：用来衡量优化难度

---

## 六、直觉总结

**Taylor 展开在优化中的角色**：

```
一阶优化（梯度下降）:          二阶优化（Newton/拟Newton）:
                              f(x) ≈ f(x₀) + ∇f·Δx + ½Δx^T H Δx
f(x) ≈ f(x₀) + ∇f·Δx              ↓
      ↓                        最小化这个二次模型
  最陡下降                           ↓
  固定步长                          自动步长（考虑曲率）
  每步 O(n)                        每步 O(n²) ~ O(n³)
  收敛慢（线性）                    收敛快（二次）
```

**一句话总结**：Taylor 展开告诉我们，目标函数在局部可以用"线性函数（梯度）"或"二次函数（梯度+Hessian）"近似。梯度下降只用线性近似，Newton 法用二次近似。二次近似更精确，但计算代价大。Adam 等一阶方法用梯度统计调节各参数步长；L-BFGS、K-FAC 等方法才显式构造曲率近似，不能混为一谈。

---

## 七、延伸阅读

### 论文
- **L-BFGS**：Liu & Nocedal (1989) — 有限内存拟 Newton
- **K-FAC**：Martens & Grosse (2015) — 利用 Kronecker 结构
- **Hessian 谱**：Sagun et al. (2016) — 深度网络 Hessian 的经验研究
- **Sharpness-Aware Minimization**：Foret et al. (2020) — 寻找平坦极小值

### 代码
- [PyTorch L-BFGS](https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html)
- [PyTorch Hessian](https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html) — 自动计算 Hessian（小模型用）

### 关联文章
- [梯度 → 反向传播](01-梯度→反向传播.md)（轴线C上一篇）
- [Lagrange乘数 → 约束优化](03-Lagrange乘数→约束优化.md)（轴线C第三篇）
- [梯度消失爆炸的数学根源](../../PART-04-从理论到工程/02-梯度消失爆炸的数学根源.md)

---

## 联系网络

⬆ 上游: [一元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/02-一元微分学/一元微分学详解.md)（Taylor展开），[多元微分学详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/03-高等数学/04-多元微分学/多元微分学详解.md)（多元微分·Hessian），[二次型详解](https://github.com/CacinieP/Mathematics-Universe/blob/main/04-线性代数/06-二次型/二次型详解.md)（正定判别·Hessian几何意义）

⬇ 下游: [Lagrange乘数 → 约束优化](03-Lagrange乘数→约束优化.md)

↔ 横联: [01-梯度→反向传播](01-梯度→反向传播.md)（反向传播计算梯度），[学习率调度（优化算法）](../../PART-02-深度学习核心/03-优化算法.md)

🔗 跨域: 科学计算（Newton法求解非线性方程），经济学（效用最大化·二阶条件），控制理论（LQR用Hessian近似）

━━━━━━━━━━━━━━
