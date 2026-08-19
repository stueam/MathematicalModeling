# A/B 题知识与算法卡片库

本文覆盖前面清单中的 30 个知识卡片和 33 个算法卡片。每张卡片按统一格式组织：

```text
Name
Description and core knowledge
A specific applicable scene
Code, Python first
```

代码定位是“比赛中的最小可用写法”，不是完整工程实现。真正比赛时应把这些片段沉入 `templates/`，补齐输入检查、单位说明、随机种子和输出记录。

## 目录

1. 数学基础知识卡片
2. 物理与机理建模知识卡片
3. 优化与决策知识卡片
4. 数据、信号与反演知识卡片
5. 数值计算算法卡片
6. 几何与仿真算法卡片
7. 优化算法卡片
8. 统计与机器学习算法卡片

---

# 一、数学基础知识卡片

## K01. 向量点积、叉积与夹角

### Name

向量点积、叉积与夹角。

### Description and core knowledge

点积衡量同方向成分，叉积给出垂直于两向量的有向面积向量。二者是几何建模、反射、遮挡、力矩、面积和法向量的基础。

核心公式：

- `dot(a,b) = |a||b|cos(theta)`
- `cross(a,b) = |a||b|sin(theta) n`
- 点到直线距离可用叉积模长除以底边长度。

### A specific applicable scene

2025A 烟幕干扰弹中，需要判断导弹到真目标的视线是否穿过烟幕球体。可先用向量夹角筛选，再做射线求交。

### Code

```python
import numpy as np

a = np.array([1.0, 2.0, 2.0])
b = np.array([2.0, 0.0, 0.0])

cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
theta = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
cross = np.cross(a, b)

print(theta, cross, np.linalg.norm(cross))
```

---

## K02. 矩阵与线性方程组

### Name

矩阵表示与线性方程组求解。

### Description and core knowledge

线性方程组 `Ax=b` 是离散化、最小二乘、图模型、平衡方程和参数估计的共同基础。要能判断方程组是唯一解、无解、欠定还是超定。

核心知识：

- 方阵满秩则唯一解。
- 超定系统常用最小二乘。
- 奇异或条件数过大会导致结果不稳定。
- 大规模稀疏矩阵应使用稀疏求解器。

### A specific applicable scene

2017A CT 成像中，像素吸收率与投影数据可写成线性方程组；探测器角度和像素网格构造完之后，就是求解不适定线性系统。

### Code

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 3.0],
              [0.0, 1.0]])
b = np.array([4.0, 5.0, 1.0])

print("condition number:", np.linalg.cond(A.T @ A))
x, residual, rank, _ = np.linalg.lstsq(A, b, rcond=None)
print(x, residual, rank)
```

---

## K03. 二维与三维解析几何

### Name

圆、球、圆柱、平面、螺线等曲线曲面方程。

### Description and core knowledge

A/B 题大量场景本质上是几何问题。需要熟练写出常用曲线曲面，并能把约束转化为距离、相交、切线、夹角条件。

常用对象：

- 圆：`(x-x0)^2+(y-y0)^2=R^2`
- 球：`(p-c)^T(p-c)=R^2`
- 平面：`n^T(p-p0)=0`
- 阿基米德螺线：`r=p theta/(2pi)`

### A specific applicable scene

2024A 板凳龙中，各把手位于等距螺线上，板凳长度成为相邻点之间的距离约束。

### Code

```python
import numpy as np

pitch = 0.55
theta = np.linspace(0, 8*np.pi, 500)
r = pitch * theta / (2*np.pi)
x = r * np.cos(theta)
y = r * np.sin(theta)

print("radius range:", r.min(), r.max())
```

---

## K04. 微积分与弧长

### Name

积分、面积、弧长与曲线指标。

### Description and core knowledge

积分用于总量、面积、功、能量、概率和期望。弧长公式是曲线运动建模的基础。

核心公式：

- 平面曲线弧长：`L = int sqrt(1+(dy/dx)^2) dx`
- 参数曲线弧长：`L = int ||r'(t)|| dt`
- 曲线下面积或区域面积可由数值积分得到。

### A specific applicable scene

2020A 炉温曲线中，要求超过 217 摄氏度到峰值温度覆盖的面积，需要数值积分计算曲线区域指标。

### Code

```python
import numpy as np
from scipy.integrate import quad

f = lambda t: np.sqrt(1 + np.cos(t)**2)
length, error = quad(f, 0, 2*np.pi)
print(length, error)

t = np.linspace(0, 1, 100)
y = np.exp(-t)
print(np.trapezoid(y, t))
```

---

## K05. 常微分方程

### Name

常微分方程与状态空间表示。

### Description and core knowledge

ODE 描述变量随时间的演化。比赛常见形式是二阶方程转化为两个一阶方程。

核心知识：

- 初值问题：`dy/dt=f(t,y), y(t0)=y0`
- 二阶系统：位置和速度组成状态。
- 稳态、周期解、收敛性和刚性需要检查。

### A specific applicable scene

2022A 波浪能装置中，浮子与振子的垂荡和纵摇可建立耦合 ODE，再计算 PTO 平均功率。

### Code

```python
import numpy as np
from scipy.integrate import solve_ivp

m, c, k = 1.0, 0.3, 4.0

def f(t, y):
    return [y[1], (-c*y[1] - k*y[0] + np.sin(2*t))/m]

sol = solve_ivp(f, (0, 20), [1.0, 0.0], max_step=0.01)
print(sol.y[0, -1], sol.y[1, -1])
```

---

## K06. 概率与统计

### Name

随机变量、置信区间与假设检验。

### Description and core knowledge

统计建模的核心是用有限样本推断总体规律，并量化不确定性。要区分置信度、显著性水平、功效和期望损失。

核心知识：

- 二项分布描述次品检测。
- 置信区间描述参数估计范围。
- 假设检验控制拒真和取伪错误。
- 随机仿真必须报告重复次数和波动范围。

### A specific applicable scene

2024B 抽样检测中，企业要以 95% 或 90% 信度决定是否接收零配件批次。

### Code

```python
from scipy import stats

n, k, p0 = 100, 12, 0.10
p_value = stats.binomtest(k, n, p0, alternative="greater").pvalue
ci = stats.binomtest(k, n).proportion_ci(0.95)

print(p_value, ci)
```

---

## K07. 量纲与量级分析

### Name

量纲一致性、单位换算与量级检查。

### Description and core knowledge

所有公式必须量纲一致；所有结果必须先通过量级检查。这是最快发现建模错误的手段。

核心做法：

- 建立统一单位制，如 SI。
- 检查左右两边量纲。
- 对时间、长度、速度、力、能量做合理范围估计。
- 对异常数值先查单位和索引，再改算法。

### A specific applicable scene

2025A 中导弹速度为 300 m/s，烟幕有效时间为 20 s，二者乘积给出约 6000 m 的相互作用尺度，可用来判断候选遮蔽位置是否合理。

### Code

```python
missile_speed = 300.0       # m/s
effective_time = 20.0       # s
interaction_scale = missile_speed * effective_time

print("interaction scale:", interaction_scale, "m")
assert interaction_scale > 0
```

---

# 二、物理与机理建模知识卡片

## K08. 质点与刚体受力分析

### Name

受力分析、力矩与平衡方程。

### Description and core knowledge

先画受力图，再写力和力矩平衡。对刚体，除合力为零外，合力矩也必须为零。

核心公式：

- `sum F = 0`
- `sum M = 0`
- 力矩 `M = r times F`

### A specific applicable scene

2016A 系泊系统中，钢缆、浮筒、风浪流载荷组成静平衡系统，需要逐段建立受力方程。

### Code

```python
import numpy as np
from scipy.optimize import root

def equations(x):
    T1, T2 = x
    Fx = -T1 + 2*T2/np.sqrt(5)
    Fy = T1/np.sqrt(2) + T2/np.sqrt(5) - 10
    return [Fx, Fy]

print(root(equations, [10, 10]).x)
```

---

## K09. 牛顿第二定律与状态方程

### Name

牛顿第二定律建模。

### Description and core knowledge

牛顿第二定律 `F=ma` 把受力翻译为加速度，再通过位置和速度状态方程求解运动轨迹。

核心写法：

```text
position' = velocity
velocity' = force / mass
```

### A specific applicable scene

2025A 烟幕干扰弹脱离无人机后受重力作用，可建立三维抛体运动方程。

### Code

```python
import numpy as np
from scipy.integrate import solve_ivp

g = 9.8
v0 = np.array([120.0, 0.0, 0.0])
p0 = np.array([17800.0, 0.0, 1800.0])

def motion(t, y):
    return [y[3], y[4], y[5], 0, 0, -g]

sol = solve_ivp(motion, (0, 5), [*p0, *v0], t_eval=np.linspace(0, 5, 11))
print(sol.y[:3, -1])
```

---

## K10. 阻尼、弹簧与共振

### Name

弹簧-阻尼-质量系统。

### Description and core knowledge

弹簧力与位移相关，阻尼力与速度相关。周期激励下系统会出现瞬态和稳态响应，共振由激励频率和固有频率关系决定。

核心公式：

- `F_spring = -kx`
- `F_damping = -cv`
- 固有频率：`omega_n = sqrt(k/m)`

### A specific applicable scene

2022A 中 PTO 系统包含弹簧和阻尼器，阻尼器做功即为输出能量。

### Code

```python
import numpy as np
from scipy.integrate import solve_ivp

m, k, c, omega = 1.0, 9.0, 0.4, 3.0

def system(t, y):
    x, v = y
    a = (-k*x - c*v + np.cos(omega*t))/m
    return [v, a]

sol = solve_ivp(system, (0, 50), [0, 0], t_eval=np.linspace(0, 50, 2000))
average_power = np.mean(c * sol.y[1, -1000:]**2)
print(average_power)
```

---

## K11. 传热与热阻

### Name

热传导、对流边界与热阻网络。

### Description and core knowledge

热传导由 Fourier 定律描述；稳态多层壁可用热阻串联简化；瞬态问题需要 PDE 或集中参数模型。

核心公式：

- `q = -k dT/dx`
- `R = L/(kA)`
- 边界对流：`q = hA(T_env-T_surface)`

### A specific applicable scene

2018A 高温作业专用服装是典型多层瞬态热传导问题。

### Code

```python
import numpy as np

L = np.array([0.0006, 0.006, 0.0035, 0.005])
k = np.array([0.082, 0.037, 0.045, 0.028])
A = 1.0
R = L/(k*A)
T_hot, T_cold = 75.0, 37.0
q = (T_hot-T_cold)/R.sum()

print("thermal resistance:", R)
print("heat flux:", q)
```

---

## K12. 流体质量守恒与孔口流量

### Name

质量守恒、可压缩性与孔口流。

### Description and core knowledge

控制体内质量变化等于流入减流出。孔口流量通常与压力差的关系由题面给出，要严格按照题面单位代入。

核心关系：

```text
dV/dt = Q_in - Q_out
pressure-density relation
Q = C A sqrt(2 Delta P / rho)
```

### A specific applicable scene

2019A 高压油管中，单向阀进入、喷油嘴喷出和燃油可压缩性共同决定管内压力变化。

### Code

```python
import numpy as np
from scipy.integrate import solve_ivp

C, A, rho = 0.85, 1.54e-6, 850.0

def pressure_ode(t, P):
    dP = 100e6 - P[0]
    Q = C*A*np.sqrt(max(2*abs(dP)/rho, 0))
    return [Q if dP > 0 else -Q]

sol = solve_ivp(pressure_ode, (0, 1), [100e6], max_step=0.001)
print(sol.y[0, -1])
```

---

## K13. 太阳位置与反射定律

### Name

太阳高度角、方位角与镜面反射。

### Description and core knowledge

太阳位置由纬度、赤纬角和时角决定。镜面反射满足入射角等于反射角，反射向量由法向量计算。

核心公式：

- `r = d - 2(d cdot n)n`
- 余弦效率等于入射光线与镜面法向量夹角余弦的绝对值。

### A specific applicable scene

2023A 定日镜场中，每个时刻都要计算太阳方向、反射方向、遮挡和截断效率。

### Code

```python
import numpy as np

def reflect(d, n):
    d = d/np.linalg.norm(d)
    n = n/np.linalg.norm(n)
    return d - 2*np.dot(d, n)*n

sun_direction = np.array([1.0, 0.2, -0.8])
mirror_normal = np.array([0.0, 0.0, 1.0])
reflected = reflect(sun_direction, mirror_normal)

cosine_efficiency = abs(np.dot(sun_direction, mirror_normal)/(
    np.linalg.norm(sun_direction)*np.linalg.norm(mirror_normal)))
print(reflected, cosine_efficiency)
```

---

## K14. 抛体运动与碰撞检测

### Name

抛体运动、时间窗与几何碰撞。

### Description and core knowledge

抛体在重力作用下运动，水平速度不变，竖直速度线性变化。碰撞检测要把连续轨迹与几何约束离散或解析求交。

核心公式：

- `x(t)=x0+v0t`
- `z(t)=z0+w0t-0.5gt^2`

### A specific applicable scene

2025A 中，烟幕弹从无人机投放后作抛体运动，起爆后的球形云团需要在导弹观测时间窗内遮挡视线。

### Code

```python
import numpy as np

g = 9.8
p0 = np.array([0.0, 0.0, 100.0])
v0 = np.array([50.0, 0.0, 0.0])
t = 3.0

p = p0 + v0*t + np.array([0, 0, -0.5*g*t**2])
print("position at 3s:", p)
```

---

## K15. 参数反演

### Name

从观测数据反推模型参数。

### Description and core knowledge

参数反演的核心是构造残差，让模型输出尽量接近观测数据。要讨论可辨识性、初值敏感性和不适定性。

关键问题：

- 参数是否有唯一解。
- 数据噪声如何传播。
- 是否需要正则化或附加约束。

### A specific applicable scene

2025B 中由红外干涉光谱反演外延层厚度，本质是光谱模型参数反演。

### Code

```python
import numpy as np
from scipy.optimize import least_squares

x = np.linspace(0, 1, 50)
true_theta = [2.0, 0.5, 0.1]
y_obs = true_theta[0]*np.exp(-true_theta[1]*x) + true_theta[2]
y_obs += np.random.default_rng(1).normal(0, 0.01, x.size)

def residual(theta):
    return theta[0]*np.exp(-theta[1]*x) + theta[2] - y_obs

res = least_squares(residual, [1.0, 0.1, 0.0], bounds=([0, 0, -1], [10, 2, 1]))
print(res.x, np.sqrt(np.mean(res.fun**2)))
```

---

# 三、优化与决策知识卡片

## K16. 线性规划与整数规划

### Name

线性目标、线性约束与整数决策。

### Description and core knowledge

当目标和约束均可线性表达，且决策变量连续时用 LP；决策变量必须取整时用 IP/MILP。建模关键是决策变量、目标函数和约束条件三件套。

### A specific applicable scene

2024B 生产决策中，是否检测零件、是否拆解不合格成品可用 0-1 变量或动态决策模型表示。

### Code

```python
from scipy.optimize import linprog

c = [-4, -3]
A_ub = [[2, 1], [1, 2]]
b_ub = [10, 8]
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(res.x, -res.fun)
```

---

## K17. 非线性约束优化

### Name

非线性目标与非线性约束优化。

### Description and core knowledge

物理仿真目标往往由数值计算得到，约束可能是安全阈值、几何范围或工程尺寸。要区分可行解、局部最优和全局最优。

核心方法：

- 少维问题用多初值局部优化。
- 多峰黑箱问题用全局优化。
- 先粗搜索，再局部精修。

### A specific applicable scene

2020A 炉温曲线中，温区温度和过炉速度是决策变量，制程界限是约束。

### Code

```python
from scipy.optimize import minimize

def objective(x):
    return (x[0]-2)**2 + (x[1]+1)**2

def constraint(x):
    return 1 - x[0] - x[1]

res = minimize(objective, x0=[0, 0], constraints={"type": "ineq", "fun": constraint})
print(res.x, res.fun)
```

---

## K18. 动态规划

### Name

多阶段决策与最优子结构。

### Description and core knowledge

动态规划适用于可分解为阶段、状态和转移的问题。定义状态是关键，状态必须包含影响未来决策的全部必要信息。

核心形式：

```text
V_t(state) = max_action { reward + V_{t+1}(next_state) }
```

### A specific applicable scene

2020B 穿越沙漠中，日期、位置、资金、水和食物共同构成状态。

### Code

```python
values = {3: 10.0}

for s in reversed(range(3)):
    candidates = []
    for a in [1, 2]:
        nxt = min(s+a, 3)
        candidates.append(a*2 + values[nxt])
    values[s] = max(candidates)

print(values)
```

---

## K19. 图与最短路

### Name

节点、边、路径成本与最短路。

### Description and core knowledge

图建模关注实体关系和转移成本。非负权重可用 Dijkstra，带启发函数时可用 A*。

### A specific applicable scene

2020B 沙漠地图中，相邻区域构成图，天气和行动消耗影响边权。

### Code

```python
import heapq

graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}

def dijkstra(graph, source, target):
    dist = {source: 0}
    pq = [(0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return None

print(dijkstra(graph, 0, 3))
```

---

## K20. 离散事件系统

### Name

事件队列、系统状态与时钟推进。

### Description and core knowledge

离散事件仿真按事件发生时间推进，而不是固定时间步长。事件可能改变资源占用、队列和后续事件。

核心要素：

- 系统状态
- 未来事件队列
- 事件处理规则
- 统计指标

### A specific applicable scene

2018B 智能 RGV 调度中，机床加工完成、RGV 到达、上下料、清洗和故障都可作为事件。

### Code

```python
import heapq

events = [(0.0, "machine_finish_1"), (3.0, "rgv_arrive"), (8.0, "fault")]

while events:
    time, event = heapq.heappop(events)
    if event.startswith("machine_finish"):
        print(time, "machine idle")
    elif event == "rgv_arrive":
        print(time, "unload/load")
        heapq.heappush(events, (time+0.5, "clean_finish"))
    elif event == "fault":
        print(time, "fault occurs")
```

---

## K21. 随机决策

### Name

随机环境下的期望、风险与鲁棒策略。

### Description and core knowledge

当天气、故障或参数不确定时，策略比较应基于期望收益、方差、失败概率或分位数，而不能只看单次结果。

常用准则：

- 期望值最优。
- 方差最小。
- 最坏情况约束。
- CVaR 风险度量。

### A specific applicable scene

2018B 中机床故障概率约 1%，调度策略需要通过蒙特卡洛重复评估。

### Code

```python
import numpy as np

rng = np.random.default_rng(42)
strategies = {
    "safe": [8, 9, 10, 11, 12],
    "risky": [0, 5, 10, 15, 20]
}

for name, reward in strategies.items():
    values = rng.choice(reward, size=(10000, 5), p=[.1, .2, .4, .2, .1]).sum(axis=1)
    print(name, values.mean(), values.std(), np.quantile(values, 0.05))
```

---

## K22. 抽样检验

### Name

批次次品率验收抽样。

### Description and core knowledge

通过抽检部分产品推断整批质量。抽样方案需明确样本量、接受判定数、生产方风险和使用方风险。

核心概念：

- 原假设与备择假设。
- 第一类错误、第二类错误。
- OC 曲线描述不同真实次品率下的接受概率。

### A specific applicable scene

2024B 问题 1 要求设计检测次数尽量少的零配件验收方案。

### Code

```python
import numpy as np
from scipy import stats

n, accept_if_bad = 30, 3
p_grid = np.linspace(0, 0.3, 100)

def oc_curve(n, c, p):
    return stats.binom.cdf(c, n, p)

print("P(accept | p=0.10):", oc_curve(n, accept_if_bad, 0.10))
print("P(accept | p=0.20):", oc_curve(n, accept_if_bad, 0.20))
```

---

## K23. 多阶段成本决策

### Name

检测、装配、拆解、售后与循环决策。

### Description and core knowledge

生产过程每个阶段的选择会影响后续状态和成本。可用决策树、递推期望成本或动态规划描述。

关键成本：

- 购买成本
- 检测成本
- 装配成本
- 拆解成本
- 售后损失

### A specific applicable scene

2024B 问题 2 和问题 3 是典型多阶段生产检测决策。

### Code

```python
p_bad = 0.1
cost_buy, cost_check = 4, 2
cost_assemble, cost_market_loss = 6, 10

cost_without_check = cost_buy + cost_assemble + p_bad*cost_market_loss
cost_with_check = cost_buy + cost_check/(1-p_bad) + cost_assemble

print(cost_without_check, cost_with_check)
```

---

# 四、数据、信号与反演知识卡片

## K24. 数据清洗与探索分析

### Name

缺失值、异常值、量纲和分布检查。

### Description and core knowledge

建模前先理解数据，不急着跑模型。要检查缺失、重复、异常、单位和变量类型，并保存清洗日志。

常用统计量：

- 均值、方差、分位数
- 相关系数
- 缺失率
- 空间或时间覆盖率

### A specific applicable scene

2023B 海水深度数据需要先检查坐标范围、单位换算、异常深度和缺失网格。

### Code

```python
import pandas as pd

df = pd.DataFrame({
    "x": [0, 1, 2, 3, 4],
    "y": [0, 1, 2, 3, 4],
    "depth": [100, 102, 101, 999, 103]
})

print(df.isna().sum())
print(df.describe())
print(df[df.depth > 200])
```

---

## K25. 插值与拟合

### Name

插值、最小二乘拟合与模型选择。

### Description and core knowledge

插值要求穿过数据点，拟合允许误差但追求整体规律。比赛里要明确数据是观测噪声还是精确值。

常用方法：

- 线性插值、样条插值
- 多项式拟合
- 指数、周期或机理模型拟合

### A specific applicable scene

2023B 中可由已有测深数据插值估计未测位置水深，进而计算覆盖宽度。

### Code

```python
import numpy as np
from scipy.interpolate import interp1d

x = np.array([0, 1, 2, 3])
y = np.array([0, 1, 0, 2])
f = interp1d(x, y, kind="cubic")

print(f([0.5, 1.5, 2.5]))
```

---

## K26. 峰谷检测与频谱分析

### Name

信号峰谷、频率间隔与谱分析。

### Description and core knowledge

干涉、振动、周期载荷等问题常通过峰谷位置或频谱提取周期信息。峰谷检测要先平滑，再检查阈值和距离。

常用工具：

- `find_peaks`
- FFT
- 带通滤波
- 频率间隔回归

### A specific applicable scene

2025B 红外干涉光谱中，可通过峰谷波数间隔估计光学厚度和薄膜厚度。

### Code

```python
import numpy as np
from scipy.signal import find_peaks

x = np.linspace(400, 1200, 2000)
y = np.sin(2*np.pi*0.02*x) + 0.05*np.random.randn(x.size)
peaks, props = find_peaks(y, distance=20, prominence=0.2)

print(x[peaks][:10])
print(np.diff(x[peaks])[:10])
```

---

## K27. 非线性最小二乘

### Name

非线性模型参数拟合。

### Description and core knowledge

当模型参数非线性进入时，最小化残差平方和。需要给初值和参数边界，并检查残差是否系统性偏移。

### A specific applicable scene

2022B 无人机定位中，可根据角度观测方程反推无人机坐标。

### Code

```python
import numpy as np
from scipy.optimize import least_squares

t = np.linspace(0, 5, 30)
y_obs = 3*np.sin(1.2*t + 0.2) + 0.05*np.random.randn(t.size)

def residual(theta):
    A, w, phi = theta
    return A*np.sin(w*t + phi) - y_obs

res = least_squares(residual, [1, 1, 0])
print(res.x)
```

---

## K28. 回归与方差分析

### Name

线性回归、交互项与因子效应。

### Description and core knowledge

回归用于量化因素对响应的影响。分类因子需编码，交互项描述一个因子的影响依赖另一个因子水平。

### A specific applicable scene

2021B 中催化剂组合、装料比、乙醇浓度和温度会影响转化率、选择性和收率。

### Code

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

rng = np.random.default_rng(1)
n = 30
df = pd.DataFrame({
    "temperature": np.tile([300, 325, 350], 10),
    "load": rng.uniform(0.5, 1.5, n)
})
df["yield_"] = 10 + 0.05*df.temperature + 0.5*df.load + rng.normal(0, 0.3, n)

model = smf.ols("yield_ ~ temperature + load + temperature:load", data=df).fit()
print(model.summary())
```

---

## K29. 实验设计

### Name

补充实验点、正交设计和最优设计。

### Description and core knowledge

当允许追加实验时，目标不是随机多测几个点，而是选择信息量最大的位置，降低参数不确定性或寻找最优条件。

常用设计：

- 全因子、部分因子
- 正交表
- 中心复合设计
- D-optimal
- 贝叶斯优化

### A specific applicable scene

2021B 问题 4 允许增加 5 次实验，应围绕当前模型不确定度大或收率可能更高的区域补点。

### Code

```python
import numpy as np

levels = [-1, 0, 1]
full_factorial = np.array(np.meshgrid(levels, levels, levels)).T.reshape(-1, 3)
star = np.vstack([np.eye(3)*1.5, -np.eye(3)*1.5])
center = np.zeros((3, 3))
ccd = np.vstack([full_factorial, star, center])

print("number of points:", len(ccd))
print(ccd)
```

---

## K30. 不确定性量化

### Name

误差传播、重抽样与扰动分析。

### Description and core knowledge

结果不仅是点估计，还应给出可信范围和结论稳定性。可通过解析误差传播、Bootstrap 或扰动实验实现。

### A specific applicable scene

2025B 厚度估计应分析峰谷检测误差、折射率误差和不同入射角结果的一致性。

### Code

```python
import numpy as np

rng = np.random.default_rng(2)
data = rng.normal(10, 1, size=100)

means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(3000)]
print(np.mean(data), np.percentile(means, [2.5, 97.5]))
```

---

# 五、数值计算算法卡片

## A31. 二分法与 Brent 求根

### Name

一维方程求根：二分法、Brent 方法。

### Description and core knowledge

二分法稳健但收敛慢；Brent 方法结合二分、割线和逆二次插值，通常是一维求根首选。要求区间两端函数值异号。

### A specific applicable scene

2024A 板凳龙中，可求某把手在等距螺线上对应的参数，使其与前一把手的距离等于板凳有效长度。

### Code

```python
from scipy.optimize import brentq

f = lambda x: x**3 - 2*x - 5
root = brentq(f, 1, 3)
print(root, f(root))
```

---

## A32. Newton 法与非线性方程组求解

### Name

Newton 迭代与非线性方程组求根。

### Description and core knowledge

Newton 法利用导数或雅可比矩阵快速收敛，但依赖初值。多变量问题可显式给雅可比，提高速度和稳定性。

### A specific applicable scene

2022B 无人机定位中，角度观测方程是非线性方程组，可由多个方向约束反解位置。

### Code

```python
import numpy as np
from scipy.optimize import root

def equations(x):
    return [x[0]**2 + x[1]**2 - 4,
            x[0] - x[1]]

def jacobian(x):
    return [[2*x[0], 2*x[1]],
            [1, -1]]

sol = root(equations, x0=[1.5, 1.0], jac=jacobian)
print(sol.x, sol.fun)
```

---

## A33. 弧长积分

### Name

参数曲线弧长与等弧长采样。

### Description and core knowledge

对参数曲线 `r(t)=(x(t),y(t))`，弧长微元为 `sqrt(x'(t)^2+y'(t)^2)dt`。速度恒定时，可按弧长反解时间参数。

### A specific applicable scene

2024A 板凳龙中，龙头前把手以 1 m/s 沿等距螺线运动，需要按弧长推进。

### Code

```python
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

pitch = 0.55

def curve(theta):
    r = pitch*theta/(2*np.pi)
    return np.array([r*np.cos(theta), r*np.sin(theta)])

def speed_factor(theta):
    r = pitch*theta/(2*np.pi)
    dr = pitch/(2*np.pi)
    return np.sqrt(r**2 + dr**2)

target_arc = 2.0
total_arc = lambda theta: quad(speed_factor, 0, theta)[0]
theta_target = brentq(lambda th: total_arc(th)-target_arc, 0.01, 10)
print(theta_target, curve(theta_target), total_arc(theta_target))
```

---

## A34. 数值微分

### Name

中心差分、梯度与敏感性估计。

### Description and core knowledge

中心差分比前向差分精度更高。步长不能任意小，否则舍入误差放大；常用相对步长 `eps=max(1e-6,1e-6*|x|)`。

### A specific applicable scene

2023A 定日镜场中，可对镜面位置或安装高度做小扰动，估计输出功率变化率。

### Code

```python
import numpy as np

def numerical_gradient(f, x, eps=1e-6):
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        dx = eps*np.maximum(1.0, abs(x[i]))
        xp = x.copy(); xp[i] += dx
        xm = x.copy(); xm[i] -= dx
        grad[i] = (f(xp)-f(xm))/(2*dx)
    return grad

print(numerical_gradient(lambda v: v[0]**2 + 3*v[0]*v[1], [1.0, 2.0]))
```

---

## A35. ODE 初值问题求解

### Name

`solve_ivp` 数值积分。

### Description and core knowledge

用于求解常微分方程初值问题。要设置 `t_eval`、最大步长和误差容限，并检查事件、守恒量和稳态。

### A specific applicable scene

2022A 波浪能中，可求浮子和振子的位移、速度，再计算 PTO 输出功率。

### Code

```python
import numpy as np
from scipy.integrate import solve_ivp

def rhs(t, y):
    return [y[1], -2*y[0]-0.5*y[1]+np.sin(t)]

sol = solve_ivp(rhs, (0, 20), [1, 0],
                t_eval=np.linspace(0, 20, 1000),
                rtol=1e-8, atol=1e-10, max_step=0.01)

print(sol.success, sol.message)
print(sol.y[0, -1], sol.y[1, -1])
```

---

## A36. 刚性 ODE 求解

### Name

刚性常微分方程与隐式方法。

### Description and core knowledge

当系统不同时间尺度差异很大时，显式方法需要极小步长，应改用 `Radau`、`BDF` 或 `LSODA`。传热、化学反应和多体接触常出现刚性。

### A specific applicable scene

2018A 多层服装传热中，薄层和界面会导致时间尺度差异，隐式求解更稳。

### Code

```python
import numpy as np
from scipy.integrate import solve_ivp

lambda1 = 1000.0

def stiff_rhs(t, y):
    return [-lambda1*y[0] + np.cos(t)]

sol = solve_ivp(stiff_rhs, (0, 1), [1.0], method="BDF",
                jac=lambda t, y: [[-lambda1]])
print(sol.y[0, -1])
```

---

## A37. 一维 PDE 有限差分

### Name

热传导方程显式有限差分。

### Description and core knowledge

将空间离散成网格，用差分近似二阶空间导数，再逐时间步推进。显式格式需满足稳定性条件；多层材料可在界面处变化参数。

### A specific applicable scene

2018A 可建立一维多层热传导模型，求解假人皮肤外侧温度随时间变化。

### Code

```python
import numpy as np

L, T = 0.01, 1.0
nx, nt = 100, 2000
alpha = 1e-7
dx = L/(nx-1)
dt = T/nt

if alpha*dt/dx**2 > 0.5:
    raise ValueError("unstable explicit scheme")

u = np.ones(nx)*300.0
u[0] = 375.0
u[-1] = 310.0

for _ in range(nt):
    u_new = u.copy()
    u_new[1:-1] = u[1:-1] + alpha*dt/dx**2*(u[2:]-2*u[1:-1]+u[:-2])
    u = u_new

print(u[-1])
```

---

## A38. 非线性最小二乘拟合

### Name

`least_squares` 参数拟合。

### Description and core knowledge

最小化残差平方和。要提供合理初值和边界；拟合后检查残差、参数协方差和不同数据子集的稳定性。

### A specific applicable scene

2025B 可用干涉模型拟合光谱反射率曲线，估计外延层厚度。

### Code

```python
import numpy as np
from scipy.optimize import least_squares

x = np.linspace(0, 10, 100)
y_obs = 2.0*np.sin(1.3*x + 0.2) + 0.08*np.random.default_rng(3).normal(size=x.size)

def residual(theta):
    A, w, phi = theta
    return A*np.sin(w*x+phi) - y_obs

res = least_squares(residual, x0=[1, 1, 0], bounds=([0, 0.1, -np.pi], [5, 3, np.pi]))
print(res.x, np.mean(res.fun**2))
```

---

# 六、几何与仿真算法卡片

## A39. 点到线段与点到平面距离

### Name

最近点与距离计算。

### Description and core knowledge

点到线段距离需要判断投影参数是否在 `[0,1]` 区间内；点到平面距离用点积除以法向量模长。

### A specific applicable scene

2024A 中可计算相邻板凳中心线之间的距离，判断是否发生碰撞。

### Code

```python
import numpy as np

def point_to_segment(p, a, b):
    ab = b-a
    t = np.clip(np.dot(p-a, ab)/np.dot(ab, ab), 0, 1)
    closest = a+t*ab
    return np.linalg.norm(p-closest), closest

def point_to_plane(p, p0, n):
    return abs(np.dot(p-p0, n))/np.linalg.norm(n)

print(point_to_segment(np.array([0.5,1.0]),
                       np.array([0,0]), np.array([1,0])))
print(point_to_plane(np.array([1,2,3]),
                     np.array([0,0,1]), np.array([0,0,2])))
```

---

## A40. 线段相交检测

### Name

二维线段相交。

### Description and core knowledge

用叉积判断两条线段是否互相跨越。必须单独处理共线、端点重合和平行情况。

### A specific applicable scene

2024A 板凳龙碰撞检测中，板凳可近似为有一定宽度的线段或矩形。

### Code

```python
import numpy as np

def orient(a, b, c):
    return np.cross(b-a, c-a)

def segments_intersect(a, b, c, d, eps=1e-12):
    o1, o2, o3, o4 = orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)

    def on_segment(a, b, p):
        return min(a[0],b[0])-eps <= p[0] <= max(a[0],b[0])+eps and \
               min(a[1],b[1])-eps <= p[1] <= max(a[1],b[1])+eps

    if ((o1>eps and o2<-eps) or (o1<-eps and o2>eps)) and \
       ((o3>eps and o4<-eps) or (o3<-eps and o4>eps)):
        return True
    if abs(o1) <= eps and on_segment(a,b,c): return True
    if abs(o2) <= eps and on_segment(a,b,d): return True
    if abs(o3) <= eps and on_segment(c,d,a): return True
    if abs(o4) <= eps and on_segment(c,d,b): return True
    return False

print(segments_intersect(np.array([0.,0.]), np.array([1.,1.]),
                         np.array([0.,1.]), np.array([1.,0.])))
```

---

## A41. 射线与球体求交

### Name

视线遮挡与射线求交。

### Description and core knowledge

将视线写成 `p=o+td, t>=0`，代入球方程得到二次方程。若判别式非负且解在有效时间窗内，则视线被球遮挡。

### A specific applicable scene

2025A 中烟幕云团为球体，可判断导弹到真目标的视线是否穿过云团。

### Code

```python
import numpy as np

def ray_sphere(o, d, c, r):
    d = d/np.linalg.norm(d)
    oc = o-c
    a = np.dot(d,d)
    b = 2*np.dot(oc,d)
    cc = np.dot(oc,oc)-r*r
    disc = b*b-4*a*cc
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    t1, t2 = (-b-sq)/(2*a), (-b+sq)/(2*a)
    return t1, t2

print(ray_sphere(np.array([0.,0.,0.]),
                 np.array([1.,0.,0.]),
                 np.array([5.,0.,0.]), 1.0))
```

---

## A42. 向量反射

### Name

镜面反射方向计算。

### Description and core knowledge

若 `d` 为入射方向，`n` 为单位法向量，则反射方向为 `d-2(d·n)n`。反射后应归一化。

### A specific applicable scene

2021A FAST 和 2023A 定日镜场都需要用反射定律追踪光线。

### Code

```python
import numpy as np

def reflect(d, n):
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)
    n = n/np.linalg.norm(n)
    r = d - 2*np.dot(d,n)*n
    return r/np.linalg.norm(r)

print(reflect(np.array([1.,-1.,0.]), np.array([0.,1.,0.])))
```

---

## A43. 旋转矩阵

### Name

二维旋转与三维绕轴旋转。

### Description and core knowledge

二维旋转矩阵作用于平面向量；三维旋转可用 Rodrigues 公式或若干基本旋转矩阵复合。注意角度单位是弧度。

### A specific applicable scene

2022B 无人机编队调整中，理想位置和角度观测都依赖平面旋转。

### Code

```python
import numpy as np

def rotation_2d(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s],[s,c]])

def rotation_axis(axis, theta):
    axis = np.asarray(axis, dtype=float)
    axis = axis/np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(theta)*K+(1-np.cos(theta))*(K@K)

print(rotation_2d(np.pi/2) @ np.array([1.,0.]))
print(rotation_axis([0,0,1], np.pi/2) @ np.array([1.,0.,0.]))
```

---

## A44. 光线追踪

### Name

批量光线发射与能量统计。

### Description and core knowledge

对太阳光锥、镜面反射或探测器接收进行采样，逐条计算路径是否命中目标。适合复杂几何中无法解析求积分的效率。

### A specific applicable scene

2023A 定日镜场中，可用锥形光线采样估计截断效率和阴影遮挡效率。

### Code

```python
import numpy as np

rng = np.random.default_rng(7)
n = 100000
rho = np.sqrt(rng.random(n))*0.05
phi = rng.random(n)*2*np.pi

directions = np.column_stack([
    rho*np.cos(phi),
    rho*np.sin(phi),
    np.sqrt(1-rho**2)
])

# 假设接收器为 z=80 平面上的圆盘
hit_radius = np.hypot(directions[:,0]*50, directions[:,1]*50)
print("estimated acceptance:", np.mean(hit_radius < 3.5))
```

---

## A45. 蒙特卡洛积分

### Name

随机采样估计积分与复杂面积。

### Description and core knowledge

在定义域上均匀采样，用函数平均值乘以区域测度估计积分。误差通常按 `1/sqrt(n)` 下降，可重复多次估计波动范围。

### A specific applicable scene

2023A 中可用蒙特卡洛估计镜面反射能量被集热器截获的比例。

### Code

```python
import numpy as np

rng = np.random.default_rng(8)
n = 200000
x = rng.uniform(-1, 1, n)
y = rng.uniform(-1, 1, n)
inside = x**2+y**2 <= 1

area_estimate = 4*inside.mean()
se = 4*np.std(inside, ddof=1)/np.sqrt(n)
print(area_estimate, se)
```

---

## A46. 离散事件仿真

### Name

事件优先队列与调度规则仿真。

### Description and core knowledge

按下一事件时间推进系统，可评估调度规则、等待时间、资源利用率和故障影响。比固定步长更高效且逻辑更清晰。

### A specific applicable scene

2018B RGV 调度中，可仿真不同上下料和移动策略的 8 小时产量。

### Code

```python
import heapq

time, produced = 0.0, 0
events = [(0.0, "start")]

def add_event(delay, name):
    heapq.heappush(events, (time+delay, name))

while time < 8*3600 and events:
    time, event = heapq.heappop(events)
    if event in ("start", "unload_load_finish"):
        add_event(545, "machine_finish")
    elif event == "machine_finish":
        produced += 1
        add_event(28, "unload_load_finish")

print(time, produced)
```

---

# 七、优化算法卡片

## A47. 线性规划

### Name

LP 求解。

### Description and core knowledge

目标与约束均为线性时，可直接调用单纯形法或内点法。建模重点是变量、约束和目标的一致性，注意 `min` 与 `max` 的符号转换。

### A specific applicable scene

2024B 生产检测问题中，若将检测和拆解决策简化为线性期望成本模型，可用 LP 求最优策略。

### Code

```python
from scipy.optimize import linprog

c = [3, 5]          # minimize 3x+5y
A_ub = [[1, 1]]
b_ub = [10]
bounds = [(0, None), (0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(res.success, res.x, res.fun)
```

---

## A48. 非线性约束优化 SLSQP

### Name

序列二次规划局部优化。

### Description and core knowledge

适用于少维连续参数、目标可微或数值可微、存在等式或不等式约束的问题。应多初值运行并检查 KKT 条件和约束余量。

### A specific applicable scene

2020A 炉温曲线中，可优化各温区温度和传送速度，使超过 217 摄氏度区域面积最小并满足工艺约束。

### Code

```python
from scipy.optimize import minimize

def objective(x):
    T1, T2, v = x
    return (T1-240)**2 + (T2-250)**2 + 0.1*(v-80)**2

constraints = [
    {"type": "ineq", "fun": lambda x: x[0]-175},
    {"type": "ineq", "fun": lambda x: 260-x[0]},
    {"type": "ineq", "fun": lambda x: x[2]-65},
    {"type": "ineq", "fun": lambda x: 100-x[2]},
]

res = minimize(objective, x0=[220, 240, 80], method="SLSQP", constraints=constraints)
print(res.success, res.x, res.fun)
```

---

## A49. 差分进化全局优化

### Name

差分进化算法。

### Description and core knowledge

差分进化通过种群、变异、交叉和选择搜索全局最优，适合多峰、不可微或黑箱目标。代价是评价次数多，必须控制目标函数耗时。

### A specific applicable scene

2025A 烟幕弹策略中，飞行方向、速度、投放点和起爆时间构成多峰优化问题。

### Code

```python
import numpy as np
from scipy.optimize import differential_evolution, minimize

def black_box(x):
    return np.sin(3*x[0]) + np.cos(2*x[1]) + 0.05*x[0]**2

bounds = [(-5, 5), (-5, 5)]
global_res = differential_evolution(black_box, bounds, seed=1, tol=1e-8)
local_res = minimize(black_box, global_res.x, method="Nelder-Mead")
print(global_res.x, global_res.fun)
print(local_res.x, local_res.fun)
```

---

## A50. 动态规划算法

### Name

资源与路径动态规划。

### Description and core knowledge

适用于无后效性的多阶段决策。状态要包含位置、时间、资源等影响未来可行性的信息；用字典或数组保存阶段值函数。

### A specific applicable scene

2020B 单人已知天气情形下，可对资金、水量、食物量和位置做状态搜索。

### Code

```python
from collections import defaultdict

max_water, max_food = 5, 5
values = defaultdict(lambda: -1e18)
next_values = defaultdict(lambda: -1e18)

for w in range(max_water+1):
    for f in range(max_food+1):
        values[(0, w, f)] = 100.0  # 到达终点的剩余资金

for t in range(1, 8):
    for pos in [0, 1, 2, 3]:
        for w in range(max_water+1):
            for f in range(max_food+1):
                # 示例转移：走一步或停留
                candidates = []
                for npos, move_cost in [(pos, 0), (min(pos+1, 3), 1)]:
                    nw, nf = w-move_cost, f-move_cost
                    if nw >= 0 and nf >= 0:
                        candidates.append(values[(npos, nw, nf)])
                next_values[(pos,w,f)] = max(candidates)
    values, next_values = next_values, defaultdict(lambda: -1e18)

print(values[(0, 5, 5)])
```

---

## A51. Dijkstra 与 A* 最短路

### Name

最短路搜索。

### Description and core knowledge

Dijkstra 保证非负边权最短路；A* 加入到目标点的启发估计以减少搜索状态。若启发函数不高估，仍可得到最优路径。

### A specific applicable scene

2020B 穿越沙漠中，在固定天气序列下可先构造状态图再求最优策略。

### Code

```python
import heapq

def shortest_path(graph, source, target):
    dist = {source: 0.0}
    pq = [(0.0, source)]
    prev = {}
    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph.get(u, []):
            nd = d+w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    path, node = [], target
    while node in prev or node == source:
        path.append(node)
        if node == source:
            break
        node = prev[node]
    return dist[target], path[::-1]

graph = {0: [(1,2), (2,5)], 1: [(2,1), (3,4)], 2: [(3,1)], 3: []}
print(shortest_path(graph, 0, 3))
```

---

## A52. MILP 与 CP-SAT

### Name

混合整数规划和约束规划。

### Description and core knowledge

MILP 适合线性目标和约束加整数变量；CP-SAT 更适合含复杂时序、排列和资源的组合约束。求解前必须显式写变量、目标、约束和索引。

### A specific applicable scene

2018B RGV 调度中，可将机床分配、加工顺序、上下料时间写成整数规划或约束规划。

### Code

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()
x = [model.NewBoolVar(f"x{i}") for i in range(4)]
y = model.NewIntVar(0, 20, "y")

model.Add(sum(x) >= 2)
model.Add(y == sum(3*x[i] for i in range(4)))
model.Maximize(y)

solver = cp_model.CpSolver()
status = solver.Solve(model)
print(solver.StatusName(status), [solver.Value(v) for v in x], solver.Value(y))
```

---

## A53. 贪心与滚动时域

### Name

在线决策与滚动优化。

### Description and core knowledge

当全局最优难以求解或信息逐步揭示时，每轮只优化有限时间窗或选择当前边际收益最大的动作。必须用仿真验证规则，而非只凭直觉。

### A specific applicable scene

2018B 故障情形下，可根据当前 RGV 位置和机床完成时间滚动选择下一任务。

### Code

```python
positions = [1, 3, 5, 7]
remaining = [560, 120, 560, 30]
rgv_position = 4

def move_cost(source, target):
    return abs(source-target)*20

while remaining:
    # 优先完成时间短且移动代价低的机床
    scores = [(remaining[i]/(1+move_cost(rgv_position, positions[i])), i)
              for i in range(len(positions))]
    _, chosen = min(scores)
    print("serve machine", chosen+1)
    rgv_position = positions[chosen]
    positions.pop(chosen)
    remaining.pop(chosen)
```

---

## A54. 多目标优化与 Pareto 前沿

### Name

NSGA-II 与非支配排序。

### Description and core knowledge

多目标问题通常没有单一最优解，而是一组互不支配的 Pareto 解。可通过加权、约束法或进化算法生成前沿。

### A specific applicable scene

2023A 定日镜场中，同时关注额定功率、总镜面面积、单位面积输出热功率和光学效率。

### Code

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize

problem = get_problem("zdt2")
algorithm = NSGA2(pop_size=100)
res = minimize(problem, algorithm, ("n_gen", 100), seed=1, verbose=False)

print(res.F[:5])
```

---

## A55. 贝叶斯优化

### Name

昂贵黑箱函数全局优化与实验补点。

### Description and core knowledge

贝叶斯优化用代理模型拟合目标，并通过采集函数在“可能更优”和“不确定较大”的区域之间权衡，适合实验次数有限的问题。

### A specific applicable scene

2021B 追加 5 次实验时，可在高收率潜力和模型不确定度大的催化剂条件区域补点。

### Code

```python
from skopt import gp_minimize

def objective(x):
    T, load = x
    return -((T-340)**2 + 100*(load-1.2)**2)

res = gp_minimize(
    objective,
    dimensions=[(300.0, 380.0), (0.5, 2.0)],
    n_calls=20,
    n_random_starts=5,
    random_state=1
)

print(res.x, -res.fun)
```

---

# 八、统计与机器学习算法卡片

## A56. 二项检验与置信区间

### Name

次品率比例推断。

### Description and core knowledge

抽到的不合格品数服从二项分布。可做单侧或双侧检验，并给出比例置信区间。比赛时必须说明原假设和备择假设。

### A specific applicable scene

2024B 中根据样本不合格数判断批次次品率是否超过 10%。

### Code

```python
from scipy import stats

n, k, p0 = 100, 12, 0.10
result = stats.binomtest(k, n, p0, alternative="greater")
ci = result.proportion_ci(0.95)

print(result.pvalue, ci)
```

---

## A57. OC 曲线与 SPRT

### Name

抽检特征曲线与序贯概率比检验。

### Description and core knowledge

OC 曲线描述给定抽样方案在不同真实次品率下的接受概率。SPRT 每抽一个样本就判断接受、拒收或继续，常能减少平均检测次数。

### A specific applicable scene

2024B 要求检测次数尽可能少，可用 SPRT 设计动态验收规则。

### Code

```python
import numpy as np
from scipy import stats

def oc_curve(n, c, p_grid):
    return stats.binom.cdf(c, n, p_grid)

def sprt(p0, p1, alpha=0.05, beta=0.10, max_n=100):
    A, B = (1-beta)/alpha, beta/(1-alpha)
    k = 0
    rng = np.random.default_rng(1)
    for n in range(1, max_n+1):
        k += rng.binomial(1, p1)
        likelihood_ratio = ((p1/p0)**k) * (((1-p1)/(1-p0))**(n-k))
        if likelihood_ratio >= A:
            return n, k, "reject H0"
        if likelihood_ratio <= B:
            return n, k, "accept H0"
    return max_n, k, "continue"

print(oc_curve(30, 3, np.array([.05,.1,.2])))
print(sprt(0.10, 0.20))
```

---

## A58. 线性与多项式回归

### Name

最小二乘回归与交互模型。

### Description and core knowledge

回归用于解释和预测。要检查拟合优度、残差正态性、多重共线性和外推风险。多项式阶数不宜盲目提高。

### A specific applicable scene

2021B 可建立温度、装料比与 C4 烯烃收率之间的响应面模型。

### Code

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = np.random.default_rng(1).uniform(-1, 1, (100, 2))
y = 2*X[:,0] - X[:,1] + 0.5*X[:,0]*X[:,1] + np.random.normal(0, .05, 100)

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
).fit(X, y)

print(model.score(X, y))
print(model.named_steps["linearregression"].coef_)
```

---

## A59. 随机森林与梯度提升

### Name

树模型非线性建模。

### Description and core knowledge

树模型能捕捉非线性和交互效应，对单调变换不敏感，适合探索性分析和特征重要性排序。需用交叉验证防止过拟合。

### A specific applicable scene

2017B 可分析任务位置、定价、会员密度与完成情况的关系。

### Code

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(2)
X = rng.normal(size=(300, 4))
y = (X[:,0] + 0.5*X[:,1]**2 + rng.normal(0, .1, 300) > 0).astype(int)

model = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=1)
print(cross_val_score(model, X, y, cv=5).mean())

model.fit(X, y)
print(dict(zip([f"x{i}" for i in range(4)], model.feature_importances_)))
```

---

## A60. 聚类

### Name

K-means 与 DBSCAN。

### Description and core knowledge

K-means 适合凸形簇，需要预设簇数；DBSCAN 能识别任意形状和噪声点，但参数敏感。聚类结果必须结合业务含义解释。

### A specific applicable scene

2017B 可将地理集中、竞争激烈的任务打包发布。

### Code

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

rng = np.random.default_rng(3)
X = np.vstack([
    rng.normal([0,0], .2, (100,2)),
    rng.normal([3,3], .2, (100,2)),
    rng.normal([0,4], .2, (100,2))
])

km = KMeans(n_clusters=3, n_init=20, random_state=1).fit(X)
db = DBSCAN(eps=.5, min_samples=5).fit(X)

print(km.labels_[:10])
print(db.labels_[:10])
```

---

## A61. 主成分分析与特征筛选

### Name

PCA、方差解释与特征重要性。

### Description and core knowledge

PCA 用于降维和共线性诊断，但主成分的解释性可能下降。特征筛选应结合模型重要性、稳定性和业务意义。

### A specific applicable scene

2017B 任务定价分析中可先压缩空间、价格、会员特征，再解释未完成原因。

### Code

```python
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(4)
x1 = rng.normal(size=(200,1))
X = np.hstack([x1, 2*x1+0.05*rng.normal(size=(200,1)), rng.normal(size=(200,3))])

pca = PCA()
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.components_[0])
```

---

## A62. Bootstrap

### Name

重抽样置信区间。

### Description and core knowledge

Bootstrap 通过有放回重抽样估计统计量分布，适合难以解析推导标准误的场景。对强时间序列依赖需用块 Bootstrap。

### A specific applicable scene

2025B 可用峰谷间隔重抽样估计厚度不确定度。

### Code

```python
import numpy as np

rng = np.random.default_rng(5)
data = rng.normal(10, 1, 200)
statistics = []

for _ in range(5000):
    sample = rng.choice(data, size=len(data), replace=True)
    statistics.append(np.median(sample))

print(np.percentile(statistics, [2.5, 50, 97.5]))
```

---

## A63. 贝叶斯更新

### Name

Beta-Binomial 次品率更新。

### Description and core knowledge

用先验分布表达检测前认知，用似然函数吸收样本信息，得到后验分布。次品率常用 Beta 先验和二项似然。

### A specific applicable scene

2024B 问题 4 中，抽检得到的不确定次品率可更新后用于生产决策。

### Code

```python
import numpy as np
from scipy import stats

alpha_prior, beta_prior = 1.0, 9.0
n, k = 50, 4
alpha_post = alpha_prior+k
beta_post = beta_prior+n-k

posterior = stats.beta(alpha_post, beta_post)
print(posterior.mean(), posterior.interval(0.90))
```

---

---
