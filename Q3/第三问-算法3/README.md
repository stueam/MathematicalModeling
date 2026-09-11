# 算法3：贝叶斯定位＋共享测向＋动态开放 TSP

仓库发布路径为 `Q3/第三问-算法3`；历史实验材料的保留范围见 [归档说明](results/README.md)。

**当前默认：`bayes-fast`（V6＋等价计算缓存）。新留出900～939共40图，545/545源全清，汇总216.01虚拟秒/源；同图V5为218.37秒/源，改善1.08%。170秒/源目标尚未达到。**

**最新官方问题3演练10轮已完成：141/141源全清，汇总211.64秒/源，按局平均216.58秒/源，客户端平均3.17秒；1/10局低于170秒/源。** 同配置连续10轮，没有使用正式测试。详见 [bayes-fast官方演练十轮报告](bayes-fast官方演练十轮报告.md)。与历史V5演练地图不同，不能直接计算新旧提升率。

本轮缓存版与V6在全部40图的动作、反馈和决策评分完全一致，实际循环均值由6.25秒降至3.44秒（减少44.91%；批次并发负载下计时）。另有24组极值/相关误差压力场景，缓存版与V6全部完成且虚拟费用逐局相同。详情见 [本轮提速与负实验报告](本轮提速与负实验报告.md)。

本地及演练脚本默认已接入`bayes-fast`，本地验证后已按用户授权完成上述官方演练。`bayes-sector-fast`保留V5行为并使用相同缓存；`bayes-sector`和`bayes-v6`保留原计算实现用于对照。

依据用户提供的 [算法3指导原文](第三问_机器狗搜索定位清除算法.md) 实现。V5阶段的实现、消融和结果见 [V5三项优化报告](算法3_V5三项优化报告.md)。参考来源见 [B2审查](GitHub_B2参考审查.md)。

## 已测结论

此前按用户要求撤回V7/V8改革，恢复到 `ee44bc0` 的V6快照。本轮从该快照继续开发，未恢复那些未获收益的改革。实验代码保存在Git历史，原始结果保留；[回滚记录与V7报告备份](results/20260911-190053-681279/)。

新增 [V6路线审查与150秒目标](V6路线审查与150秒目标.md)：十轮事后清除点TSP及20m清除余量推导表明，**保留现有非移动费用时，任务下界仍约185.27秒/源**。下一版需同时改进共用路线和逐频道覆盖，不能只重排TSP。该审计没有运行新策略，不能作为新版成绩。

此前的 [十轮演练复盘与V6本地试验](十轮演练复盘与V6本地试验.md)：定位了测站优化的微小数值越界拒绝及连续低概率clear问题；当时新留出仅改善约0.96%，仍不接近150s/源。**下列历史官方演练成绩均属于V5；V6缓存版的新成绩见页首。**

**官方软件问题3演练累计10轮：119/119源全部清除，汇总243.27秒/源，按局平均247.66秒/源。只使用演练，没有使用正式测试。** 实际客户端时间平均2.94秒，1/10局低于200秒/源。详见 [十轮演练报告](V5官方演练十轮报告.md)，[前三轮记录](V5官方模拟器演练报告.md)保留作历史记录。

V5历史实现版本 `bayes_tsp_v5_sector_services`，对照配置 `bayes-sector`：

- 开发地图0～9：10/10完成，平均 **2950.75虚拟秒、208.34 s/源**，同图V4为3436.36秒。
- 新留出地图400～419：20/20完成，平均 **2910.81虚拟秒、10.87km、224.40 s/源**，同图V4为3436.18秒。
- 平均measure由201.55降至111.55；主要收益是减少冗余扫描。相对V4路程略增约73m，不能声称每个分项都下降。
- 压力配对24/24完成，平均比V4快9.95%，其中5个配置退步。
- 加上3张诊断图，默认共57个指定配置完成；48项单元/回归测试通过。
- 留出批次单局实际循环均值约6.95秒，V4约7.30秒，均受16进程批次负载影响；与虚拟任务耗时分开报告。

普通留出集约200 s/源目标尚未达到。历史记录见 [V4报告](算法3_V4联合路线与移动优化报告.md)、[V3报告](算法3_V3路线调度改进报告.md)、[V2说明](算法3_贝叶斯与动态TSP实现说明.md)、[V2结果](算法3_实验报告.md)。

种子10～12同图诊断：算法2既有默认记录平均3212.37秒，V4为3306.99秒，V5降至2886.24秒。双方均完整清除；只有三张诊断图，不外推固定总体优势。

## 实现内容

1. 原点扫描必要未知频道，建立第一批观测。
2. 复用第二问的面积先验与固定未知 R 后验：对 R 解析积分，在位置网格上确定性求积。
3. 把定位/清除节点和仍需覆盖的测站放在同一开放路线中，比较“先处理源”与“先顺路扫描”。
4. 12个节点以内精确求解固定代表点开放TSP；更大规模用多起点、2-opt和重插入。每个候选的剩余路线从其完成位置重新优化。
5. 比较近距离侧移、沿路前进、距离估计目标 100/150/200 m 的途中补测和尝试清除。
6. 按频道形成默认7个扇区覆盖任务；清除点能承担完整任务时才安排对应频道扫描，减少V4的广泛扫描。
7. 用SLSQP在完整任务顶点覆盖约束下移动测站，再优化开放路线；不合格结果回退原合法点。
8. 算法三专用CoupledBelief使用同源固定R的阳性/阴性半平面收紧，保留事务性、舍入与幂等规则。

决策器中没有随机数生成、假想世界采样、rollout、MCTS 或 POMCP。预测反馈使用确定性角度分档积分。随机地图仅用于本地评价。

## 安装与运行

Python 3.10+，所有下列命令在本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python run.py local --seed 0
python -m pytest -q

# 本轮默认、V6原实现与V5同图比较
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py benchmark --seed 900 --rounds 40 --compare bayes-sector bayes-v6 bayes-fast --workers 16

# 16 个进程跑独立本地案例；每个数值库单线程
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py benchmark --seed 0 --rounds 10 --compare baseline-ring bayes-joint bayes-radius bayes-task-scan bayes-mobile bayes-sector-fixed bayes-sector --workers 16

# 四种空间场景 × 三种误差机制 × 两种 N/R 极值配置，共 24 局
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py validate --validation-compare bayes-joint bayes-sector --workers 16

# 留出地图配对比较
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run.py benchmark --seed 400 --rounds 20 --compare baseline-ring bayes-joint bayes-sector --workers 16

# 读取已有结果，生成新的分析目录
python analyze.py results/某次实验目录

# 公开动作日志的移动分段与静态路线差距审计
python audit_routes.py results/某次实验目录 --policy bayes-joint
```

本机器当前的临时Shapely可通过 `PYTHONPATH=/tmp/mathmodel_q4_deps` 使用；测试用 `PYTHONPATH=/tmp/mathmodel_q4_deps:.`。源码没有硬编码该路径。

### 策略与参数

| 选项 | 含义 |
|---|---|
| `bayes-fast` | 当前默认：V6＋同决策候选缓存及预计算路线边长 |
| `bayes-sector-fast` | V5等价计算加速，保留V5动作与评分 |
| `bayes-station-fix` | 实验：额外测站细化与严格可行回缩 |
| `bayes-recovery` | 实验：失败clear后优先有交会角的新测量 |
| `bayes-v6` | 两项实验组合，7扇区；仅本地验证 |
| `bayes-v6-six` / `bayes-v6-eight` | 实验组合的6/8扇区对照 |
| `bayes-sector` | V5原实现对照，三项优化全部启用 |
| `bayes-sector-fixed` | 任务级扫描＋R收紧，关闭移动测站 |
| `bayes-mobile` | 任务级扫描＋移动测站，关闭R收紧 |
| `bayes-task-scan` | 仅使用新任务级扫描，固定测站、原几何 |
| `bayes-radius` | 仅在V4上加固定R几何收紧 |
| `bayes-joint` | V4原默认，保留对照 |
| `bayes-tsp4` | 只增强固定点TSP与条件后缀，不加联合覆盖/服务 |
| `bayes-service` | V3路线加实际清除点扫描服务，不加联合节点 |
| `bayes-joint-probe` | V4加“顺路测量后继续整条路线”候选，实验对照 |
| `bayes-reviewed` | 上项再加长距离小收益改道的细求积复核，实验对照 |
| `bayes-route-scan` | V3原默认，可复现历史对照 |
| `bayes-efficient` | V3组合实验：另外开启稳定/待测分层、20秒改道门槛和待测插入绕行估计；不是当前推荐 |
| `bayes-stable` | V3分层实验，使用旧机会扫描规则 |
| `bayes-strict-stable` | 复现V3首轮严格稳定优先，不允许附近待测目标竞争，也不加待测绕行代理 |
| `bayes-no-cover` | V3组合版关闭完整补盲路线，沿用V2补盲规则 |
| `bayes-tsp` | 原V2完整算法3，可重复的历史对照 |
| `bayes-nearest` | V2最近邻消融 |
| `bayes-no-shared` | V2关闭共享测量消融 |
| `baseline` | 算法1的快速移动成本策略，固定 1500 m 环 |
| `baseline-ring` | 同一快速基线，使用 `--ring`，默认 1200 m，隔离环半径差异 |
| `--resolution 24` | 每次旋转包围矩形划分为 24×24 单元，再裁剪、面积求积 |
| `--bearing-bin 1` | 预测角度档宽（度）；真实观测更新始终按 0.01° 返回档计算似然 |
| `--p0 .65` | 频道存在性的工作先验，之后结合观测并按总数 10～16 条件化 |
| `--future-stops 3` | 新版机会测量的近期路线预测点数量 |
| `--exact-limit 12` | 固定坐标TSP精确DP的节点上限；更大节点集用确定性局部优化 |
| `--sectors 7` | V5角扇区数量，允许6～12 |
| `--station-sweeps 2` | V5每次规划的“调站点—重排路线”轮数，允许0～4 |
| `--stable-radius 100` | 分层实验的90%离散质量半径阈值，只影响路线，不证明清除 |
| `--switch-gain 20` | 分层实验的改道迟滞，秒；推荐配置不启用 |
| `--no-speculative-clear` | 只使用保守几何保证清除或有限覆盖清除 |
| `--real-limit 1200` | 本地实际运行时限；正式接口以 `/enter` 返回值为准 |

`--workers` 只并行独立的本地案例；单次决策是确定性的、单进程执行。官方 HTTP 始终串行。

## 代码结构与复用依赖

| 文件 | 内容 |
|---|---|
| `bayes_tsp/posterior.py` | 固定 R 解析边际化、裁剪单元求积、总数条件化、反馈分档积分、清除点搜索 |
| `bayes_tsp/routing.py` | 最近邻初始化、动态 cheapest insertion、开放 2-opt |
| `bayes_tsp/policy.py` | V2对照、复用的局部候选与完成后备 |
| `bayes_tsp/efficient.py` | V3调度、默认策略和稳定目标分层实验 |
| `bayes_tsp/scheduling.py` | 同一固定R的路线测量价值、完整几何覆盖子集与开放路线DP |
| `bayes_tsp/joint.py` | V4联合任务、清除服务、条件覆盖、沿路探测和复核 |
| `bayes_tsp/open_routes.py` | 精确开放TSP、较大路线局部优化和几何后缀缓存 |
| `bayes_tsp/coupling.py` | 算法三专用固定R半平面及事务几何状态 |
| `bayes_tsp/sectors.py` | V5任务级扫描、移动测站与各项对照 |
| `bayes_tsp/refinement.py` / `repair.py` | 根据演练诊断实现的本地修复候选，保留V5行为 |
| `bayes_tsp/cached.py` | 等价路线边长复用、单次决策候选及局部评分缓存 |
| `audit_practice.py` | 公开演练历史回放、费用分段和站点数值诊断 |
| `bayes_tsp/shared.py` | 加载兄弟目录中已验证的几何、协议和有限清除代码 |
| `run.py` | 本地/配对/压力实验、显式官方连接、日志与源码哈希 |
| `analyze.py` | 完整配对集合的统计和决策诊断 |
| `audit_routes.py` | 公开动作日志的末尾费用、移动类别与静态TSP差距 |
| `tests/test_bayes_tsp.py` | 解析概率对照、舍入、路线、零积分回退和有限完成测试 |
| `tests/test_scheduling.py` | 同R接收事件、开放路线插入、覆盖DP穷举对照、微小盲区、分层切换与无随机规划 |
| `tests/test_joint_routes.py` | 条件TSP穷举、缓存隔离、服务兑现、预测覆盖隔离、无随机完整任务 |
| `tests/test_sectors.py` | R耦合、事务、扇区接缝、约束站点优化、任务扫描及完整运行 |

**交付依赖：本目录需要与 `../第三问-算法1/q3/` 一起保留。** 复用其 `core.py`、`policy.py`、`simulator.py`、`client.py`；比较基线时另加载 `movement.py` 及其依赖。私有包命名避免与其他 `q3` 包冲突，不修改兄弟目录。它不是可单独复制运行的独立目录。

每批输出新建 `results/时间戳/`，保存参数、实际使用源码 SHA-256、动作、完整候选评分、逐局 summary 和 `batch.json`。批次中断单独标记，配对统计不会把不完整批次作为完整成绩。

## 正式接口

连接官方软件时优先使用**仅演练入口**：在问题3演练列表或已结束演练页面，明确授权后运行：

```bash
python practice_windows.py --connect --rounds 3
```

该入口通过Windows UI核验问题3演练标题和案例身份，拒绝正式/含糊界面。保留实际截止时间、同ID同内容重试、合法反馈只应用一次、clear不切换接收频道，以及不确定请求结果后的停止逻辑。通用 `run.py official` 本身没有GUI模式防护，不用于此次仅演练操作。
