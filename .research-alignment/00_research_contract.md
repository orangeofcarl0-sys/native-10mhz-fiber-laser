# 研究与发布范围

主要问题：固定 20.42 m 腔长、两个 OC/CNT 顺序、指定 OC/GDD/pump 网格下是否存在稳定 0.1–0.5 nJ 单脉冲候选。

此次发布只整理已经运行的物理模型和复现流程，不扩大仿真结论。输出包括便携 CPU/GPU 代码、有效参数、离线报告、结果摘要与验证脚本。大体积原始 MAT 留本地。

对照为同参数/同初态双拓扑，以及单案例/批量算法、粗/细网格、短期/长程延续。证据需同时覆盖数值边界、波形、能量、反转及扰动响应。未解析不得判为物理不可行；短时命中不得判为稳定锁模。

P1/P2 增补合同：检验共享EDF多窗口扰动及数值等价性能改造；新增证据见07_shared_gain_and_gpu.md，不将短时扰动下降升级为稳定性认证。

GPU增补合同：以801fd03为基线验证频谱传递、CNT前缀与EDF融合的FP64数值等价性及实测收益。证据见08_spectral_gpu.md；本轮不重画物理可行区、不升级单脉冲稳定性结论。

完整重算增补：现已完成既定20组/1280轨迹，主地图更新为新GPU实算；证据见09_full_scan.md。没有新增稳定锁模认证结论。

机制复核增补：本轮解释已运行数据，区分CNT调制、慢增益、数值与可观测量限制；见10_mechanism_review.md。不运行新的大规模扫描，不把相关性提升为根因证明。

增益年龄与真实时间续算：见11_gain_age.md。逐单元条件松弛系数不代替耦合敏感度；本轮先验证g08_c53，不沿未收敛末态追踪稳定支路。

周期一自洽求解试验：见12_steady_state.md。固定物理参数与离散网格，检验去除增益残差小因子后的Newton–Krylov求解；多初猜失败不证明无解，未通过稳态和数值检查不启动Floquet认证。

块预条件对照：见13_block_preconditioner.md。同一保存态、模板、缩放、网格、相位初估及迭代预算，只切换预条件；检验真实线性残差和总调用成本，再判断是否具备稳态验证入口。

Newton全局化：见14_hookstep.md。先重放并独立验证第8–11步导数、规范与窗口；采用同一Arnoldi/右预条件对照线搜索与实际缩放状态信赖域hookstep，规范重设作为单独因素，不能用子空间驻点证明全空间无根。

## Hookstep末态窗口与软方向续验（2026-09-19）
固定上一轮末态、物理参数、单位和规范。先在dt不变的2.048 ns窗口计算全向量/中心残差差，随后在相同局域扰动子空间比较窄窗和宽窗投影雅可比。投影谱不能代替全空间奇异谱或Floquet。若全向量差>0.1%，在宽窗继续最多25步；保持真实宽窗残差/Jv，只将预条件逆限制在原1.024 ns局域Fourier子空间，外部采用-I。保留半径0.0125，最近5接受步下降<2%且至少4个rho>0.5时停止并诊断，不推断无根。

## 参数增广验证（2026-09-19）
当前2.048 ns末态作为唯一入口。先重建3093维局域投影雅可比并保存最弱20个左右模、残差系数、Newton系数；同时通过完整输出列计算原始merit的子空间梯度，不能只用Jc^T Rc判断全系统驻点。四个参数为pump、GDD、OC、CNT Psat=Esat/tau，固定tau；GDD通过SMF/NDF长度分配改变并保持20.42 m。尺度分别5mW、0.02ps²、0.05、4W；中心差分q步长1e-3及5e-4核对。比较归一化耦合与带边框矩阵条件，不按不同单位的裸导数排序。参数释放仅在±2尺度内，先作局部求根试验，不把未认证末态当作分支种子，也不把有耦合当作fold证明。

2026-09-19: leakage controls, fixed-pump profiles and positive 0.9 nJ energy closure; see 17_leakage_energy.md.

2026-09-19: fixed-pump linear support benchmark and narrow warm-start search; energy stays outside merit. See18_fixed_pump_linear.md.

Current continuation: three fixed-state merit/curvature/coarse-difference controls; see19_merit_geometry.md. No extra nonlinear search without diagnostic justification.

Current work: four fixed-state Cauchy/span controls; no long-run continuation; see20_cauchy_span.md.

Current experiment: 27.5 mW paired 40-step original/augmented outer loops from identical published down_275. Shared C preconditioner, centered Jv, <1% linear gate, adaptive trust controller. Fresh full-output restricted gradient on rebuild; cheap projected gradient checked for descent between rebuilds. No pump scan or physical-root claim without residual closure.

Step-relevant gate audit and conditional B continuation: see22_step_gate.md.

Current: single-state descent-source controls and thresholded continuation; see23_descent_sources.md.

Current: frozen S3 endpoint active-history audit only; see24_active_history.md. No continuation.
