# Native 10 MHz fiber laser — dual-topology simulation

20.42 m 固定总腔长的 EDF + SMF + NDF + OC + CNT 矢量腔映射模型。比较 **OC→CNT** 与 **CNT→OC**，扫描输出耦合、净色散和泵浦，寻找 **0.1–0.5 nJ** 单脉冲候选区。Python/NumPy CPU 为基准，CuPy GPU 为可选加速；不需要 MATLAB。

**新增 P0 自适应求解器：** `attractor_search.py` 提供泵浦相关初态、多种子/双向延续、受限慢增益搜索、越界回退重算、无源自适应步长和 EDF 网格自适应，以及复场/偏振/反转周期检查。详见 [新算法与使用方法](docs/adaptive_solver.md)。新路径目前是 CPU 参考实现；旧 CuPy 地图及以下历史结论未被新算法重新计算替换。

**当前结论：尚未确认稳定单脉冲可行区。** 640 个不同物理参数点分别用粗、细网格计算，细网格有 8 个短时目标能量单峰状态，严格稳定初筛通过数为 0。376 个细网格点触发数值边界，属于未解析，不能认定为物理不可行。这里没有实物实验数据。

下载 [完整离线 HTML 报告](docs/report.html) 后在浏览器打开；包含 28 节、40 张可切换地图和独立诊断图。GitHub 文件预览不执行 HTML。

## 安装与快速验证

Python 3.11 或更新版本，建议虚拟环境：

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python self_check.py
python validate_batch.py
python energy_budget.py
python -m unittest test_adaptive -v
python validate_edf_mesh.py
python validate_adaptive_recovery.py
python validate_search_paths.py
```

`self_check.py` 检查腔长/色散、两拓扑初始化公平性、Kerr 和无源传播能量守恒、OC 能量预算、已知平移/相位恢复及归档结果计数。`validate_batch.py` 对照独立单案例传播与批量传播：两拓扑、3 组 OC/pump、10 圈。静态能量预算另外检查空间步长收敛和泵浦光子转换上界。

## 扫描与复现

```sh
# 一个色散切片 × 8 OC × 8 pump；group 取 0–9
python scan.py 0 600

# 完整粗、细网格，共 1280 条轨迹；CPU 运行可能较久
python run_maps.py

# 三组候选各做 0.9/1.0/1.1 倍初始能量的长程延续
python continue_cases.py
python -c "from continue_cases import run; run(2,9,6000,'fine_')"

python check_resolution.py
python audit_results.py
python energy_budget.py
python build_report.py
```

输出默认放在 `outputs/`（已忽略），可用环境变量 `LASER_OUTPUT_DIR` 改目录。粗、细网格原始 MAT 约需 GB 级磁盘空间；MAT 只是 SciPy 的数组存储格式。报告重建需要完成全部扫描和三组延续，不会从结果摘要虚构轨迹。中文图形建议安装 Microsoft YaHei 或 Noto Sans CJK SC 字体。

分组：`group = topology_index * 5 + GDD_index`；拓扑顺序 `OC_CNT, CNT_OC`，GDD 顺序 `−0.1, 0, +0.1, +0.2, +0.3 ps²`。组内 `index = OC_index * 8 + pump_index`。

| 参数 | 取值 |
|---|---|
| 输出耦合 OC | 20–90%，步长 10% |
| 泵浦 LD 输出 | 5, 10, 15, 20, 30, 50, 100, 150 mW |
| EDF | 固定 1.5 m |
| 总腔长 | 固定 20.42 m；共用群折射率 1.4682 |
| 粗网格 | Δt=0.5 ps，2048 点 |
| 细网格 | Δt=0.125 ps，8192 点 |
| 最大传播步长 | EDF 0.1 m；无源光纤 0.5 m |
| 初筛 | 最多 600 圈；每 20 圈检查时间/频谱边界 |

## 文件与模型对应

| 文件 | 用途 |
|---|---|
| `parameters/effective_parameters.json`, `config.py` | 带单位参数、分段光纤、等长色散配置与初态 |
| `python_engine.py` | 独立单案例参考实现，保留损耗与增益诊断 |
| `batch_engine.py` | 批量矢量分步傅里叶传播、动态 EDF、动态 CNT |
| `gpu_batch_core.py`, `gpu_engine.py` | 可选 CuPy/CUDA 实现与 CPU 接口兼容层 |
| `scan.py`, `run_maps.py` | 参数扫描、停止原因与候选筛选 |
| `continue_cases.py` | 同一最终腔内场延续与能量扰动 |
| `energy_budget.py` | 独立静态增益/损耗能量预算，不能用于证明锁模 |
| `build_report.py` | 从保存轨迹重建离线中文报告 |
| `results/reference/` | 本次原始计算的摘要与验证记录；非本机重新计算结果 |

详见 [算法、单位与判据](docs/algorithm.md) 和 [结果边界与来源](docs/provenance.md)。

## 可选 GPU

### P1/P2 新增验证路径

`shared_gain.py` 把多个远端时间窗口接到同一份 EDF 纵向反转上；原 `BatchEngine`
仍表示互不相关的独立腔，二者不可混用。`close_satellite` 提供同窗相干扰动。
`attractor_search.classify` 新增积分 Stokes 均值与跨度，保留复场、周期和纵向反转判据。

```sh
python -m unittest test_adaptive test_shared_gain -v
python validate_satellites.py
python validate_gpu.py
python validate_resident.py
python build_p1p2_report.py
```

`validate_gpu.py` 和 `validate_resident.py` 需要可用 CuPy/CUDA（`build_p1p2_report.py` 本身不需要 GPU，但读取其结果）。
`ResidentEngine(c, dt, a, pop, q, pumps, oc).run(rounds)` 返回紧凑诊断，
`checkpoint()` 显式取回场；检查 `failed` 后才能使用结果。它是 FP64 固定网格初筛接口，
没有替代 CPU 自适应回放。原 `scan.py` 已默认使用新的频谱传播兼容核心；常驻分块扫描见下方新入口。

`floquet.multipliers` 是有收敛门槛的通用实返回映射工具：状态必须无量纲化，
返回映射须固定网格、去除共同相位/时间规范，周期轨道需组成完整一周期。
通过 `neutral_vectors` 提供相位/时间切向量；正式使用需比较差分步长和 Arnoldi 子空间。
目前仅完成已知线性映射验证，**没有激光腔 Floquet 稳定性结果**。

32圈共享增益扰动只观察到卫星比例短时下降，不能认证长期单脉冲。
上一轮小批量 GPU 常驻计时没有显著一致加速；历史记录见 `results/p1p2_validation/`。
本轮已实现 CNT prefix、EDF 融合与频谱传递，详见下方；FP32、自动多保真调度未实施。

自行安装与显卡、CUDA 驱动/运行时匹配的 CuPy。CPU 安装不包含 CuPy，也不依赖任何本机 MATLAB CUDA DLL。

```powershell
$env:MAP_GPU = '1'
python validate_gpu.py
python run_maps.py
```

Linux/macOS shell 用 `export MAP_GPU=1`。运行前先通过 `validate_gpu.py`；不要把历史 GPU 验证记录当作当前机器验证。不同 CUDA 版本的微小数值差异可能影响临界点的长程轨迹。

## 适用范围

这是参数化研究模型，不是指定国产器件的认证数字孪生。NDF/CNT 等有效参数未联合实验标定；无 ASE 自启动；时间窗 1.024 ns 小于约 100 ns 腔周期，无法排除窗口外其他脉冲；600 圈约 60 μs 也不足以保证增益达到长期稳态。净色散仅扫描五个切片，不能排除其他参数中的窄稳定区域。

仓库未附开源许可证；公开可见不等同于授予任意使用或再分发许可。


## 补充验证：最低短时波动候选

对 OC→CNT、OC30%、GDD+0.2 ps²、pump20mW 增加 9 条延续轨迹：原窗口、双倍窗口、时间/无源空间联合细化，各含 0.9/1/1.1 能量倍率。

```sh
python -c "from continue_cases import run; run(3,11,6000,'fine_')"
python -c "from continue_cases import run; run(3,11,6000,'fine_',2048,'_wide')"
python -c "from continue_cases import run; run(3,11,1500,'fine_',1024,'_refined',.0625,.125)"
python validate_best_candidate.py
python build_report.py
```

联合细化未扰动分支完成新增1500圈（累计2100圈），无边界停止，末500圈能量0.063–1.943nJ、CV约124%，仍为多峰。原/双倍窗口首次离带均为累计705圈；细化也为705圈。原网格的频谱停止时间不应被解释为物理寿命；已计算的轨迹尚不满足稳定单脉冲，后续长期吸引态未确定。

## 频谱传播与融合 GPU 内核

详见 [详细分析与模型对应](docs/gpu_optimization.md)。默认 `MAP_GPU=1` 已使用新核心；
设置配置 `gpu_pipeline="reference"` 可回退旧 GPU 核心。原独立 CPU 标量模型未更改。

```sh
python -m unittest test_adaptive test_shared_gain test_spectral -v
python validate_spectral_gpu.py
python benchmark_spectral_gpu.py
python validate_resident_workflow.py
python -c "from scan_resident import scan; print(scan(3, rounds=600, block=20))"
python build_gpu_optimization_report.py
```

除CPU单元测试及报告构建外，上述命令需要CuPy/CUDA。新常驻扫描独立输出
`resident_screen_group_*.npz/json`，不会覆盖原地图；每个案例有 `failed` 和 `completed`。
运行中的粗略峰数只用于初筛，最后另算SciPy精确峰数；仍需完整自适应认证。

实测 RTX 5050 Laptop / CuPy 14.2：64案例单圈FP64核心提速约1.58–1.95倍。
包括上传、诊断和下载的10圈常驻工作流：2048点2.439→1.247秒，8192点3.917→2.459秒。
这是本机、指定批量和参数的中位数，不是其他硬件或640点完整长期搜索的性能保证。
原始计时样本、数值回归和工作流结果见 `results/gpu_optimization/`。

## 完整双网格重算与独立结果目录

```powershell
$env:MAP_GPU = '1'
$env:LASER_OUTPUT_DIR = 'outputs/full_run'
python run_full_maps.py
```

每组64案例，共20组（粗/细网格各640案例），每案例最多600圈；触边仍记为数值未解析。
运行清单保存代码指纹、开始/结束时间和已完成组，可按组恢复；不会把粗网格结果当作细网格补齐。
完整重算与稳定性认证不同，未满足慢增益/复场/全腔扰动要求时不宣称稳定单脉冲。

有旧原始地图时，可设置 `LASER_PREVIOUS_DIR` 后运行 `summarize_full_run.py`，
逐组核对初态、停止状态和能量演化，生成1280行JSON/CSV及文件SHA256。
`build_full_run_report.py` 生成新增诊断节，`build_report.py` 读取实际轨迹重建地图。
构建完整历史报告还需已有的稳态预算及历史HTML片段；它们必须标明历史来源，不能伪装成本次续算。

2026-09-19 已完成全量20组（约32.3分钟）：细网格264点通过数值边界检查，8点目标范围末段单峰，严格筛选0点。新旧细网格有2点停止状态改变，已逐点记录。详见 `results/full_scan_20260919/` 和报告第31节；这些结果不代表已找到稳定锁模解。

## 逐单元增益年龄与长程入口

`gain_age.py` 累积真实接受圈的 Gamma[j]=sum B[j,k]T_R，输出 `gain_age_min` 和 `gain_memory_max=exp(-gain_age_min)`。这是给定速率历史的条件松弛系数，不是完整耦合系统的初态敏感度。GPU融合更新现在直接输出每个EDF单元的B；CPU/GPU诊断有独立回归检查。

`run_gain_relaxation.py` 读取 `LASER_SCAN_DIR/fine_group_08.mat` 中 g08_c53@600 的完整状态，结果写入独立 `LASER_OUTPUT_DIR`。先追加10000真实圈，若年龄不足7可延长到20000；失败重试不累计年龄。预算完成不等于稳定。`audit_gain_terminal.py` 复核末态局部网格，`build_gain_relaxation_report.py` 生成独立HTML。未知的前600圈年龄不回填。

原有自适应入口也改为逐单元累计，并采用年龄7门槛。`continuation` 默认遇到未收敛点就停止该方向；探索性传递需显式 `allow_unconverged=True`，不能作为稳定支路。历史报告保持原口径。

机制复核见 `docs/mechanism_review.html`：CNT调制偏弱与EDF慢暂态均有证据，尚未将前者认定为唯一根因。

本轮已实际续算10586圈（总11186圈），Gamma_min=7.00024，条件系数9.1166e-4。末500圈能量0.909–3.486 nJ、CV42.04%、两个局部峰；1–16圈复场递归均未通过。约250圈的能量自相关线索尚未作长周期复场认证。末态8圈双网格复核通过当前筛选误差要求，但不是全程误差证明。报告见 `docs/gain_relaxation_report.html`，数据与执行源码快照见 `results/gain_relaxation_20260919/`；未继续泵浦支路追踪。

## 周期一联合自洽求解试验

`steady_state.py` 在固定网格上联合求解复光场、纵向反转及整体相位/时间位移，采用平滑模板约束、矩阵无关Newton–LGMRES和反转边界线搜索。用 `N-Neq=0` 替换带有微小恢复因子的动态增益残差，只对周期一固定点等价；不改变物理寿命，不归一化演化能量。CNT未充分恢复时明确拒绝消去其状态。

设置独立 `LASER_OUTPUT_DIR` 后运行 `python run_steady_pilot.py`（需要CuPy）。`continue_steady_pilot.py` 还需 `LASER_SCAN_DIR/fine_group_08.mat` 的已有原始状态；运行后用 `build_steady_report.py` 生成离线报告。CPU检查为 `python -m unittest test_steady_state`，GPU残差/方向导数对照为 `python validate_steady_gpu.py`。

本次五次求解均未取得合格稳态锚点。较好的重启结果相对光场残差约0.82%，仍为两个局部峰、1.81 nJ；无预条件线性求解后期停滞。没有将求解失败解释为物理无解，也没有运行不满足前提的Floquet认证。见 `docs/steady_state_report.html`、`results/steady_pilot_20260919/`。

## 预条件对照与批量雅可比探测

`steady_preconditioner.py` 提供平均光学块/Schur补和完整传播器傅里叶子空间两种实验预条件。`CavityResidual.batch` 支持独立探测批量运行；差分仍使用完整细网格。`solve(..., precondition='coarse', coarse_cutoff=384, rebuild_every=3)` 启用宽带右预条件，默认仍不启用预条件。求解迭代和雅可比探测不是物理圈数。

同一点线性真实残差从0.156降至0.0186，但构建成本3093次映射，总约54秒。12步完整Newton耗时约290秒，最终联合残差0.00971，未优于基线0.00838，也未得到稳态单脉冲。后期线性精度合格而线搜索步长缩至1/32，需要继续检查非线性全局化。34项CPU测试通过；不能把局部峰数1或线性精度改善写成锁模认证。

复现时设置 `LASER_STEADY_DIR=results/steady_block_20260919/input`，另设独立 `LASER_OUTPUT_DIR`。已提供这一个数值初始态，不需要下载完整MAT扫描。依次运行 `run_preconditioned_pilot.py`、`check_block_linear.py`、`check_coarse_linear.py`、`inspect_steady_spectrum.py`、`check_selected_linear.py`、`check_wide_linear.py`、`run_wide_newton.py`，最后用 `build_block_report.py` 生成离线报告。需要匹配的CuPy环境。早期成对非线性试验使用左预条件，执行源码另存；现版本使用右预条件，重跑时以实际输出为准。

报告：`docs/block_preconditioner_report.html`；诊断JSON、可复现输入、末态和源码快照：`results/steady_block_20260919/`。不同原型的串行/批量构建耗时不能用作纯粹的选频性能对照，原历史报告保持不变。

## Newton–GMRES–Hookstep 全局化对照

新增 `steady_hookstep.py`，以真实缩放步长 ||DZy|| 约束右预条件子空间解。共享Arnoldi的12步成对实验：线搜索联合残差0.0106954，Hookstep为0.00460335（降低57%），映射调用15789→12679，耗时274.9→208.7秒。多数接受步rho为0.65–1.05，最后一步0.147；接受半径0.0125–0.2，末态规范条件数2.14，未触发规范重设。仍未获得认证稳态，1.820 nJ也超出目标能量。

原轨迹重放逐步残差差异为零。第11步方向差分误差约8.4e-6，完整Newton步严重偏离局部模型，人口边界不限制；窗口加倍的残差向量差约0.063%。粗子空间归一化梯度0.162，不支持已到投影驻点的判断，但不能据此证明全空间有根或无根。

设置独立 `LASER_OUTPUT_DIR` 后依次执行 `replay_steady_directions.py`、`run_globalization_diagnostics.py`、`run_hookstep_pair.py`、`build_hookstep_report.py`（前三项需CuPy）。输入来自已发布 `results/steady_block_20260919/`。39项CPU回归测试通过。报告 `docs/hookstep_report.html`；原始方向、末态、诊断和执行源码快照位于 `results/steady_hookstep_20260919/`。旧报告保持不变。

## Hookstep末态窗口与软方向续验

报告 `docs/tail_window_report.html`。原Hookstep末态加宽1.024→2.048 ns，残差范数变化仅0.0075%，但全残差向量变化1.22%。相同局域扰动子空间的最小奇异值约1.197e-4，宽窗未使其消失；完整Jv约为投影奇异值4.5倍，不能认定完整物理雅可比存在同样小的奇异值。

宽窗追加23步，因五步下降1.923%且5/5接受rho>0.5触发预设停止。残差0.0046037→0.0037820、输出1.81364 nJ，仍不是认证稳态。新末态扩展至4.096 ns，全残差向量变化0.0087%。PTC同态方向试验最好只再下降0.264%；大移位压小方向却增加原残差，未运行多步PTC或multiple shooting。

复现：设置独立 `LASER_OUTPUT_DIR`，依次运行 `run_tail_diagnostics.py`、`continue_tail_hookstep.py`、`check_tail_final_window.py`、`check_tail_derivative.py`；若停止原因为停滞，再运行 `check_tail_ptc.py`，最后 `build_tail_report.py`。数值脚本需要CuPy。输入是已发布 `results/steady_hookstep_20260919/hookstep.npz`。`steady_window.py`只改变预条件逆的局域表示，完整宽窗残差与Krylov保持全维；`steady_ptc.py`使用场负、反转正、规范零质量的人工流符号，不等于真实慢增益动力学。44项CPU测试通过。结果见 `results/steady_tail_20260919/`。

## Parameter coupling and bordered root diagnostic (2026-09-19)

[Offline report](docs/parameter_geometry_report.html) and [numerical records](results/steady_parameter_20260919) extend the previous tail endpoint. No certified nonzero root or stable single pulse was obtained. The weakest projected mode has a full response 244 times its projected singular value; this is not evidence of a full-system fold. The separate 5–60 mW trial ends at 27.744 mW with residual 0.001793 after 25 steps. Late linear solves reach the 120-vector cap.

With CuPy configured, set `LASER_OUTPUT_DIR` to a separate output directory and run in order:

```text
python run_parameter_geometry.py
python run_bordered_pilot.py
python audit_parameter_modes.py
python check_bordered_endpoint.py
python extend_bordered_pump.py
python check_expanded_endpoint.py
python build_parameter_report.py
```

The extension requires the local trial to reach the lower pump boundary and retains the original hyperplane. The large reconstructed Jacobian cache is not committed. The source archive preserves the exact local execution files separately from the configurable-bound extension. Published SHA manifests link execution logs to those sources. These are root-search diagnostics, not physical stability certification or pseudo-arclength continuation.

## Leakage and fixed-energy controls

The weak-mode leakage implementation in `steady_leakage.py` audits full responses before truncating or capping projected inverse weights. At the latest27.744mW endpoint, no audited mode exceeds5/10/20: these rules do not improve the120-vector solve there. `steady_energy.py` replaces the old hyperplane with a positive output-energy constraint and free pump. No solver change alone constitutes physical root or stability certification.

Use a separate `LASER_OUTPUT_DIR`, with CuPy configured:

```text
python run_leakage_control.py
python run_leakage_fixed_control.py
python run_energy_profiles.py
python check_energy_endpoint.py
python build_energy_report.py
```

The fixed-pump controls have equal8-step budgets;35/30/27.744mW share the saved final guess, while40mW uses its saved earlier endpoint. Intermediate historical fields were not saved. The0.9nJ trial has20 steps and does not automatically continue to smaller energies without root certification.

[Offline leakage/energy report](docs/leakage_energy_report.html) documents a negative result for the proposed weak-mode thresholds at the latest endpoint. The0.9nJ20-step trial reaches0.90256nJ but worsens physical closure to0.003273; no nonzero root or stability certification is claimed.


## Fixed-pump support and directional-derivative controls

[Offline report](docs/fixed_pump_linear_report.html) and [records](results/steady_fixed_pump_20260919) separate energy tuning from physical period-one closure. The fixed-pump residual has no output-energy row. A/B/C preconditioners compare localized time support, wider bandwidth, and full time support on the identical saved state. At 240 vectors, independently checked linear residuals are 2.193%, 2.012%, and 0.063%, respectively. These are linear benchmarks, not nonlinear roots.

A subsequent forward-difference pilot exposed inconsistent large Newton directions: a 0.78% Arnoldi prediction became approximately 14% under centered directional verification. At that same state, centered differences with normalized perturbation 1e-5 give 0.78–0.96% across three verification scales. The formal inner solve therefore uses centered Jv and a separate 1e-6 verification step, with a 1% acceptance gate for the linear solve. Stale coarse factors are refreshed at the same state; failed verification after refresh stops explicitly. LU factors the same projected Jacobian as the SVD baseline, with condition estimation, backsolve checks, and an SVD fallback.

With CuPy available, use a fresh `LASER_OUTPUT_DIR`:

```text
python run_support_benchmark.py
python run_fixed_pump_deep.py
python check_fixed_deep.py
python build_fixed_pump_report.py
```

The report also requires `jvp_consistency.json`: copy the published `forward_quality_gate/` into the output directory and run `run_jvp_consistency.py` to reproduce that same-state control. The input is the archived forward pilot anchor, never the newly overwritten central-run anchor. To reproduce the historical forward pilot itself, use its archived execution sources; the current driver uses centered differences. Input states are the published tail template and `steady_energy_20260919/fixed_final.npz`.

Each formal pump point permits 30 nonlinear steps and 240 Arnoldi vectors; the old 2% progress stop is disabled. The supplied warm path and finite budgets do not establish a stationary branch or residual minima. Root threshold remains 1e-7; no outer energy secant/Brent solve is permitted on unconverged states. Numerical root closure and physical stability certification remain distinct.

Formal result: four points complete30steps with independently checked linear errors below1%, but physical residuals remain1.29e-3–1.43e-3. The27.75/28mW points stop for linear-accuracy limits. No period-one root was obtained. Endpoint recomputation is exact; doubling the window changes full residual vectors by0.026–0.049%.


## Fixed-state merit-gradient and trust-radius diagnostics

`run_merit_geometry.py` performs three controls on published fixed-pump endpoints: full-output C-subspace merit gradients at27.0/27.5mW; six Hooksteps from an identical Arnoldi basis per state; and forward-versus-centered coarse matrices at the27.75mW failed state. No state is advanced. `steady_support_lu.build_factored` keeps its default forward stencil; optional centered columns and streaming audit expose B^T R before output projection without retaining the full B matrix.

Configure CuPy, set a fresh `LASER_OUTPUT_DIR`, then run `run_merit_geometry.py`, `check_merit_steps.py`, and `build_merit_report.py`. Inputs are the published fixed-pump endpoints and tail template. Gradient columns use centered1e-6; the production forward1e-6 preconditioner is retained for radius comparisons. The two27.75mW matrices use the same1e-6 step, factor policy, centered1e-5 Krylov Jv, and three independent checking scales. The report distinguishes restricted first-order stationarity from full-space minima, and does not use a universal Frobenius-normalized gradient threshold.

[Offline merit-geometry report](docs/merit_geometry_report.html) and [raw controls](results/steady_merit_20260919):27.0/27.5mW negative-gradient probes decrease residual0.495%/0.925%, versus0.166%/0.284% for the best tested reduced Hooksteps. Small Frobenius-normalized gradients therefore do not establish stationarity. Larger radii0.025/0.05 worsen both states. Centering the27.75mW coarse matrix leaves conditioning~1.58e9 and checked linear error~2.85%; it does not fix that failure. `check_merit_steps.py` independently replays the best probes; all norms match, with optical improvement partly offset by population residual growth. No new root/stability claim or nonlinear continuation.


## Cauchy-augmented Hookstep: fixed-state audit

`run_cauchy_audit.py` reuses the published 27.0/27.5 mW states and their verified restricted-gradient directions. It audits raw/whitened spans, independently checks the projected candidate, compares model-based selection, and augments the original state/output bases at radii 0.00156, 0.003125, 0.00625. Configure CuPy and a fresh `LASER_OUTPUT_DIR`, run this script, then `build_cauchy_report.py`.

[Offline report](docs/cauchy_span_report.html) and [compact results](results/steady_cauchy_20260920): the original spaces capture only 0.538%/0.411% of the Cauchy direction length; whitening discards no direction. All six raw augmented steps beat Cauchy in model and actual reduction, with rho 0.950–0.999 and no fallback. Actual residual decreases 0.639–0.659% / 1.180–1.206%, versus Cauchy 0.495% / 0.925%. Projected-candidate checks do not implicate the reduced optimizer.

`steady_cauchy.augmented_step` explicitly flags raw model failure and falls back to Cauchy; tests cover this failure path. This is a fixed-state prototype, not a changed default nonlinear loop, full-adjoint method, convergence proof, new period-one root, or stability certificate. Full Z/V/H archives are kept locally with SHA in the JSON; published compact NPZ files retain all trial steps, direction/projection coefficients and singular values. Reproduction can regenerate the larger bases.


## Paired multi-step Cauchy-augmented outer loop

`steady_augmented_outer.solve_augmented` runs the original and augmented methods with a shared trust controller and independent <1% Newton linear gate. `steady_support_lu.build_factored(..., descent=cache)` streams full-output restricted-gradient products from existing forward columns without extra map calls. Between rebuilds, the cached projected Jacobian is applied to the current residual; its lifted direction must pass two current-state directional descent checks. Low rho, model disagreement, linear failure, or non-descent trigger documented refreshes. Raw augmentation failures remain visible even when Cauchy fallback is accepted.

Configure CuPy and a fresh `LASER_OUTPUT_DIR`; run `run_augmented_pair.py`, `check_augmented_pair.py`, `check_augmented_linear_budget.py`, then `build_augmented_pair_report.py`. The paired protocol fixes 27.5 mW and the published `down_275` input for 40 outer steps per arm. The radius starts at0.003125 and may grow up to2.0. No pump scan, artificial energy residual, progress-based early stop, or physical-stability claim is introduced. A failed independent linear gate after rebuilding ends that arm with its unused budget explicit.

[Offline paired report](docs/augmented_pair_report.html), [raw paired results](results/steady_augmented_pair_20260920). CPU tests include exact streamed gradients without extra evaluations, a known linear root, nonlinear Rosenbrock convergence with rejected trials, non-descent-triggered refresh, and trust-radius expansion past the former one-step range.


Paired outcome: A accepts18steps (residual1.2758381e-3), B accepts30steps (1.2180302e-3); neither completes the40-step allowance because the independent linear gate fails. At the same18 accepted steps B removes3.45times as much residual norm as A. B decreases the norm5.946% overall, but its last5steps average only0.0424% per step: no rapid root-convergence regime is observed. All30 accepted B steps are raw augmentations, with9fresh full-output restricted directions and21checked cheap directions; no fallback or raw model failure. Full240-vector endpoint checks lower internal model errors to4.09e-11/7.68e-12 yet independent errors remain3.009%/1.861%. Endpoint replay matches exactly. No new root or stable pulse; no27.0mW replication/pump continuation is claimed.


## Step-relevant trust-region gate

The optional `gate_policy="step"` in `solve_augmented` verifies the selected
constrained candidate at normalized central-difference scales 3e-6 and 1e-6.
Both predicted reductions must agree with its subspace model within 5%; the
full nonlinear map must decrease with rho > 0.1. The unconstrained Newton 1%
gate is retained when its norm is inside the current radius, and otherwise
logged without blocking a valid constrained candidate. Cauchy safeguards and
current-state descent checks remain active. The default `"newton"` preserves
prior paired-run reproducibility. Arnoldi stopping is unchanged in this control.

With CuPy configured and a fresh `LASER_OUTPUT_DIR`, run:

```text
python run_step_gate_audit.py
python run_step_gate_continue.py
python check_step_gate.py
python report_step_gate.py
```

The second command refuses to run unless the fixed-state audit admits a raw
augmented candidate. Inputs are the published paired endpoints, saved full-budget
Newton vectors, and original tail template. No new pump scan or stability claim.
The report distinguishes Arnoldi's linear combination of basis responses from
direct finite differences on its final direction, and does not assume the
smallest difference step is most accurate.


[Offline step-gate report](docs/step_gate_report.html) and
[records](results/steady_step_gate_20260920): all three fixed-state raw augmented
candidates pass, with model discrepancy about 0.049% and actual rho 0.977,
0.917, 0.619. Thirty further B steps decrease residual 1.21803022e-3 to
1.20814392e-3 (0.8117%); four accepted steps would fail the old Newton gate.
There are nine fresh and twenty-one checked cheap directions, no fallback,
and exact endpoint replay. The last five steps average only 0.004995% relative
improvement. False stopping is repaired, but rapid root convergence is not
obtained. The endpoint remains uncertified, at about 0.903049 nJ.


## Single-state descent-source audit and rolling history

[Offline report](docs/descent_sources_report.html) and
[records](results/steady_descent_sources_20260920) compare S1=current C direction,
S2=plus375--750GHz optical-envelope annulus, S3=plus last3fresh historical directions,
and S4=plus the physical parameter selected by a prespecified scalar trust model.
All share one current C preconditioner/Arnoldi basis and three radii. C is NOT a hard
cutoff on Krylov search: its complement inverse remains -I. Saved steps verify this.
Annular full-output central gradients stream without a larger dense factorization.
All physical probes keep the current27.5mW pump except the pump coordinate itself.

Copy the published `history_directions.npz` and `history_recovery.json` into a fresh
`LASER_OUTPUT_DIR`, configure CuPy, then run:

```text
python run_descent_sources.py
python check_descent_sources.py
python check_descent_support.py
python run_descent_continue.py
python check_descent_continue.py
python report_descent_sources.py
```

To independently reconstruct the historical vectors instead of using the verified
archive, run `recover_descent_history.py` first. It checks every previous residual
and the final state, and is a replay rather than a further nonlinear search.

`solve_augmented(..., gate_policy="step", enrichment=callback)` accepts extra state
directions as columns and recomputes their responses at each current state. It uses
the same metric whitening, Cauchy lower bound and independent candidate gate. The
S3 driver rolls the last3fresh directions and archives new fresh vectors and states.
No added energy/hyperplane equation or physical-parameter change occurs in S3.

At the audit state, S2/S3 actual merit reductions are about5.3--5.5e-9, versus
3.1--7.8e-11 for S1. All12 candidates replay exactly. Ratios68--176 depend on radius
and the weak baseline; they are not long-run speedup factors. S4 selects Psat by its
bounded scalar model and is weaker in these radii; this does not test all joint
parameter optima or larger parameter steps. Only candidates with >=25% actual
improvement over a valid same-radius S1 qualify for continuation.

S3 achieves99.56% of the best one-step reduction without recurring annulus sweeps,
so it is selected for30steps. All30are accepted with no fallback; residual falls
1.2081439205e-3 to1.1883945817e-3 (1.634684%). Final5steps average0.015245% norm
improvement. Endpoint residual and output replay exactly. No period-one root or
stable-pulse certificate is obtained; S2 has not received a matched long-run trial.

## Active-history audit at the S3 endpoint

[Offline report](docs/active_history_report.html) and
[records](results/steady_active_history_20260920) compare H3, G2, G3, G4, G3+A
at one frozen state (residual 1.1883945817e-3), three radii, one C/Arnoldi model.
Seven archived fresh histories receive new current-state Jacobian responses.
Greedy selection maximizes predicted trust-model decrease; no slope/cosine filter.
Two orthogonal compressions preserve the actual state norm and residual norm.
The forced four-addition audit is separate from the prospective 5% stopping rule.

All radii select histories from outer steps 18, 28, 14, 1 in that order. The first
is uphill along its positive orientation; subspace coefficients can reverse it.
G3 achieves only 32--34% of G4 actual reduction. None of the first four additions
triggers the 5% rule. G3+A delivers 11.9--12.5 times G3 actual reduction, about
four times G4. All 15 candidates pass and independently replay exactly.
Both seven-column history and response matrices have numerical rank seven at
relative tolerance 1%. This does not support a two/three-dimensional saturation.
The oracle advantage is specific to this state; no periodic schedule is proved.
G4+A and longer continuations are not tested. No root or stability certificate.

With CuPy configured, use a fresh LASER_OUTPUT_DIR and run:

```text
python run_active_history.py
python check_active_history.py
python report_active_history.py
```

The input vectors are already archived; no historical replay is required.
CPU validation: `python -m unittest test_active_history -v`.

## Complete history bank and annulus novelty

[Offline report](docs/history_complete_report.html) and
[records](results/steady_history_complete_20260920) extend the same frozen S3
state through G7 and G7+A. No continuation, parameter release or annulus sweep.
The missing common Arnoldi basis is rebuilt; all prior fifteen models replay
exactly. Current responses are recomputed for the persisted directions.

All 128 subsets are enumerated at each of three radii. Greedy matches the best
prediction at every equal cardinality. G7+A still achieves 2.52--2.57 times G7
actual merit decrease. State/response novelty is 0.98856/0.97408; SVD tolerances
1e-8--1e-12 agree, retaining 46 dimensions. The response projection excludes the
residual target column. A normalized shared-coefficient fit also has large error.
This supports a useful extra direction at this state, not a reseeding schedule,
physical dimension claim, or proof that good steps live primarily in the annulus.

All 24 new candidates pass; full-map and 384 small-model replays match exactly.
The compressed model is saved for future subset diagnostics without rebuilding.
With CuPy configured and a fresh LASER_OUTPUT_DIR:

```text
python run_history_complete.py
python check_history_complete.py
python report_history_complete.py
```

CPU unit check: `python -m unittest test_history_complete -v`.
No period-one root or stable-pulse certificate is claimed.

## Persisted annulus seed lifetime

[Offline report](docs/persisted_seed_report.html) and
[records](results/steady_persisted_seed_20260920) follow one immutable seed from
the same S3 endpoint, keeping all current history directions (capacity12).
Fresh C directions enter the next model; over-capacity history is removed by
minimum current leave-one-out model loss with seed present. Current C remains
separate. All seed/history responses are recomputed at each state. The optional
model_factory path preserves the existing step gate and Cauchy/full-map checks.

Accepted steps: 30; status: iteration_budget_reached.
Residual: 0.00118839458173 -> 0.00113064270299.
Last-five mean norm gain: 0.0842852%/step.
GA>1.15 at 1/9 fresh states.
Prespecified progress criterion A: False; durability criterion B: False.
No fresh annulus sweep occurred. Hypothetical trigger state indices (zero-based):
[]. These do not establish actual reseeding need,
fresh-seed recovery or adjoint economics. No stable-pulse certificate is claimed.

With CuPy configured and a fresh LASER_OUTPUT_DIR:

```text
python run_persisted_seed.py
python check_persisted_seed.py
python report_persisted_seed.py
```

`trajectory.npz` contains accepted states, immutable seed, fresh C vectors and
their generating states. The final pending bank may include one newly accepted
direction awaiting next-state capacity selection; every executed model uses<=12.
CPU check: `python -m unittest test_seed_bank test_augmented_outer -v`.

## Four frozen-state fresh-annulus controls

[Offline report](docs/fresh_retrospective_report.html) and
[numeric records](results/steady_fresh_retrospective_20260920).
Zero-based saved states 5,11,20,27 correspond to report steps 6,12,21,28.
Each old candidate was replayed before the sweep; state, history IDs, C direction,
Arnoldi basis and radius were fixed. No continuation or trigger change.

| Saved state | Gold | Gfresh | cos_x | cos_J | Fresh descent |
|---|---|---|---|---|---|
| 5 | 1.00005303 | 1.28422752 | -9.512757e-05 | -0.2482268 | 8.746542e-05 |
| 11 | 1.00278560 | 1.84211947 | -3.943598e-05 | -0.2498778 | 7.418687e-05 |
| 20 | 1.00121208 | 3.01138017 | 8.537639e-06 | -0.1833764 | 9.501684e-05 |
| 27 | 1.00006233 | 8.94955392 | -1.548184e-05 | -0.1648918 | 9.761863e-05 |

Prespecified threshold met: 3/4; validated recovery: 3/4.
Gate to begin adjoint work (at least 3/4): True.
All comparisons concern one frozen model per state; these do not measure an
adjoint speedup, certify a physical root or establish a reseeding period.

With CuPy configured and a fresh LASER_OUTPUT_DIR:

```text
python run_fresh_retrospective.py
python check_fresh_retrospective.py
python report_fresh_retrospective.py
```

State archives include full candidate steps and current seed responses;
model archives allow independent reduced-model replay. Raw execution sources,
input hashes and independent full-map replay are archived with results.
