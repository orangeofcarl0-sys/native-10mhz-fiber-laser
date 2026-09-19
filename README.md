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
没有替代 CPU 自适应回放，也尚未自动接入原 `scan.py`。

`floquet.multipliers` 是有收敛门槛的通用实返回映射工具：状态必须无量纲化，
返回映射须固定网格、去除共同相位/时间规范，周期轨道需组成完整一周期。
通过 `neutral_vectors` 提供相位/时间切向量；正式使用需比较差分步长和 Arnoldi 子空间。
目前仅完成已知线性映射验证，**没有激光腔 Floquet 稳定性结果**。

32圈共享增益扰动只观察到卫星比例短时下降，不能认证长期单脉冲。
小批量 GPU 常驻计时没有显著一致加速；记录见 `results/p1p2_validation/`。
CNT prefix、FP32、自动多保真调度尚未实施。

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
