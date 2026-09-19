# 诊断输入

三份 NPZ 均来自已归档的 OC→CNT、OC30%、GDD+0.2ps²、pump20mW 计算。

- near_candidate：细网格第600圈的腔内场、15个EDF单元反转，dt0.125ps。
- boundary_snapshot：原窗口延续中未扰动分支的末圈腔内场，已触发频谱边界。只用于传播器数值压力测试，不作为可信物理输出。
- multipulse_snapshot：联合细化延续的未扰动分支，总第2100圈腔内场，dt0.0625ps。

NPZ只包含 a、pop、dt；EDF独立网格检查中保持输入场与时间采样不变，保守重分配反转单元平均值。局部EDF步长对照不能替代完整腔长期收敛认证。
