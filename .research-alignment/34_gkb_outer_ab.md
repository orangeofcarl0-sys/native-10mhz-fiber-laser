# 34 深层与回收GKB正式20步A/B

主问题：回收空间能否跨多个状态以更低成本保持fresh深层的推进？
起点：pure40末态R=0.0009746756121000674，两臂物理/网格/gauge完全相同。
A每状态fresh192；B每状态fresh64与上个接受状态fresh64正交合并，所有旧响应在当前J重算；从保存的旧状态V64启动。旧缓存来源为pure40第40步之前的状态。B只携带fresh64，不携带union。
半径初值0.0125；继承上步接受候选半径。差候选（不满足基本保护、实际<=0或rho<0.25）减半；良好rho>0.8且边界解时乘1.5外试，新点实际提升且rho>0.5才继续。每状态只构建一次空间。上限各12次内试/外试，半径底1e-10，非物理停止而非结果接受。
所有实际尝试候选中，按真实下降降序，选择通过feasible、正实际与预测、rho>0.1、同半径Cauchy以及三尺度Js(h=1e-5,3e-6,1e-6,5%一致性)的候选。epsilonNL只记录，不门限。无候选通过则同一空间继续向内减半，至预算/底停止。
两臂最多20接受步或根阈值1e-7。若预算不能完成，明确状态，不假装完成。
计时：同GPU顺序运行，不并行测试；每臂记录basis、radius probes(含小SVD)、gate、总算法时长（不含文件归档），另列实耗和初始化。B warm使用现存缓存，cold另计重建成本参考，不混同。
决策：两臂都完成20接受步(或更早达根)，P=-log(Rend/R0)，P_B/P_A>=0.9且T_B/T_A<=0.75支持B默认；报告total与basis口径。首轮不自动设计hybrid或改其他默认入口。
验证：已知答案非线性测试与合并基测试；独立重放40接受步、三尺度最终方向、检查选择为已尝试且过门的最大真实下降；场/反转/gauge闭合分项。
交付：轨迹、逐试记录、可复现代码、离线HTML及图、公开结果归档。数值下降不等于周期根或稳定锁模认证。

## Observed results and limits

Both arms completed20 accepted steps. A R=0.0009315468344505816,
P=0.04525824279308954, algorithm2255.1003s. B R=0.0009562233316559857,
P=0.019113213813161455, algorithm959.0057s. P_B/P_A=0.4223145,
T_B/T_A=0.4252608: time gate passes, progress gate fails. Keep fresh192 plus
inner-radius control as baseline; do not promote pure recycled64+64.
A had3 boundary accepted steps, B only1; B steps2–20 have lambda0. Both keep
accepted radius0.0125. Last-five mean relative norm decrease A0.16482% vs
B0.05401%. Recycled rank stays128, and accepted directional gates pass.
The data support smaller usable steps in this recycled trajectory; they do
not uniquely identify a physical mechanism or validate a hybrid trigger.
Independent replay checks all40 accepted updates and three-scale predictions.
HTML QA:17sections,9figures,zero broken/external assets,430px mobile,zero JS errors.
