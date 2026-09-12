import argparse,pathlib,json,hashlib,datetime,subprocess,shutil,difflib,xml.etree.ElementTree as ET,ast
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();r=a.run
assert not (r/'manifest.json').exists(),'sealed evidence is immutable'
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
plan=read(r/'plan.json');stats=read(r/'statistics.json');checks=read(r/'input_validation.json');audit=read(r/'additional_audit.json');runtime=read(r/'runtime_config.json');cfg=runtime['configs']['A-seed42'];repo=pathlib.Path(plan['repository']);toolroot=pathlib.Path(__file__).parent
assert sha(pathlib.Path(plan['model']['path']))==plan['model']['sha256']
for path,h in plan['protected_source_hashes'].items():assert sha(pathlib.Path(path))==h,path
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==plan['source_head']
assert subprocess.check_output(['git','status','--short'],cwd=repo,text=True)==plan['source_dirty']
assert len(stats)==9 and all(x['result']['status']=='completed' for x in stats.values())
for cid,v in checks.items():
 assert not any(x is False for x in v.values()),(cid,'failed bool check')
 assert v['records']==6000 and v['physics_steps']==60000 and v['pd_updates']==24000
 assert v['reset_steps']==[0] and v['terminated_steps']==[5999]
 assert all(x<1e-6 for k,x in v.items() if k.endswith('error'))
 c=runtime['configs'][cid];assert c['model_inputs']==[{'dtype':'float32','name':'obs','shape':[1,195]}];assert c['model_outputs']==[{'dtype':'float32','name':'actions','shape':[1,10]}]
 assert c['physics_timestep']==.002 and c['decimation']==4 and c['gravity']==[0.,0.,-9.81]
 for key in ['initial_qpos','initial_qvel','initial_ctrl','geoms','bodies','joints','actuators','rl_kp','rl_kd','default_dof_pos','action_scale','history_indices','clip_obs']:assert c[key]==cfg[key],(cid,key)
 assert all(s['noise']==0 for s in c['sensors'] if not s['name'].endswith('_torque'))
 assert all(s['cutoff']==0 for s in c['sensors'])
 g=[g for g in c['geoms'] if g['body_id']==0];assert len(g)==1 and g[0]['type']==0 and g[0]['quat_body_wxyz']==[1.,0.,0.,0.]
 av=audit['cases'][cid];assert not any(x is False for x in av.values());assert av['first_applied_delay_seconds']==[.006]
assert all(audit['comparisons'].values())
# Actual model graph metadata; use the existing structural protobuf parser, no dependency install.
tree=ast.parse((toolroot.parent/'lw_sim2sim/seal.py').read_text());ns={}
for node in tree.body:
 if isinstance(node,ast.FunctionDef) and node.name in ['varint','fields']:exec(compile(ast.Module(body=[node],type_ignores=[]),'<existing-parser>','exec'),ns)
fields=ns['fields'];mf=list(fields(pathlib.Path(plan['model']['path']).read_bytes()));graph=next(v for t,w,v in mf if t==7);nodes=[];inits=[]
for t,w,b in fields(graph):
 if t==1:
  d=list(fields(b));nodes.append({'name':next((v.decode() for t,w,v in d if t==3),''),'op':next((v.decode() for t,w,v in d if t==4),''),'inputs':[v.decode() for t,w,v in d if t==1],'outputs':[v.decode() for t,w,v in d if t==2]})
 if t==5:inits.extend(v.decode() for t,w,v in fields(b) if t==8)
write(r/'onnx_graph_contract.json',{'nodes':nodes,'initializer_names':inits,'input':cfg['model_inputs'],'output':cfg['model_outputs'],'normalization_enabled':False,'evidence':'user and archive training description; graph has no normalization operators or named normalization parameters'})
assert len(nodes)==14 and not any('normal' in n.lower() for n in inits)
assert all(n['op'] not in ['BatchNormalization','InstanceNormalization','LayerNormalization'] for n in nodes)
# Source snapshots and exact diagnostic-only diff, including the runtime header hook.
for p in toolroot.iterdir():
 if p.is_file():shutil.copy2(p,r/'tools'/p.name)
patch=[]
for current,base in [(toolroot/'collect.cpp',toolroot.parent/'lw_sim2sim/collect.cpp'),(toolroot/'lw_runtime_core.hpp',repo/'src/rl_sar/library/core/safety/lw_runtime_core.hpp')]:
 patch.extend(difflib.unified_diff(base.read_text().splitlines(True),current.read_text().splitlines(True),fromfile=str(base),tofile=str(current)))
(r/'tools/diagnostic_changes.patch').write_text(''.join(patch))
helper=repo/'.agents/skills/monitor-tune-isaaclab-training/scripts/sim2sim_report_bundle.py';shutil.copy2(helper,r/'tools/sim2sim_report_bundle.py')
# Runtime dependencies are identified by the actual executable's dynamic linker output.
ldd=subprocess.check_output(['ldd',str(r/'tools/collect')],text=True);(r/'tools/ldd.txt').write_text(ldd);deps=[]
for line in ldd.splitlines():
 parts=line.split();paths=[x for x in parts if x.startswith('/')]
 for x in paths:
  p=pathlib.Path(x)
  if p.is_file():deps.append({'path':str(p.resolve()),'sha256':sha(p)})
write(r/'runtime_dependencies.json',{'libraries':deps,'analysis_python':'/usr/bin/python3','analysis_numpy':'1.21.5','analysis_matplotlib':'3.5.1','simulation':'MuJoCo 3.2.7; ONNX Runtime 1.22.0 CPU','initial_geometry':'same MuJoCo 3.2.7, zero physical steps','video':'not requested; not recorded'})
geometry=read(r/'initial_geometry.json');runtime['initial_geometry']=geometry;runtime['unavailable']=[x for x in runtime['unavailable'] if not x.startswith(('Minimum geometric','Per-step lowest'))]+['Per-step lowest collision geometry clearance; initial mesh clearance and per-step site heights are available'];write(r/'runtime_config.json',runtime)
compile_record=read(r/'tools/build_command.json');assert sha(toolroot/'collect.cpp')==compile_record['source_sha256'][str(toolroot/'collect.cpp')]
final={'completed_cases':9,'early_terminated_cases':0,'failed_cases':0,'simulation_seconds':sum(s['result']['final_sim_time'] for s in stats.values()),'authorized_maximum_seconds':1080,'policy_records':54000,'physics_substeps':540000,'pd_updates':216000,'initial_resets':9,'midrun_resets':0,'all_integrity_checks_pass':True,'production_sources_unchanged':True,'training_noise_config_evidence':'hash-matched inherited source plus user-confirmed absence of flat overrides','training_runtime_exact_source_version':'unknown; official reference implementation captured for operation ordering','receiver_inventory_received':False,'git_commit_push_performed':False,'preflight_note':'preflight advanced zero physics and performed zero closed-loop inference; production model loading performs validation/warmup inference. The original compiled preflight field model_inferences=0 refers to closed-loop calls and was clarified in preflight_config.json.'}
write(r/'final_checks.json',final)
def win(cid,lo,hi):return next(w for w in stats[cid]['windows'] if w['window_s']==[lo,hi])
def f(v,d=6):return f'{v:.{d}f}'
def vec(v,d=6):return '('+', '.join(f(x,d) for x in v)+')'
A=win('A-seed42',40,120);B=[win('B-seed'+str(s),40,120) for s in [42,43,44]];C=[win('C-seed42',lo,hi) for lo,hi in [(40,60),(60,80),(80,100),(100,120)]]
deltas=[{'seed':seed,'window_s':[40,120],'B_minus_A_net_displacement_m':b['xy_net_m']-A['xy_net_m'],'B_minus_A_fitted_drift_speed_m_s':b['fit_drift_speed_m_s']-A['fit_drift_speed_m_s'],'B_minus_A_mean_planar_speed_m_s':b['speed_xy']['mean']-A['speed_xy']['mean'],'B_minus_A_body_vx_m_s':b['vx_body']['mean']-A['vx_body']['mean'],'B_minus_A_yaw_change_deg':b['yaw_change_deg']-A['yaw_change_deg']} for seed,b in zip([42,43,44],B)];write(r/'comparison.json',{'A_B_40_120':deltas,'across_seed_identity':audit['comparisons']})
lines=[f'# Flat wheel DWAQ 零指令漂移诊断：{r.name}',
'',f'本批次在 LFR 的独立平地诊断入口运行 A/B/C × seeds 42、43、44，各 120 s，共 1080 s 仿真。9 次全部完成，无跌倒、异常终止或中途 reset。没有批准的验收阈值，本报告只给出测量事实，不给出“通过”或硬件可用结论。',
'',f'**核心结果：无附加噪声也存在持续漂移。** A 的 40–120 s 净位移 {A["xy_net_m"]:.6f} m，拟合漂移速度 {100*A["fit_drift_speed_m_s"]:.4f} cm/s，平均机体 vx={A["vx_body"]["mean"]:.6f} m/s（向后）。B 三个 seed 同样漂移，但本次净位移、拟合速度及平均平面速度均低于 A；未复现 Isaac 参考中“仅加噪声才产生明显漂移”的关系。C 在停车后的最后两个 20 s 窗口仍持续移动。',
'', '## 身份、条件与采集边界','',f'- 实际加载 ONNX：`{plan["model"]["path"]}`。SHA-256：`{plan["model"]["sha256"]}`，与指定值完全一致。',f'- 归档：`{plan["model"]["archive"]}`；JIT SHA-256：`{plan["model"]["jit_sha256"]}`，匹配但未用于闭环。训练 run `2026-09-11_17-04-10`，checkpoint `model_49999.pt`，任务 `RobotLab-Isaac-Velocity-Flat-LW-wheel-Dwaq-v0`；算法/runner 为归档声明的 DWAQPPO / OnPolicyRunnerDwaq。训练 checkpoint 字节本机未持有，其哈希仅作为归档声明。',f'- rl_sar HEAD：`{plan["source_head"]}`。相关未提交差异、源码原字节及归档说明在 `config_snapshot/`；生产文件运行前后哈希与 Git 状态一致。policy_storage 更名提交 `1854e61` 在本机历史中。', '- ONNX 实际接口 float32 `obs[1,195] -> actions[1,10]`。每帧 39 维，5 帧由旧到新排列；DWAQ 内部 actor 顺序为 `[encoded_velocity, encoded_latent, current_obs]`，已包含在模型中，不是外部拼接要求。训练说明声明归一化关闭，图结构也无归一化算子或对应参数，本程序不添加归一化。',
'- 实际 XML 在 `config_snapshot/src/rl_sar_zoo/LW_description/mjcf/scene.xml`：唯一地面为世界 z=0 的无限 plane，单位姿态，法向 +Z，坡度 0°，无台阶/heightfield；独立场景移除了原六个台阶，机器人模型未改。地面/碰撞默认摩擦 `[0.8,0.005,0.0001]`、condim=3、solref=`[0.02,1]`、solimp=`[0.9,0.95,0.001,0.5,2]`。完整编译后模型参数在 runtime_config.json。',
'- 无参数/初态随机化、推扰或外力；每个物理轨迹记录的 xfrc_applied/qfrc_applied 均为零。进入策略的传感器 noise 声明为零，全部 cutoff=0；另有10个扭矩传感器声明 noise=0.01，扭矩不是195维策略输入，本次保留该声明。无附加滤波或可配置量化；保留 float64 仿真到 float32 观测的精度转换。',
'- 初态完全沿用 home_wheel：base 原点 `[0,0,0.70]` m，body-to-world wxyz `[1,0,0,0]`，全部根部/关节 qvel=0、ctrl=0；策略顺序默认角 `[0,0,0.733,-0.733,-0.1745,0.1745,0,0,0,0]` rad。右/左轮碰撞网格初始最低点分别为 '+f(geometry['collision_mesh_min_z_m']['right_wheel_collision'])+'/'+f(geometry['collision_mesh_min_z_m']['left_wheel_collision'])+' m；这是同一 MuJoCo 3.2.7 对初态的静态几何计算，不含接触 margin，没有推进仿真。',
'- 每用例独立进程；先物理 home_wheel reset，再正常 WheelLocomotion Enter/Activate 初始化控制器、上一动作和历史；没有额外姿态恢复等待。每次首帧复制填满 5 帧，之后每策略步移动一次。物理热键 reset 单独使用并不会重新激活控制器；本次不执行中途热键 reset，也不声称验证其历史行为。',
'- MuJoCo **3.2.7**，ONNX Runtime **1.22.0**，现有 CPU 后端。physics dt=0.002 s；policy=0.02 s；名义低层=0.005 s、decimation=4，沿用原诊断调度，实际低层间隔为 0.006/0.004 s 交替。新策略输出首次写入 ctrl 的实测延迟固定为 **0.006 s**；传感器状态通常来自前一物理步起点，实测年龄 0.002 s（初始为0）。未添加执行器延迟，仍保留以上调度延迟。未测试交互部署程序的异步线程墙钟抖动。',
'- 完整实际启动 argv/cwd 在 commands.json 和每个 process.json，编译命令、动态库路径/哈希在 tools/build_command.json、runtime_dependencies.json。模型加载有原部署验证/warmup，不计入闭环仿真预算。',
'- 停止条件：非有限输入/动作/状态、MuJoCo 警告、控制保护锁存、base z<0.25 m、直立轴总倾斜>75°、机身碰撞、程序异常；单次墙钟600 s，上层超时615 s。均未触发。总倾斜仅用于保护，不代替 pitch 指标。',
'', '## 观测与控制核对','', '| 帧索引 | 字段及坐标/单位 | 缩放 | B 加噪前幅度 |','|---|---|---|---|','| 0:3 | 机体角速度 rad/s | 0.25 | uniform ±0.2 |','| 3:6 | Rᵀ[0,0,-1]，机体系单位重力 | 1 | uniform ±0.05 |','| 6:9 | vx、vy m/s，wz rad/s | 1 | 无 |','| 9:19 | q-default_q，rad；轮位置按名字对应列屏蔽 | 1 | 非轮 ±0.01；轮为0 |','| 19:29 | 全部关节 dq，rad/s | 0.05 | uniform ±0.5 |','| 29:39 | 上一步模型动作经原动作 clip、未缩放 | 1 | 无 |',
'', '索引从0开始，最新帧位于156:195，轮位置绝对索引173、174。完整输入直接从模型调用的 TensorView 复制，非离线重建。四元数为 base_link 到世界的 wxyz，body +X 前向、+Z 向上；机身可视 mesh 的内部变换保留在 XML。原始 ZYX pitch=`asin(2*(qw*qy-qx*qz))`，本报告后仰正值=`degrees(asin(clamp(2*(qx*qz-qw*qy),-1,1)))`。roll 单独计算 `atan2(R[2,1],R[2,2])`，yaw 独立解缠。速度测量点为 freejoint/base 原点；世界速度为 qvel前三维，机体速度为 Rᵀv；COM速度另存，不混用。',
'', '| policy索引 | 关节 | MuJoCo joint ID | qpos地址 | qvel地址 |','|---|---|---|---|---|']
for m in cfg['joint_mapping_evidence']:lines.append(f'|{m["policy_index"]}|{m["joint_name"]}|{m["mj_joint_id"]}|{m["mj_qpos_adr"]}|{m["mj_dof_adr"]}|')
lines+=['','关节/模型动作顺序相同，没有交换左右关节。前8个关节为位置目标，最后右/左轮为速度目标。动作先按各维 ±100 裁剪；`q_target=default_q+0.25*a`（前8关节），轮 q_target=default_q、`dq_target=a`；其余 dq_target=0。外部 PD 为 `tau=feedforward+Kp*(q_target-q)+Kd*(dq_target-dq)`，再限制到各关节扭矩上限写入 MuJoCo ctrl。执行器为直接力矩 motor，gear=1；另有 ctrlrange 限幅，无独立 forcelimited 动态环节。',
'','| 关节组 | Kp | Kd | 力矩/ctrl上限 Nm |','|---|---:|---:|---:|','| 髋、大腿、小腿（6） |90|3|120|','| 脚（2） |36|1.8|27|','| 轮（2） |0|0.5|40|',
'', '关节名义 damping=0.01、armature=0.01、frictionloss=0.2；freejoint 为0。完整逐body质量、COM、主惯量及惯性轴四元数、关节范围和执行器编译值均保存在有效配置。base质量10.802295 kg，body COM=`[0.08622748,-0.00009382,-0.03217461]` m。',
'', '## 噪声证据与隔离','', '基础 rough_env_cfg.py 与归档哈希一致；训练时 flat_env_cfg.py 原字节及有效配置JSON未在本机找到，用户明确确认 flat 继承且不覆盖相关配置。以此确定本次训练噪声幅度和非轮位置选择规则，报告保留这一人工来源。操作顺序为观测项原值→选择性加噪→项裁剪±100→缩放→历史。该顺序也与捕获的 [IsaacLab官方观测管理器参考实现](https://raw.githubusercontent.com/isaac-sim/IsaacLab/main/source/isaaclab/isaaclab/managers/observation_manager.py) 一致；参考文件不能证明训练机安装版本，版本绑定仍缺失。',
'', '独立目录的 lw_runtime_core.hpp 仅在原观测构建与原历史插入之间添加钩子；生产原文件未改。A/C 帧逐值不变。B 每帧从该帧原始值加一次噪声，再进入原5帧历史；初始化复制同一首个带噪帧，后续旧帧不重新加噪。每个实测状态均验证无噪声训练裁剪路径与原部署输出完全相等，因此本次没有额外引入裁剪顺序差异。',
'', '噪声生成使用独立 std::mt19937(seed)，取输出高24位形成float32 uniform。每帧按gyro3、gravity3、position10、velocity10抽样；位置轮列抽样后丢弃以保持0。与Torch RNG不是同一条序列，不能将同 seed 解释为跨引擎逐噪声配对。原值、抽样、scale和最终帧逐步存储，独立Python复算全部 B 序列完全一致。A/C 没有可用随机源影响动力学，三seed的位姿/输入/动作逐值一致，属于同一确定性轨迹的重复执行，不构成三个不同初态样本。',
'', '## 测试与统计定义','', 'A/B全程零指令，B仅开启上述噪声；C：0–40 s零指令，40–60 s `[0.5,0,0]`，60–120 s零指令。A和C前40 s位姿、输入、动作逐值一致。每用例6000策略记录、60000物理步、24000低层更新，全部终止原因 time_limit。',
'', '逐控制步前状态、动作、各物理子步和执行后状态分开保存。统计窗口状态采样为50 Hz，通常为[lo,hi)；位移及航向使用精确窗口端点（120 s取最后post状态）；位置拟合使用包括端点的原始位置，不平滑、不降采样。XY拟合斜率为分别对 x(t)、y(t)普通最小二乘，漂移速度是两斜率向量的模；最大距离相对窗口起点，活动范围=max-min。速度RMS/P95指世界XY速度模；均值vx同时报告机体系有符号值。累计路程为50 Hz位置差模之和，仅作补充。所有窗口均处于单个reset段；0秒含初始reset，后续无reset。完整精度及全部逐关节动作/扭矩指标见 statistics.json。',
'', '## 核心对比：40–120秒','', '|条件/seed|净位移 m|拟合速度 cm/s|平均XY速度 cm/s|平均机体vx m/s|航向变化 °|','|---|---:|---:|---:|---:|---:|']
for cid in ['A-seed42','B-seed42','B-seed43','B-seed44']:
 w=win(cid,40,120);label='A / 42、43、44（逐值相同）' if cid.startswith('A') else cid;lines.append(f'|{label}|{f(w["xy_net_m"])}|{f(w["fit_drift_speed_m_s"]*100,4)}|{f(w["speed_xy"]["mean"]*100,4)}|{f(w["vx_body"]["mean"])}|{f(w["yaw_change_deg"],3)}|')
lines+=['','|seed|B−A净位移 m|B−A拟合速度 cm/s|B−A平均XY速度 cm/s|B−A航向变化 °|','|---|---:|---:|---:|---:|']
for d in deltas:lines.append(f'|{d["seed"]}|{f(d["B_minus_A_net_displacement_m"])}|{f(d["B_minus_A_fitted_drift_speed_m_s"]*100,4)}|{f(d["B_minus_A_mean_planar_speed_m_s"]*100,4)}|{f(d["B_minus_A_yaw_change_deg"],3)}|')
selected=['A-seed42','B-seed42','B-seed43','B-seed44','C-seed42']
lines+=['','## 全窗口位置指标','', 'A和C表格各合并三个逐值相同的seed；机器可读统计仍分别保留9次结果。全部窗口覆盖完整。','', '|用例|窗口 s|净Δ(x,y) m|净位移 m|拟合(vx,vy) m/s|拟合速度 m/s|距起点最大 m|x/y活动范围 m|累计路程 m|','|---|---|---|---:|---|---:|---:|---|---:|']
for cid in selected:
 for w in stats[cid]['windows']:lines.append(f'|{cid}|{w["window_s"][0]}–{w["window_s"][1]}|{vec(w["xy_net_vector_m"])}|{f(w["xy_net_m"])}|{vec(w["xy_fit_slope_m_s"])}|{f(w["fit_drift_speed_m_s"])}|{f(w["max_distance_from_start_m"])}|{vec(w["xy_range_m"])}|{f(w["xy_path_length_m_50hz"])}|')
lines+=['','## 全窗口速度与姿态','', '|用例|窗口 s|世界均速(vx,vy) m/s|机体平均vx m/s|XY速度均值/RMS/P95 m/s|Δyaw °|后仰角范围 °|roll范围 °|','|---|---|---|---:|---|---:|---|---|']
for cid in selected:
 for w in stats[cid]['windows']:lines.append(f'|{cid}|{w["window_s"][0]}–{w["window_s"][1]}|{vec([w["vx_world"]["mean"],w["vy_world"]["mean"]])}|{f(w["vx_body"]["mean"])}|{vec([w["speed_xy"][k] for k in ["mean","rms","p95"]])}|{f(w["yaw_change_deg"],3)}|{vec([w["backward_lean_deg"][k] for k in ["min","max"]],3)}|{vec([w["roll_deg"][k] for k in ["min","max"]],3)}|')
lines+=['','## 分别回答诊断问题','',
f'1. **无附加噪声未形成有界站立。** A 的40–80 s与80–120 s净位移分别为{win("A-seed42",40,80)["xy_net_m"]:.4f}、{win("A-seed42",80,120)["xy_net_m"]:.4f} m，拟合速度分别为{100*win("A-seed42",40,80)["fit_drift_speed_m_s"]:.3f}、{100*win("A-seed42",80,120)["fit_drift_speed_m_s"]:.3f} cm/s。各后期20 s窗口仍有一致非零位置趋势，40秒不能称为已静止。小幅姿态/速度振荡叠加于持续位置迁移之上，不能用“往复平衡摆动”解释全部位移。',
'2. **本场景没有证据支持训练噪声使长期漂移增加。** B三个seed的净位移、拟合速度、平均平面速度均低于配对A；B也明显不静止。净位移下降部分伴随更大的航向改变，同时机体系向后平均速度和世界平均速度模也确实下降，因此不能仅用路径抵消解释全部差异，更不能把加噪声当作修复方案。噪声使瞬时速度和姿态波动增加，长期行为仍不满足静止描述。',
'3. **方向显示共同偏置，而非已证实的无偏随机游走。** A/B在40–120 s均为世界Δx<0、Δy>0，平均机体vx<0，航向变化均为负；B三seed的变化幅度不同但方向一致。这里只改变噪声种子、没有改变初始朝向或姿态，不能分辨偏置绑定于世界、机器人结构、控制参数还是策略，也不能把三个seed推广到所有场景。',
f'4. **运动后零指令仍持续到测试末尾。** C运动段平均机体vx={C[0]["vx_body"]["mean"]:.6f} m/s，跟踪误差均值={C[0]["vx_tracking_error"]["mean"]:.6f}、RMSE={C[0]["vx_tracking_error"]["rms"]:.6f} m/s；但yaw指令为0时航向累计变化{C[0]["yaw_change_deg"]:.3f}°，直行有转向偏差。停车60–80 s净位移{C[1]["xy_net_m"]:.6f} m，该窗口混合初始制动及后续向后运动。80–100、100–120 s仍分别移动{C[2]["xy_net_m"]:.6f}、{C[3]["xy_net_m"]:.6f} m，平均机体vx={C[2]["vx_body"]["mean"]:.6f}、{C[3]["vx_body"]["mean"]:.6f} m/s。后续位置变化没有逐渐结束，不能将整段归为短暂制动；没有批准的阈值与保持时长，不宣告“停车完成”。',
'5. **接口/时序证据支持的范围。** 54000次实际输入为195维，轮位置屏蔽、历史移位/首帧填充、上一动作、指令、q/dq映射、动作目标和PD限幅计算均经独立复算核验。A/B只改变批准的每帧噪声；原始状态随闭环分化而变化是正常现象。无法由这些事实直接定位漂移根因。',
'', '## 与Isaac参考的条件差异及后续建议','',
'Isaac参考为相同策略、seed42、平面、固定参数、零初速、无推扰/执行器延迟：无噪40–120 s净位移约0.00024 m，有噪约3.72 m、拟合约0.048 m/s。它不是验收线。本次无噪和有噪均与该现象关系不同，不能只比较最终数值便认定某个参数错误。',
'', '|项目|本次MuJoCo|可获得的Isaac证据|解释范围|','|---|---|---|---|','|脚关节Kp/Kd|36 / 1.8|归档匹配资产源码28 / 1.4|确认源码层配置差异；未取得该参考测试最终有效配置，不声称其必然实测值|','|摩擦|0.8（完整向量见配置）|归档对平面参考说明摩擦1|确认说明层条件不同，接触求解不可直接等同|','|初始base高度|0.700m；轮底约0.0283m|归档匹配资产默认0.725m；参考最终有效姿态未取得|本次完整保留部署初态，不能认定跨引擎初始落地过程相同|','|物理及低层执行|2ms；PD实际6/4ms交替；策略20ms；动作初次生效6ms后|参考称无执行器延迟，物理/PD实测时间记录未提供|“无附加延迟”不等于两端时序完全相同|','|噪声与随机数|幅度/通道按已确认继承配置；独立MT19937|训练噪声，Torch具体序列/运行时源码未取得|相同seed不构成逐随机数配对|','|质量/COM/惯量/接触求解|名义MJCF，完整编译值归档|完整参考运行编译值未提供|不能仅凭固定参数便宣称两端物理等价|',
'', '最有区分力的下一步（仅建议，未执行）：先取得Isaac参考的有效配置和初始状态/实际模型输入，做同一原始状态的跨端观测及动作核验；之后若另获批准，优先只改变脚PD至对应Isaac实测值做成对诊断，观察无噪漂移和转向是否同步变化。也可单独隔离控制/传感器调度延迟，或改变初始朝向区分世界偏置与机体偏置。不要一次同时改PD、摩擦、初态和时序；不建议凭本批次直接改奖励或部署参数。',
'', '## 验证、缺失项与交付','',
'- 9/9完整结束，0跌倒/非有限/程序异常终止，9个初始reset、0中途reset；数据可读、时间连续、形状正确。零原始动作clip，全部记录物理力矩均未达到所设上限；完整逐关节振荡与饱和率在statistics.json。全程有效地面接触仅左右wheel_collision；初始约落地前的一小段无接触，原始接触点/力保留，未将转轮本身当作漂移证据。',
'- 采集器保留全部物理子步ctrl、actuator_force、qfrc_actuator、接触和状态；没有物理子步降采样。q/dq原始值、默认值、关节/执行器映射、噪声前后与完整195维输入均可追溯。',
'- 未获取：训练机有效配置JSON/YAML、修改后flat原字节、训练安装版IsaacLab管理器及Torch RNG序列；继承无覆盖由用户确认。未执行中途reset，未验证异步交互线程墙钟时序；未采集每步最低碰撞几何离地高度，初态最低mesh高度已补充，逐步site高度可用。此矩阵未要求视频，因此未录制。',
'- 分析首次使用默认Python因缺matplotlib退出，改用已安装系统Python；随后修复JSON对NumPy布尔值的序列化。相关日志保留，均发生在离线分析阶段，没有重跑或追加闭环用例，也没有安装依赖。',
'- 所有原始JSON数字按实际float32/float64可往返精度保存；表格只为阅读格式化。图中A/C三个seed逐值重合，B按颜色区分；完整曲线在下图。',
'', '![全程位置、后80秒XY轨迹、速度、姿态及航向](overview.png)',
'', '报告：report.md；统计：statistics.json、comparison.json；输入/时序核验：input_validation.json、additional_audit.json、final_checks.json；实际配置：runtime_config.json；原始数据：cases/*/telemetry.jsonl.gz；命令：commands.json；源码及差异：tools/、config_snapshot/。manifest.json逐文件记录SHA-256，manifest.sha256校验清单本身。',
'', '使用技能原有sim2sim_report_bundle.py生成独立完整tar.gz；接收方younghit尚未确认资源哈希，全部实际模型mesh/texture资源均随包提供，不使用缓存省略。压缩包不包含策略权重或依赖环境；模型身份由完整哈希绑定。本机原始目录保留。本次没有Git提交/推送，没有远端传输或接收端验证。']
assert all(not any(w['action_clip_steps'] or max(w['torque_saturation_fraction_by_joint'])>0 for w in s['windows']) for s in stats.values())
(r/'report.md').write_text('\n'.join(lines)+'\n')
# Enumerate mesh dependencies from actual XML declarations, rather than suffix scanning.
xml=r/'config_snapshot/src/rl_sar_zoo/LW_description/mjcf/LW.xml';root=ET.parse(xml).getroot();resources=[str((xml.parent/root.find('compiler').get('meshdir','')/m.get('file')).relative_to(r)) for m in root.findall('asset/mesh') if m.get('file')]
work=pathlib.Path(plan['policy_root']).parent;write(work/'resources.json',resources)
if a.prepare_only:print('prepared report and validations, not yet sealed');raise SystemExit
files=[]
for path in sorted(r.rglob('*')):
 if path.is_file():
  assert path.suffix not in ['.onnx','.pt','.pth']
  files.append({'path':str(path.relative_to(r)),'sha256':sha(path),'size_bytes':path.stat().st_size})
manifest={'schema_version':1,'sealed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_id':'LFR','receiver_id':'younghit','model':plan['model'],'scene':{'path':plan['scene'],'sha256':sha(pathlib.Path(plan['scene']))},'cases':{k:s['result'] for k,s in stats.items()},'files':files,'self_hash':'manifest.sha256'}
write(r/'manifest.json',manifest);(r/'manifest.sha256').write_text(sha(r/'manifest.json')+'  manifest.json\n');print('sealed',len(files),'files',r)
