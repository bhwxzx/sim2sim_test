import argparse,pathlib,json,gzip,math,hashlib,subprocess,shutil,datetime,xml.etree.ElementTree as ET,difflib
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();out=a.run
assert not (out/'manifest.json').exists(),'Sealed evidence is immutable'
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while data:=f.read(1024*1024):h.update(data)
    return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
plan=read(out/'plan.json');statistics=read(out/'statistics.json');validation=read(out/'input_validation.json');runtime=read(out/'runtime_config.json')
repo=pathlib.Path(plan['repository']);tools=pathlib.Path(__file__).parent
# Read ONNX protobuf structural metadata without installing a Python ONNX runtime.
def varint(data,pos):
    value=shift=0
    while True:
        b=data[pos];pos+=1;value|=(b&127)<<shift
        if b<128:return value,pos
        shift+=7
def fields(data):
    pos=0
    while pos<len(data):
        key,pos=varint(data,pos);wire=key&7
        if wire==0:value,pos=varint(data,pos)
        elif wire==2:n,pos=varint(data,pos);value=data[pos:pos+n];pos+=n
        elif wire in [1,5]:n=8 if wire==1 else 4;value=data[pos:pos+n];pos+=n
        else:raise ValueError('unsupported protobuf wire type')
        yield key>>3,wire,value
model=pathlib.Path(plan['model']['path']);assert sha(model)==plan['model']['sha256'];mf=list(fields(model.read_bytes()))
opsets=[]
for tag,wire,data in mf:
    if tag==8:
        d={tag:v for tag,w,v in fields(data)};opsets.append({'domain':d.get(1,b'').decode(),'version':d.get(2)})
graph=next(data for tag,w,data in mf if tag==7);nodes=[];initializer_names=[]
for tag,w,data in fields(graph):
    if tag==1:
        d=list(fields(data));nodes.append({'name':next((v.decode() for t,w,v in d if t==3),''),'op':next((v.decode() for t,w,v in d if t==4),''),'inputs':[v.decode() for t,w,v in d if t==1],'outputs':[v.decode() for t,w,v in d if t==2]})
    if tag==5:initializer_names += [v.decode() for t,w,v in fields(data) if t==8]
graph_info={'opsets':opsets,'nodes':nodes,'initializer_names':initializer_names,'normalizer_named_initializers':[n for n in initializer_names if any(s in n.lower() for s in ['normal','running_mean','running_var'])],'method':'read protobuf structural fields only; TensorView/runtime interface independently recorded by ONNX Runtime'}
write(out/'onnx_graph_contract.json',graph_info);assert {'domain':'','version':17} in opsets
assert not graph_info['normalizer_named_initializers']
assert all(n['op'] not in ['BatchNormalization','InstanceNormalization','LayerNormalization'] for n in nodes)
cfg=runtime['configs']['stand'];names=statistics['stand']['joint_names']
assert all(c==cfg for c in runtime['configs'].values()),'Effective configurations differ across cases'
assert cfg['model_path']==str(model) and cfg['physics_timestep']==.002 and cfg['decimation']==4
assert cfg['model_inputs']==[{'dtype':'float32','name':'obs','shape':[1,410]}]
assert cfg['model_outputs']==[{'dtype':'float32','name':'actions','shape':[1,10]}]
assert cfg['gravity']==[0.,0.,-9.81]
ground=[g for g in cfg['geoms'] if g['body_id']==0]
assert len(ground)==1 and ground[0]['type']==0 and ground[0]['quat_body_wxyz']==[1.,0.,0.,0.]
assert ground[0]['name']=='floor'
# Preserve current collector bytes and changes relative to the inspected predecessor,
# rather than requiring paths from a previous package to execute this collector.
for source in tools.glob('*'):
    if source.is_file():shutil.copy2(source,out/'tools'/source.name)
shutil.copy2(repo/'.agents/skills/monitor-tune-isaaclab-training/scripts/sim2sim_report_bundle.py',out/'tools/sim2sim_report_bundle.py')
old=pathlib.Path('/home/lfr/sim2sim_test/sim2sim_diagnostics/20260911-115259-jointpos-ab/tools/collect.cpp')
(out/'tools/collector_changes.patch').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),(tools/'collect.cpp').read_text().splitlines(True),fromfile='inspected_previous_collector.cpp',tofile='parameterized_collect.cpp')))
libraries=[]
ldd=subprocess.check_output(['ldd',str(out/'tools/collect')],text=True)
for line in ldd.splitlines():
    words=line.split();path=next((pathlib.Path(x) for x in words if x.startswith('/') and pathlib.Path(x).is_file()),None)
    if path:libraries.append({'path':str(path.resolve()),'sha256':sha(path.resolve())})
write(out/'runtime_dependencies.json',{'ldd':ldd,'libraries':libraries})
# Additional endpoint and event inspection of existing evidence; no extra rollout.
details={}
initial_reference=None
for case in plan['cases']:
    cid=case['case_id'];c=runtime['configs'][cid];folder=out/'cases'/cid;result=read(folder/'result.json');v=validation[cid]
    assert result['status']=='completed' and result['control_records']==1000 and result['physics_steps']==10000
    assert v['rows']==1000 and v['physics_substeps']==10000 and v['pd_updates']==4000
    assert v['all_raw_values_finite'] and v['chronology'] and v['all_history_frames_exact'] and v['wheels_zero_all_history']
    assert v['native_joint_sensor_mapping_exact'] and v['raw_q_matches_sensor_evaluation_state']
    assert v['reset_steps']==[0] and v['terminal_steps']==[999]
    assert v['commands_exact'] and v['previous_action_exact'] and v['phase_mask_correct']
    assert v['q_target_max_error']==v['dq_target_max_error']==v['pd_candidate_max_error']==v['pd_bounded_max_error']==0
    vm=read(folder/'video_metadata.json');assert vm['validation_pass'] and vm['frames']==vm['decoded_frames']==1000 and vm['robot_visibility_all_sampled_frames_pass']
    assert abs(vm['duration_s']-20)<1e-6
    events={'first_saturation':None,'first_overspeed':None,'phase_samples':[],'one_second_heading_increments_deg':[],'max_external_xfrc':0.,'max_external_qfrc':0.,'active_contact_pairs':{}}
    geom_names={g['id']:g['name'] for g in c['geoms']}
    heading={};last_yaw=None;unwrapped=0;prior=0
    initial_samples=[]
    with gzip.open(folder/'telemetry.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);q=r['pre']['base_quaternion_wxyz_body_to_world'];w,x,y,z=q;yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
            if r['control_step']<100:initial_samples.append([r['pre']['qpos_mujoco'],r['pre']['qvel_mujoco'],r['model_input'],r['model_raw_action']])
            if last_yaw is None:unwrapped=yaw
            else:unwrapped+=math.atan2(math.sin(yaw-last_yaw),math.cos(yaw-last_yaw))
            last_yaw=yaw
            if r['control_step']%50==0:heading[r['control_step']//50]=math.degrees(unwrapped)
            events['max_external_xfrc']=max(events['max_external_xfrc'],*map(abs,r['pre']['xfrc_applied_world']))
            events['max_external_qfrc']=max(events['max_external_qfrc'],*map(abs,r['pre']['qfrc_applied']))
            if r['control_step'] in [0,99,100,101,749,750,751]:events['phase_samples'].append({'step':r['control_step'],'episode_time':r['episode_time'],'command':r['command_in_observation'],'actual_phase':r['current_frame'][39:41]})
            for sub in r['physics_substeps']:
                for contact in sub['contacts']:
                    if contact['efc_address']>=0:
                        pair=' / '.join(sorted([geom_names[contact['geom1']],geom_names[contact['geom2']]]))
                        events['active_contact_pairs'][pair]=events['active_contact_pairs'].get(pair,0)+1
                if events['first_saturation'] is None:
                    ids=[i for i,(x,lim) in enumerate(zip(sub['actuator_force'],c['torque_limits'])) if abs(x)>=lim*(1-1e-7)]
                    if ids:events['first_saturation']={'from_sim_time':sub['from_sim_time'],'actuator_indices':ids,'forces':sub['actuator_force']}
                if events['first_overspeed'] is None:
                    ids=[i for i,(mapping,lim) in enumerate(zip(c['joint_mapping_evidence'],statistics[cid]['joint_velocity_reference_limits_rad_s'])) if abs(sub['post_qvel'][mapping['mj_dof_adr']])>lim]
                    if ids:events['first_overspeed']={'post_sim_time':sub['to_sim_time'],'joint_names':[names[i] for i in ids],'dq_rad_s':[sub['post_qvel'][c['joint_mapping_evidence'][i]['mj_dof_adr']] for i in ids]}
    events['one_second_heading_increments_deg']=[{'window_s':[t,t+1],'delta_deg':heading[t+1]-heading[t]} for t in range(2,15) if t+1 in heading]
    assert events['max_external_qfrc']==events['max_external_xfrc']==0
    if initial_reference is None:initial_reference=initial_samples
    else:assert initial_samples==initial_reference,'Initial zero-command trajectories differ'
    events['initial_two_second_state_input_action_equal_across_cases']=True
    details[cid]=events
write(out/'event_details.json',details)
def f(x,n=5):return 'unavailable' if x is None else f'{x:.{n}f}'
def table(headers,rows):return '|'+ '|'.join(headers)+'|\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('|'+ '|'.join(map(str,row))+'|\n' for row in rows)
main=[];stop=[];whole=[];allwindows=[]
for cid,s in statistics.items():
    w=s['windows'][2];end=s['windows'][4]
    main.append([cid,w['samples'],f(w['vx']['mean']),f(w['vx_tracking_error']['rms']),f(w['vy']['mean']),f(w['vy_tracking_error']['rms']),f(w['wz_body']['mean']),f(w['wz_body_tracking_error']['rms']),f(w['wz_world']['mean']),f(w['yaw_delta_deg'],3),f(w['lean']['mean'],3)+' ± '+f(w['lean']['std'],3)])
    stop.append([cid,f(end['vx']['mean']),f(end['vx_tracking_error']['rms']),f(end['planar_speed']['mean']),f(end['net_planar_displacement_m']),f(end['planar_path_length_m']),f(end['yaw_delta_deg'],3),f(end['lean']['mean'],3)])
    whole.append([cid,s['whole_run']['any_torque_saturation_physics_count'],s['whole_run']['any_joint_overspeed_physics_count'],s['whole_run']['abnormal_ground_contact_physics_count'],f(max(s['whole_run']['max_abs_dq_rad_s_by_joint']),3)])
    for w in s['windows']:
        allwindows.append([cid,f"{w['window_s'][0]}–{w['window_s'][1]}",w['samples'],f(w['vx']['mean']),f(w['vy']['mean']),f(w['wz_body']['mean']),f(w['wz_world']['mean']),f(w['yaw_delta_deg'],3),f(w['lean']['mean'],3)+' ± '+f(w['lean']['std'],3),f(w['lean']['p5'],3)+' / '+f(w['lean']['p95'],3),f(w['roll']['mean'],3)+' ± '+f(w['roll']['std'],3),f(w['net_planar_displacement_m']),f(w['any_torque_saturation_physics_fraction']*100,3),f(w['any_joint_overspeed_physics_fraction']*100,3)])
commands=read(out/'commands.json');command_text='\n'.join(x['shell_command'] for x in commands)
maprows=[[m['joint_name'],m['policy_index'],m['config_mapping'],m['mj_joint_id'],m['mj_qpos_adr'],m['mj_dof_adr'],m['position_sensor_adr']] for m in cfg['joint_mapping_evidence']]
contract_rows=[
['接口','匹配','ONNX Runtime 实际查询 obs float32[1,410] → actions float32[1,10]；文件结构确认 opset17、静态 batch1'],
['历史布局','匹配','全部6000个实际TensorView输入、60000个重叠历史帧，旧→新，最新在末尾；初始10帧重复当前首帧，随后每周期移位一次'],
['关节位置/轮屏蔽','匹配','q-default_q；10帧各自17、18列严格为0；其余8关节逐样本等于当前原始角度减default；没有屏蔽左脚'],
['关节速度','匹配','具名qpos/qvel和传感器映射核验；dq×0.05，轮速度保留；缩放误差0'],
['机身角速度','匹配','实际机体系传感器值×0.25，误差0；机体系/世界系/航向变化分开保存'],
['重力和坐标','匹配到float32精度','body→world wxyz，body+X前向/+Z向上，R.T×[0,0,-1]；输入复算最大误差小于2e-7'],
['命令','匹配','计划值、控制器实际值、输入6:9逐样本一致；无交互覆盖'],
['previous action','匹配','上一模型输出经过原有±100动作裁剪、尚未scale/default偏置；初始为0；全部样本精确一致'],
['相位启用条件','匹配','完整三维命令范数>0.1；纯左右转±0.5时相位启用，零命令时[0,0]'],
['相位与episode时间','不一致，原状保留','时钟每次推理前先加0.02×1.25；相对本次记录episode时间约提前0.02秒，即9°相位。运动样本与指定公式最大分量误差0.15643038'],
['相位隐藏时钟','已确认','零命令屏蔽期间时钟仍继续；在2秒启用时没有重新置零；实际float32递推复算误差≤5.96e-8'],
['reset后历史/相位','未验证；现有流程不足','部署R只mj_resetDataKeyframe+mj_forward；历史/相位在策略activation generation变化时才重置，没有现成的联合reset调用。8秒用例未执行'],
['归一化','未新增，符合关闭设置','部署没有额外观测归一化；导出图无normalizer参数或归一化算子，包含完整actor输入组合，不在外部重复拼接'],
['噪声','条件不同，如实保留','部署不加观测噪声；训练非轮位置有噪声。未为本次临时加噪；轮输入始终0'],
['动作处理','匹配部署配置','前8维位置scale=.25加default，后2维轮速度scale=1；保留raw/clip/scale/target/PD/实际ctrl，目标及PD复算误差0']]
stand_x=sum(w['net_displacement_world_m'][0] for w in statistics['stand']['windows'][:2])
report=f'''本轮完全平地 MuJoCo 评估完成 **6/6个20秒用例**，共120秒、6000个策略步、60000个物理子步，无跌倒、非有限值或提前终止。申请的8秒闭环reset检查因现有部署缺少联合reset流程，按用户批准的方案标记 **not_executed**，没有用离线张量清零代替，也没有消耗剩余预算追加测试。

**主要结果。** 零指令站立并非全程不漂：前5秒向世界+X位移约 **{stand_x:.3f} m**，随后5–15秒平均vx为 **+0.000423 m/s**，净平面位移约4.78 mm；该稳态后仰角 **−1.463°±0.084°**，即轻微前倾，没有持续站立后仰的证据。初始位移/偏航是这次明确定义的home_leg启动流程下实测，不能用后段接近零的速度掩盖。

前进/后退主运动窗的实际vx分别 **+0.2804/−0.1628 m/s**，低于±0.4命令幅度。**直行停车后仍有显著残余前移**：17–20秒平均vx **+0.08554 m/s**，净平面位移 **0.26341 m**，平均后仰 **−4.703°**（前倾）；而后退停止与前进转向停止后的同窗净位移约1.30 mm和0.125 mm。这种命令历史差别尚无独立干预证明根因。

纯左/右转向确实产生世界航向变化，2–5秒分别+26.368°/−21.577°，但5–15秒仅 **+6.275°/−2.135°**，持续响应很弱；前进转向同窗 **+28.729°**。不能将局部wz均值很小误报为“完全没有转向”，也不能把启动时转角当作持续跟踪成功。逐秒航向增量在[event_details.json](event_details.json)。

**身份和边界。** 部署端host_id **LFR**；接收端ID younghit。本次使用归档 `LW/leg_loco/2026-09-11-16-56-27`，归档提交 `153c380b830ee3a44b1cbac1d3d8acf6d7557fc5` 已核对。任务 RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0；AMPROAPPO / OnPolicyRunnerAmpROA；训练run 2026-09-11_16-56-27，model_50000.pt。

真正加载的模型：`{model}`；SHA-256 `{sha(model)}`，与指定ONNX完全一致。归档JIT也校验为`64b411b440898c15df2340ba68ccef135c68b07dfa71e92ce1c20005037195a1`，但本次未用JIT执行。模型权重只在隔离运行目录中保存，按既有报告契约不放入传输包；接收端应使用相同哈希模型独立重放实际输入。归档archive_manifest.json和策略说明.txt原样随包保存，其中训练电脑绝对路径是历史来源信息，不作为本机缺失故障，也不要求提供训练argv或checkpoint。

该检查点包含轮位置置零索引修复；**不包含**后续stand_still排除轮位置、转向专家、收紧跟踪std修改。不能把本次现象归因于这些尚未进入模型的修改。来源说明中的训练条件和Isaac参考不是本轮重新验证的训练实验。

部署仓库HEAD `{plan['git_head']}`；启动dirty状态见plan.json，保留leg_loco/wheel_loco模型修改、inspect-context-compactions和library未跟踪状态。此次仅新增独立采集/分析工具和隔离场景，没有修改生产观测、历史、动作、PD、机器人参数或模型，没有训练、安装依赖、硬件访问或自动Git提交/推送。

**场景、初态和实际时序。** 由当前scene.xml生成隔离完全平地场景，只删除六个前方台阶；无斜坡、无障碍物；floor无限水平平面，接触参数保持原值。机器人LW.xml、质量/惯量/COM、关节与执行器参数原样保留，场景diff在config_snapshot/flat_scene.patch。所有用例xfrc_applied和qfrc_applied均为0，无随机推扰。

物理 MuJoCo **{cfg['mujoco_version']}**，ONNX Runtime **{cfg['onnxruntime_version']}**，现有CPU推理后端；库路径/哈希在runtime_dependencies.json。实际物理dt **{cfg['physics_timestep']} s**；YAML名义PD=.005 s，物理边界实际 **.006/.004 s交替**，decimation=4、策略周期=.02 s。Isaac参考物理dt=.005 s、策略=.02 s；本次没有改本机步长。原程序float32周期为{cfg['policy_period_float32']!r}，相位累加按原精度运行。采集调度沿用此前已验证的同步诊断时序，未声称验证了交互程序的异步线程调度。

各用例新进程、srand(42)，无显式随机化。初态home_leg：base原点[0,0,.7] m，wxyz[1,0,0,0]，全部qvel为0；具名关节初始q见effective_config，直接进入原LegLocomotion Enter/Activate路径，无额外GetUp插值或站稳等待。策略首输出在t=0构建、下一次PD即t=.006开始消费；每个策略边界先使用前一输出做PD并发布状态，然后推理。不能把新目标当作同一边界已经施加的ctrl。

实际适配器传感器通常反映前一个物理步开始状态，即比policy边界积分后qpos/qvel落后.002秒；6000帧原始q/dq按该求值时间与全物理日志精确核对。pre/post基座原点、COM、世界/机体系速度分别命名。物理子步中force/contact/足端动力学量标注from_sim_time，积分后qpos/qvel标注to_sim_time；复制mjData重算的边界接触与真实物理子步接触另行区分。

终止判据：非有限输入/动作/qpos/qvel、MuJoCo警告、运行时安全锁定、base高度<.25 m、upright总倾斜>75°或base_collision接触；每例120秒采集墙钟超时，外层135秒兜底。此次未触发。总倾斜阈值不是pitch阈值。初始reset各1次，窗口0–2秒标记含初始reset，没有中途reset样本混入其它窗口。

**命令和精确执行入口。** stand全程0；其余用例0–2秒0，2–15秒分别[.4,0,0]、[−.4,0,0]、[0,0,.5]、[0,0,−.5]、[.4,0,.5]，15–20秒0。全部计划/实际/模型命令相同。工作目录`{repo}`；逐例精确argv与重定向如下，也保存在commands.json及process.json：

```bash
{command_text}
```

**实际输入契约核验。** 输入直接复制底层forwardInto收到的TensorView，发生在真正ONNX执行前；不是用状态重建或读取旧debug缓存。完整输入和完整当前帧在每条telemetry中，重建仅用于独立核验。所有结果见input_validation.json。

{table(['项目','结论','证据/差异'],contract_rows)}

相位的9°指的是**步态相位角**，不是机身姿态角。首个运动输入在t=2.00 s应按指定公式得到sin/cos约[0,−1]，实际约[−0.15643,−0.98769]，对应先推进一帧。屏蔽期间时钟仍累计。此差异可作为后续单因素验证候选，但本轮没有改动，不能直接断言它导致停车前漂或转向不足。

每帧顺序与单位：0:3 body角速度rad/s×.25；3:6无量纲单位重力投影；6:9 vx/vy m/s、wz rad/s×1；9:19关节位置rad相对default×1（轮置0）；19:29关节速度rad/s×.05；29:39 previous action无量纲；39:41 sin/cos。全帧原有clip±100；actual raw action也按原有各通道±100先裁剪再缩放。本轮无动作裁剪事件。

{table(['关节','策略索引','配置mapping','MJ joint id','qpos地址','qvel地址','位置sensor地址'],maprows)}

默认策略关节姿态rad：`{cfg['default_dof_pos']}`。腿Kp/Kd：hip/thigh/shank为90/3，foot为28/1.4；轮0/.5。腿目标q=default_q+.25×clipped_action；轮dq=1×clipped_action，轮q保持default、Kp=0。PD候选τ=τff+Kp(qtarget−qsensor)+Kd(dqtarget−dqsensor)，按原有力矩上下限裁剪后写入ctrl；实际gear/ctrlrange/forcerange及所有body/joint参数在runtime_config.json和每例effective_config.json。freejoint frictionloss六自由度均.2，标量关节damping=.01、armature=.01、frictionloss=.2，均保留。

**5–15秒主运动窗口。** 各500个推理前样本，标准差为总体std，P5/P95为原始样本分位数；RMSE针对实际进入模型的命令。vx/vy是基座自由关节原点速度在机体系的分量；wz_body与wz_world是角速度向量在不同轴上的分量；世界航向由原始四元数ZYX解算并展开，两端t=5/15取差。它们不能混用。详细heading欧拉角导数及对应误差也保存在统计JSON。

{table(['用例','N','vx均值','vx RMSE','vy均值','vy RMSE','wz body均值','wz body RMSE','wz world均值','净航向°','后仰均值±std°'],main)}

**17–20秒停车残余运动。** 各150个样本；stand行是持续零命令的参照，不是停车动作。平面速度/位移使用世界xy；累计路径保留往复运动。前进停止是本轮最突出的残余运动问题，没有因其它停车用例较稳而忽略它。

{table(['用例','vx均值m/s','vx RMSE','平面速度均值m/s','净平面位移m','累计路径m','净航向°','后仰均值°'],stop)}

**所有规定窗口。** 依次100、150、500、100、150个样本，合计每例1000。净位移使用窗口两端位置；姿态统计左闭右开。后仰正值/前倾负值来自 `degrees(asin(clamp(-projected_gravity_b.x,-1,1)))`；raw四元数、ZYX pitch/roll另存，未使用roll或总倾斜替代pitch。与四元数和单位重力独立计算一致。

{table(['用例','秒','N','vx均值','vy均值','wz body','wz world','Δyaw°','后仰均值±std°','后仰P5/P95°','roll均值±std°','净xy位移m','饱和子步%','超速子步%'],allwindows)}

完整各窗vx/vy/wz均值、std、P5/P95、RMSE、平面速度、位移向量、累计路径、足端高度/接触、逐关节q波动/dq RMS/动作波动及饱和比例见statistics.json，没有降采样、平滑或提前舍入；表格小数位只用于阅读。

**力矩、关节超速和接触。** 每例10000个物理子步，实际actuator_force是否达到原配置上限按数值容差1e-7计数。零指令和左转各有8个饱和子步，其余0；不能因此宣称所有用例全程未饱和。动作裁剪全为0。速度超限相对于本机LW.urdf具名参考：hip/thigh/shank20 rad/s、foot10 rad/s、wheel33 rad/s；MJCF没有直接执行这些速度上限，所以这是**参考超速统计，不等于运行时触发保护，也不证明与Isaac软限完全一致**。

{table(['用例','任一力矩饱和子步数','任一关节超速子步数','非脚几何体接地子步数','最大关节|dq| rad/s'],whole)}

超速集中于脚关节；启动、运动或停车阶段的时间分别保留在统计窗口和event_details首次事件中。前进、前进转向的5–15秒仍有超速，不能全部归为初始化。没有记录到非脚几何体接地；这是具名接触的有限观察，不是所有接触行为都正常的证明。逐物理步记录完整接触对/力，足端site位置/速度/距平面高度可用；site高度不是脚底最小间隙，脚底最小几何距离未单独计算。

![完整原始时间序列总览](overview.png)

**与Isaac参考的关系和解释限度。** 指定同checkpoint的Isaac5–15秒参考为stand vx+.146 m/s、后仰−2.71°；前进+.271、后退−.156 m/s；纯左/右转净航向+9.90°/−15.50°；前进转向+49.59°、vx+.291 m/s。MuJoCo的前进/后退跟踪幅度与参考相近，稳态站立前漂小得多、纯转向和前进转向角度偏小。但Isaac保留其他随机化/观测噪声，本机启动、物理/PD时序、相位边界也不同，**不是严格同条件对照或通过阈值**。

已证实的是具体轨迹、输入轮屏蔽/历史正确、相位时间偏移、初始前移、直行停车残余前移、持续转向不足及瞬时脚关节参考超速。仍不能排除训练零命令行为、运动历史、相位时差、启动过程、噪声/物理差异分别贡献。本轮未执行新旧策略A/B或参数干预，不能断言任何单项为根因，也不能把单seed结果推广到所有场景。

下一步建议仅作建议：优先单独验证相位时间边界对直行停车及纯转向的影响，保持其它配置和噪声不变；另行批准定义并实现真正联合reset后，再做8秒闭环reset核验。若需要严格Isaac比较，应先固定启动及随机化条件，再进行同窗对照。本轮没有执行这些补测。

**缺失证据与视频。** reset检查未执行，故不存在可交付的中途reset前后闭环输入/动作，也无法确认物理reset后的历史污染程度；只能依据源码指出R不触发联合重置。初始进程历史填充和previous action已逐帧核验。ONNX接口没有显式recurrent state/reset mask，不适用。没有新的Isaac运行、硬件证据、独立脚底最小间隙；完整三维足端site速度/高度、接触和实际力矩均已采集。

六段side_view.mp4均为**离线遥测回放**：使用现有MuJoCo3.7.0仅重建显示状态，不调用mj_step或策略；物理采集仍3.2.7。每策略步pre状态一帧，50fps、20秒，含时间/命令/前方标识；yaw跟随侧视。6000帧解码数量与时长核验通过，每秒分割检查确认机器人入镜且未触边，并保存每例首帧/10秒/末帧供复核。视频不是闭环实时渲染，未凭画面估计姿态。

**交付与封存。** 完整本地目录 `{out}`；传输包 `{out}.tar.gz`，SHA-256旁文件 `{out}.tar.gz.sha256`。仅一个report.md；每例原始telemetry.jsonl.gz、console/process/result、视频，配置/源代码/diff、模型接口、统计和哈希清单一并保留。资源列表resources.json置于封存目录外，仅列实际XML依赖mesh/texture。尚未收到younghit确认的inventory及其SHA-256，故使用技能sim2sim_report_bundle.py生成一个**完整.tar.gz包，无接收端缓存依赖**，没有删视频、遥测或拆分传输包。本机打包和字节验证不代表已传输或younghit已验证接收。

仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
'''
(out/'report.md').write_text(report)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==plan['git_head']
assert subprocess.check_output(['git','status','--short'],cwd=repo,text=True)==plan['git_status']
for item in plan['source_files']:assert sha(pathlib.Path(item['path']))==item['sha256'],item['path']
robot=out/'config_snapshot/src/rl_sar_zoo/LW_description/mjcf/LW.xml';xml=ET.parse(robot)
resources=[]
for node in list(xml.findall('.//asset/mesh'))+list(xml.findall('.//asset/texture')):
    if node.get('file'):
        path=robot.parent/('assets' if node.tag=='mesh' else '')/node.get('file');assert path.is_file();resources.append(str(path.relative_to(out)))
resource_path=pathlib.Path(plan['policy_root']).parent/'resources.json';write(resource_path,sorted(set(resources)))
checks={'schema_version':1,'completed_cases':6,'early_terminated_cases':0,'failed_cases':0,'not_executed_cases':['reset-check'],'actual_simulation_seconds':120,'authorized_maximum_seconds':128,'policy_steps':6000,'physics_steps':60000,'pd_updates':24000,'all_integrity_checks_pass':True,'full_training_input_contract_pass':False,'known_semantic_differences':['phase advanced one policy period relative to recorded episode time','no deployment observation noise; upstream noise retained'],'midrun_reset_validation':'unavailable','protected_sources_unchanged':True,'repository_head_unchanged':True,'repository_dirty_status_unchanged':True,'receiver_cache_required':False,'resource_inventory_received':False,'resource_list_external':str(resource_path)}
write(out/'final_checks.json',checks)
print('Prepared report, validations and resource list',flush=True)
if a.prepare_only:raise SystemExit(0)
files=[]
for path in sorted(out.rglob('*')):
    assert not path.is_symlink(),path
    if path.is_file():
        assert path.suffix not in ['.onnx','.pt','.pth'],path
        if path.suffix=='.json':read(path)
        files.append({'path':str(path.relative_to(out)),'sha256':sha(path),'size_bytes':path.stat().st_size})
manifest={'schema_version':1,'sealed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host_id':'LFR','receiver_id':'younghit','model':plan['model'],'scene':{'actual_path':plan['scene'],'sha256':sha(pathlib.Path(plan['scene'])),'flat':True},'report':'report.md','commands':'commands.json','statistics':'statistics.json','validation':'final_checks.json','cases':{cid:read(out/'cases'/cid/'result.json') for cid in [*[c['case_id'] for c in plan['cases']],'reset-check']},'files':files,'self_hash':'manifest.sha256; manifest excluded from its own payload list'}
write(out/'manifest.json',manifest);(out/'manifest.sha256').write_text(sha(out/'manifest.json')+'  manifest.json\n')
print(json.dumps({'sealed_run':str(out),'resource_list':str(resource_path),'payload_files':len(files)}),flush=True)
