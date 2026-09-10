from pathlib import Path
import gzip,json,math,hashlib,subprocess,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
out=Path(__file__).resolve().parent.parent
def write(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rotation(q):
    w,x,y,z=np.asarray(q,dtype=np.float64)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def stats(a):
    a=np.asarray(a,dtype=np.float64)
    if not len(a):return None
    return dict(n=len(a),mean=float(a.mean()),std_population=float(a.std()),p5=float(np.percentile(a,5)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()))
base=json.loads((out/'baseline.json').read_text())
c=json.loads((out/'cases/stand-a01/effective_config.json').read_text())
b=next(x for x in c['bodies'] if x['name']=='base_link')
Ri=rotation(b['inertial_quat_wxyz']);b['inertia_tensor_in_body_frame']=(Ri@np.diag(b['principal_inertia'])@Ri.T).tolist()
isaac=['left_hip_joint','right_hip_joint','left_thigh_joint','right_thigh_joint','left_shank_joint','right_shank_joint','left_foot_joint','left_wheel_joint','right_foot_joint','right_wheel_joint']
deployed=[c['joint_names_config'][i] for i in c['joint_mapping']]
runtime={'collection_mode':'Independent diagnostic executable using repository LWRuntimeCore, FSM Leg Enter, ONNX runtime, observation buffer and MuJoCo adapter sources; not the interactive rl_sim_LW process.',
    'effective':c,'merged_policy_yaml':yaml.safe_load(c['merged_yaml']),
    'policy_identity':{'task':'RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0','run':'2026-09-04_11-16-35','runner':'OnPolicyRunnerAmpROA','models':base['models'],'checkpoint':base['checkpoint']},
    'timing':{'physics_s':c['physics_timestep'],'nominal_pd_s':0.005,'nominal_policy_s':0.02,'policy_decimation_in_pd_ticks':4,'physics_substeps_per_policy':10,
        'pd_execution_grid':'ceil(n*0.005/0.002)*0.002; physical intervals alternate 0.006, 0.004 s',
        'same_time_order':'PD/control cycle publishes state and applies previous available output; inference; physical stepping; new output first consumed at next PD tick.',
        'difference_from_interactive':'Interactive program uses independent wall-clock threads and GUI-paced physics. This diagnostic uses deterministic simulation-time ordering. No wall-clock jitter or real-time slowdown equivalence is claimed.',
        'float32_periods':'Exact float32 runtime values retained above; event schedule reads YAML decimal dt as double. No dt/decimation/XML option was changed.'},
    'reset':{'physics':'mj_resetDataKeyframe(home_leg), then mj_forward, as R handler in rl_sim_LW.cpp',
        'policy':'fresh process/core per case; invoke existing RLFSMStateRLLocomotion_Leg.Enter through StateController; first inference initializes history with first frame repeated 10 times',
        'startup_difference':'Direct policy FSM activation at reset pose; interactive Passive/GetUp operator path and its interpolation time are not exercised.',
        'mid_episode':'none; native R handler resets physics but does not by itself reset runtime history; not modified or exercised here',
        'seed':42,'seed_method':'std::srand(42) before reset; deterministic ONNX; no added randomization'},
    'disturbance':{'scheduled_force':'none','interactive_input':'not created; deterministic command hook used','interactive_noise_default':0.0,
        'source_impact_code':'commented out','observation_noise_macros':'disabled','forward_latency_macro':'disabled',
        'sensor_noise_fields':'jointactuatorfrc noise=0.01 retained in compiled model; full sensors and enableflags recorded; no independent stochastic sensor-noise calibration performed',
        'external_force_evidence':'all pre/post xfrc_applied and qfrc_applied recorded and checked'},
    'frames':{'world':'right-handed, +Z up; gravity [0,0,-9.81] m/s^2',
        'base':'base_link; freejoint transform is body-to-world wxyz; IMU site is identity relative to base_link',
        'forward':'body +X treated as policy forward, +Y left, +Z up; supported by deployment command convention and observed +vx displacement; no independent mesh-front calibration asset supplied',
        'visual':'base mesh attached to base_link with no source MJCF pose offset; compiled geom pose includes MuJoCo mesh centering/principal-axis transform, retained in effective.geoms',
        'pitch':'R=Rz(yaw) Ry(pitch) Rx(roll); raw pitch=asin(-R[2,0]); backward_lean=-pitch; roll=atan2(R[2,1],R[2,2]); report degrees',
        'gravity':'R_body_to_world.T @ normalize(world_gravity). Distinguish full precision current freejoint from float32 runtime IMU sensor quaternion.',
        'raw_telemetry_velocity_clarification':'Original pre/post.base_linear_velocity_world/body are mj_objectVelocity(mjOBJ_BODY) at body COM, not freejoint origin. Keep these immutable. Canonical base-origin linear velocities are in telemetry_derived.jsonl.gz (join by case_id/control_step). This offset is verified numerically as omega_world cross (R @ body_ipos).',
        'angular_velocity':'original pre/post.base_angular_velocity_world/body are explicit frames; policy angular velocity is float32 imu gyro in body/site frame'},
    'action':{'formula':'a=clip(raw_action, lower, upper); for nonwheels q_target=q_default+scale*a,dq_target=0; for wheels q_target=q_default,dq_target=scale*a; runtime inference PD estimate=Kp*(q_target-q_obs)+Kd*(dq_target-dq_obs); FSM sends feedforward_tau=0; at each PD tick candidate=feedforward+Kp*(q_target-q_sensor)+Kd*(dq_target-dq_sensor); ctrl[actuator_id]=clip(candidate,+/-torque_limits); MuJoCo motor uses gear=1 with ctrlrange.',
        'modes':'8 joints position PD; 2 wheels velocity PD (Kp=0). All XML actuators are motor torque controls.',
        'model_raw_output_order':deployed,'isaac_user_supplied_order':isaac,
        'deployed_index_for_each_isaac_joint':[deployed.index(n) for n in isaac],
        'warning':'Runtime mapping is preserved, not corrected. User supplied Isaac telemetry order differs; actual training action indexing/config was not locally available for independent verification.'},
    'observations':{'one_frame_dim':41,'model_input_shape':[1,410],'dtype':'float32',
        'fields':[{'slice':[0,3],'field':'angular_velocity','unit_before_scale':'rad/s','frame':'IMU/body','scale':c['ang_vel_scale']},
                  {'slice':[3,6],'field':'projected_gravity','unit':'dimensionless unit gravity','frame':'body','scale':1},
                  {'slice':[6,9],'field':'commands vx,vy,wz','unit_before_scale':'m/s,m/s,rad/s','frame':'body-command convention','scale':c['commands_scale']},
                  {'slice':[9,19],'field':'q-q_default; wheel positions forcibly represented as 0 by existing observation code','unit_before_scale':'rad','scale':c['dof_pos_scale']},
                  {'slice':[19,29],'field':'dq','unit_before_scale':'rad/s','scale':c['dof_vel_scale']},
                  {'slice':[29,39],'field':'previous clipped action; zero at activation','unit':'dimensionless raw policy action units','scale':1},
                  {'slice':[39,41],'field':'moving*sin(2*pi*phase), moving*cos(2*pi*phase)','unit':'dimensionless'}],
        'history':'time-major 10x41 oldest to newest; indices [9..0]; latest input slice [369:410]; first frame repeated 10 times, then insertion before every forward',
        'gait':'float32 phase += policy_period*gait_frequency before observation; wrap at 1; moving=norm(command)>0.1; phase advances during zero command, output sin/cos gated to zero',
        'clip':'all 41 values clipped to +/-100 before history insertion',
        'normalization':'runtime only scales/clips fields, no running-stat normalization. Archived matching JIT wrapper declares actor_obs_normalizer Identity; archive states training observation normalization disabled. Loaded ONNX internals independently enumerated in onnx_graph.json.',
        'recurrent_state':'external interface has only obs and actions; no explicit recurrent state or reset mask'},
    'telemetry':{'raw':'cases/<case>/telemetry.jsonl.gz','derived':'cases/<case>/telemetry_derived.jsonl.gz',
        'precision':'JSON round-trip decimal representation of float64 and float32 values, gzip lossless; no smoothing/rounding/downsampling',
        'control_step':'policy inference index, zero-based; each row contains all 4 PD callbacks and all 10 physics steps',
        'pre':'full-precision state at inference boundary before PD callback; read-only mj_forward on scratch copy computes coherent derived kinematics/contacts',
        'policy_state_float32':'exact float32 values read by existing adapter from original sensor data; evaluation is normally one physical substep before current qpos, except initial reset',
        'model_raw_action':'output copied immediately from underlying ONNX forward before runtime action clipping',
        'new_policy_targets':'new output computed this row, not necessarily applied at first PD callback of row',
        'low_level_updates':'each actual delivered PD target/ctrl with nominal and actual simulation times and applied policy frame',
        'physics_substeps':'each integration interval: ctrl and force from that interval; post_qpos/qvel at its end',
        'post':'state after all recorded physics substeps; last_dynamics forces belong to the final integration interval',
        'contact_force':'mj_contactForce on refreshed scratch data at recorded boundary; constraint diagnostic, not foot force sensor wrench',
        'foot_height':'foot site origin heights only; exact sole-bottom height not derived from mesh'},
    'termination':{'diagnostic':'nonfinite qpos/qvel; any MuJoCo warning; core terminal/fallback; base origin z<0.25m; upright-axis tilt>75deg; base collision contact; simulation time reset; program exception',
        'automatic_retry':False,'additional_mid_run_resets':False}}

# Extract graph metadata without importing uninstalled ONNX dependencies. A
# minimal protobuf wire reader is enough to enumerate GraphProto nodes/inputs.
def wire(data):
    i=0
    def varint():
        nonlocal i
        v=s=0
        while True:
            b=data[i];i+=1;v|=(b&127)<<s
            if b<128:return v
            s+=7
    while i<len(data):
        tag=varint();field,kind=tag>>3,tag&7
        if kind==0:value=varint()
        elif kind==2:
            n=varint();value=data[i:i+n];i+=n
        elif kind==1:value=data[i:i+8];i+=8
        elif kind==5:value=data[i:i+4];i+=4
        else:raise ValueError(kind)
        yield field,kind,value
model_bytes=Path(c['model_path']).read_bytes()
graph=next(v for f,k,v in wire(model_bytes) if f==7)
nodes=[]
for field,kind,value in wire(graph):
    if field!=1:continue
    node={'inputs':[],'outputs':[]}
    for f,k,v in wire(value):
        if f in (1,2,3,4,7) and k==2:
            if f==1:node['inputs'].append(v.decode())
            elif f==2:node['outputs'].append(v.decode())
            else:node[{3:'name',4:'op_type',7:'domain'}[f]]=v.decode()
    nodes.append(node)
write(out/'config_snapshot/onnx_graph.json',{'model_path':c['model_path'],'sha256':hashlib.sha256(model_bytes).hexdigest(),'nodes':nodes,
    'note':'Protobuf NodeProto input/output/op enumeration only, no weights copied. Runtime interface metadata is authoritative.'})
with zipfile.ZipFile('/home/lfr/policy_storage/LW/leg_loco/2026-09-04-11-16-35/policy.pt') as z:
    (out/'config_snapshot/jit_wrapper.txt').write_bytes(z.read('policy/code/__torch__.py'))

allresults={}; series={}; validations={}
for cid in ['stand-a01','walk-stop-a01']:
    directory=out/'cases'/cid
    with gzip.open(directory/'telemetry.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
    values={k:[] for k in ['time','lean','roll','vx_body','vx_world','vy_body','yaw','z','left_contact','right_contact','contact_count','actuator_saturation']}
    checks={'rows':len(rows),'input_finite':True,'output_finite':True,'input_shapes_correct':True,'history_shift_exact':True,'first_history_repeated':True,
        'previous_action_exact':True,'commands_exact':True,'target_max_error':0.,'observation_state_max_error':0.,'gravity_sensor_max_error':0.,
        'gravity_current_pose_max_difference':0.,'com_velocity_cross_product_max_error':0.,'max_abs_xfrc_applied':0.,'max_abs_qfrc_applied':0.,
        'physics_substeps':0,'low_level_updates':0,'max_policy_period_error':0.,'reset_rows':[],'termination_rows':[],'nonfoot_ground_contacts':[],
        'max_abs_raw_action':0.,'max_abs_clipped_action':0.}
    footids={side:next(g['id'] for g in c['geoms'] if g['name']==side+'_foot_collision') for side in ['left','right']}
    groundids={g['id'] for g in c['geoms'] if g['body_id']==0}
    prev=None;orig_vcomerr=[]; lowtimes=[];prev_action=np.zeros(10,dtype=np.float32)
    derived_path=directory/'telemetry_derived.jsonl.gz'
    if derived_path.exists():raise RuntimeError('refuse derived telemetry overwrite')
    with gzip.open(derived_path,'xt',compresslevel=6) as dest:
        for i,row in enumerate(rows):
            inp=np.array(row['model_input']['values'],dtype=np.float32).reshape(10,41);raw=np.array(row['model_raw_action'],np.float32)
            act=np.clip(raw,np.array(c['clip_actions_lower'],np.float32),np.array(c['clip_actions_upper'],np.float32))
            checks['max_abs_raw_action']=max(checks['max_abs_raw_action'],float(abs(raw).max()))
            checks['max_abs_clipped_action']=max(checks['max_abs_clipped_action'],float(abs(act).max()))
            checks['input_finite'] &= bool(np.isfinite(inp).all());checks['output_finite'] &= bool(np.isfinite(raw).all())
            checks['input_shapes_correct'] &= row['model_input']['shape']==[1,410]
            checks['previous_action_exact'] &= bool(np.array_equal(inp[-1,29:39],prev_action))
            checks['commands_exact'] &= bool(np.array_equal(inp[-1,6:9],np.array(row['planned_command'],np.float32)))
            if prev is None:checks['first_history_repeated']=bool(np.array_equal(inp,np.tile(inp[-1],(10,1))))
            else:
                checks['history_shift_exact'] &= bool(np.array_equal(inp[:-1],prev[1:]))
                checks['max_policy_period_error']=max(checks['max_policy_period_error'],abs(row['simulation_time']-rows[i-1]['simulation_time']-.02))
            prev=inp;prev_action=act
            default=np.array(c['default_dof_pos'],np.float32);scaled=act*np.array(c['action_scale'],np.float32)
            qt=default+scaled;qt[c['wheel_indices']]=default[c['wheel_indices']]
            vt=np.zeros(10,np.float32);vt[c['wheel_indices']]=scaled[c['wheel_indices']]
            checks['target_max_error']=max(checks['target_max_error'],float(abs(qt-row['new_policy_targets']['q']).max()),float(abs(vt-row['new_policy_targets']['dq']).max()))
            s=row['policy_state_float32'];pg_sensor=rotation(s['quaternion_wxyz_body_to_world']).T@np.array([0,0,-1.])
            checks['gravity_sensor_max_error']=max(checks['gravity_sensor_max_error'],float(abs(pg_sensor-inp[-1,3:6]).max()))
            expq=np.array(s['q'],np.float32)-default;expq[c['wheel_indices']]=0
            expq*=np.float32(c['dof_pos_scale']);expdq=np.array(s['dq'],np.float32)*np.float32(c['dof_vel_scale'])
            expw=np.array(s['angular_velocity_body'],np.float32)*np.float32(c['ang_vel_scale'])
            checks['observation_state_max_error']=max(checks['observation_state_max_error'],float(abs(expq-inp[-1,9:19]).max()),float(abs(expdq-inp[-1,19:29]).max()),float(abs(expw-inp[-1,:3]).max()))
            enriched={'case_id':cid,'control_step':row['control_step'],'simulation_time':row['simulation_time'],
                      'projected_gravity_recomputed_from_policy_imu_quaternion':pg_sensor.tolist(),'projected_gravity_policy_sensor_evaluation_time':s['sensor_state_evaluation_time']}
            for moment in ['pre','post']:
                state=row[moment];R=rotation(state['base_quaternion_wxyz_body_to_world']);vo=np.array(state['qvel_mujoco'][:3]);vb=R.T@vo
                pg=R.T@np.array([0,0,-1.]);w=np.array(state['base_angular_velocity_world'])
                vc=vo+np.cross(w,R@np.array(b['com_body']))
                checks['com_velocity_cross_product_max_error']=max(checks['com_velocity_cross_product_max_error'],float(abs(vc-state['base_linear_velocity_world']).max()))
                checks['max_abs_xfrc_applied']=max(checks['max_abs_xfrc_applied'],max(abs(x) for x in state['xfrc_applied_world']))
                checks['max_abs_qfrc_applied']=max(checks['max_abs_qfrc_applied'],max(abs(x) for x in state['qfrc_applied']))
                contacts={}
                for side,gid in footids.items():
                    cs=[ct for ct in state['contacts'] if gid in [ct['geom1'],ct['geom2']] and ({ct['geom1'],ct['geom2']}&groundids) and ct['efc_address']>=0]
                    contacts[side]={'contact':bool(cs),'points':len(cs),'sum_normal_force_N':sum(ct['force_contact_frame'][0] for ct in cs)}
                enriched[moment]={'sim_time':state['sim_time'],'base_origin_linear_velocity_world':vo.tolist(),'base_origin_linear_velocity_body':vb.tolist(),
                    'projected_gravity_recomputed':pg.tolist(),'foot_ground_contact':contacts,
                    'base_yaw_ZYX_deg':float(np.degrees(np.arctan2(R[1,0],R[0,0])))}
                if moment=='pre':
                    for key,v in [('time',state['sim_time']),('lean',state['backward_lean_deg']),('roll',state['roll_ZYX_deg']),('vx_body',vb[0]),
                                 ('vx_world',vo[0]),('vy_body',vb[1]),('yaw',enriched[moment]['base_yaw_ZYX_deg']),('z',state['base_position_world'][2]),
                                 ('left_contact',contacts['left']['contact']),('right_contact',contacts['right']['contact']),('contact_count',len(state['contacts']))]:values[key].append(v)
                    checks['gravity_current_pose_max_difference']=max(checks['gravity_current_pose_max_difference'],float(abs(pg-inp[-1,3:6]).max()))
                    for ct in state['contacts']:
                        ids={ct['geom1'],ct['geom2']}
                        if ids&groundids and not ids&set(footids.values()):checks['nonfoot_ground_contacts'].append({'step':i,'geom_ids':sorted(ids)})
            saturation=[]
            ranges=np.array([a['ctrlrange'] for a in c['actuators']])
            for sub in row['physics_substeps']:
                force=np.array(sub['actuator_force']);saturation.append(np.isclose(abs(force),ranges[:,1],atol=1e-6,rtol=0))
            values['actuator_saturation'].append(np.any(saturation,axis=0).astype(int).tolist())
            checks['physics_substeps']+=len(row['physics_substeps']);checks['low_level_updates']+=len(row['low_level_updates'])
            lowtimes += [u['actual_sim_time'] for u in row['low_level_updates']]
            if row['reset']:checks['reset_rows'].append(i)
            if row['terminated']:checks['termination_rows'].append(i)
            dest.write(json.dumps(enriched,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    checks['pd_actual_intervals_rounded_for_summary']=sorted(set(np.round(np.diff(lowtimes),9).tolist()))
    checks['all_checks_pass']=all(checks[k] for k in ['input_finite','output_finite','input_shapes_correct','history_shift_exact','first_history_repeated','previous_action_exact','commands_exact']) and checks['target_max_error']<1e-6 and checks['observation_state_max_error']<1e-6 and checks['gravity_sensor_max_error']<1e-6 and checks['com_velocity_cross_product_max_error']<1e-10
    validations[cid]=checks
    values={k:np.asarray(v) for k,v in values.items()};series[cid]=values
    windows=[(0,2),(2,5),(5,10),(10,30)] if cid=='stand-a01' else [(5,10),(15,20),(20,22),(22,30),(30,40)]
    allresults[cid]={'case_result':json.loads((directory/'result.json').read_text()),'windows':[],
        'raw_action_max':checks['max_abs_raw_action'],'saturation_policy_step_fraction_by_actuator':values['actuator_saturation'].mean(axis=0).tolist(),
        'final_xy':rows[-1]['post']['base_position_world'][:2],'yaw_deg':stats(values['yaw'])}
    for lo,hi in windows:
        mask=(values['time']>=lo-1e-8)&(values['time']<hi-1e-8)
        w={'window_s':[lo,hi],'reset_handling':'initial reset at t=0 flagged; no mid-window reset' if lo==0 else 'no reset in window',
           **{k:stats(values[k][mask]) for k in ['lean','roll','vx_body','vx_world','vy_body','yaw','z']},
           'left_contact_fraction':float(values['left_contact'][mask].mean()),'right_contact_fraction':float(values['right_contact'][mask].mean()),
           'both_contact_fraction':float((values['left_contact'][mask]&values['right_contact'][mask]).mean()),
           'no_foot_contact_fraction':float((~values['left_contact'][mask]&~values['right_contact'][mask]).mean())}
        allresults[cid]['windows'].append(w)
write(out/'runtime_config.json',runtime);write(out/'statistics.json',allresults);write(out/'validation.json',validations)
fig,axs=plt.subplots(5,2,figsize=(14,14),sharex='col')
for col,(cid,v) in enumerate(series.items()):
    t=v['time'];axs[0,col].plot(t,v['lean'],lw=.75,label='MuJoCo backward lean (+)');axs[0,col].axhline(2.28,color='grey',ls=':',label='Isaac initial stand ref 2.28 deg')
    axs[0,col].set(title=cid,ylabel='Backward lean (deg)');axs[0,col].legend(fontsize=8)
    axs[1,col].plot(t,v['vx_body'],lw=.7,label='base-origin vx, body');axs[1,col].plot(t,v['vx_world'],lw=.7,label='base-origin vx, world')
    command=np.where((t>=10-1e-8)&(t<20-1e-8),.4,0) if col else np.zeros(len(t));axs[1,col].step(t,command,where='post',ls='--',label='vx command');axs[1,col].set_ylabel('Velocity (m/s)');axs[1,col].legend(fontsize=8)
    axs[2,col].plot(t,v['roll'],lw=.7,label='roll');axs[2,col].plot(t,v['yaw'],lw=.7,label='yaw');axs[2,col].set_ylabel('Orientation (deg)');axs[2,col].legend(fontsize=8)
    axs[3,col].step(t,v['left_contact'].astype(float),where='post',label='left');axs[3,col].step(t,v['right_contact'].astype(float)*.85,where='post',alpha=.7,label='right (offset display)');axs[3,col].set_ylabel('Foot ground contact');axs[3,col].legend(fontsize=8)
    axs[4,col].plot(t,v['actuator_saturation'].sum(axis=1),lw=.7);axs[4,col].set(ylabel='Actuators saturated\nin any physics substep',xlabel='Simulation time (s)')
    for ax in axs[:,col]:
        ax.grid(alpha=.2)
        if col:
            ax.axvspan(10,20,alpha=.08,color='green');ax.axvline(20,color='grey',lw=.5)
fig.suptitle('LW Flat AMP-ROA | unchanged policy/config | independent deterministic diagnostic\nNo smoothing; all 20 ms samples plotted; base-origin velocity; raw evidence retained',fontsize=12)
fig.tight_layout(rect=(0,0,1,.955));fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
print(json.dumps({'validation':validations,'windows':allresults},ensure_ascii=False,indent=2))
