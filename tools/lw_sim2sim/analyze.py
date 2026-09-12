import argparse,pathlib,json,gzip,math,hashlib,xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();out=a.run
plan=json.loads((out/'plan.json').read_text());windows=[(0,2),(2,5),(5,15),(15,17),(17,20)]
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def stats(x):
    x=np.asarray(x,dtype=float)
    if not len(x):return None
    return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),p5=float(np.percentile(x,5)),p95=float(np.percentile(x,95)),min=float(x.min()),max=float(x.max()),rms=float(np.sqrt(np.mean(x*x))))
def Rof(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def finite(x):
    if isinstance(x,float):return math.isfinite(x)
    if isinstance(x,list):return all(map(finite,x))
    if isinstance(x,dict):return all(map(finite,x.values()))
    return True
def error(v,key,x):v[key]=max(v.get(key,0.),float(np.max(np.abs(x))))
urdf=ET.parse(out/'config_snapshot/src/rl_sar_zoo/LW_description/urdf/LW.urdf')
velocity_limits={j.get('name'):float(j.find('limit').get('velocity')) for j in urdf.findall('joint') if j.find('limit') is not None}
statistics={};validations={};allseries={};configs={}
for case in plan['cases']:
    cid=case['case_id'];folder=out/'cases'/cid;cfg=json.loads((folder/'effective_config.json').read_text());configs[cid]=cfg
    result=json.loads((folder/'result.json').read_text());mapping=cfg['joint_mapping_evidence'];names=[m['joint_name'] for m in mapping]
    qi=[m['mj_qpos_adr'] for m in mapping];vi=[m['mj_dof_adr'] for m in mapping];sensors=[m['position_sensor_adr'] for m in mapping]
    vmax=np.array([velocity_limits[n] for n in names]);limits=np.array(cfg['torque_limits']);actmap=[next(x['id'] for x in cfg['actuators'] if x['trnid'][0]==m['mj_joint_id']) for m in mapping]
    geom={x['id']:x for x in cfg['geoms']};foot={s:next(x['id'] for x in cfg['geoms'] if x['name']==s+'_foot_collision') for s in ['left','right']}
    history=[];previous=np.zeros(10,np.float32);phase=np.float32(0);last_t=-1.;last_physics=0;low_times=[]
    v=dict(rows=0,all_raw_values_finite=True,chronology=True,all_history_frames_exact=True,wheels_zero_all_history=True,previous_action_exact=True,commands_exact=True,native_joint_sensor_mapping_exact=True,reset_steps=[],terminal_steps=[],no_added_noise=True,raw_action_clip_steps=0,physics_substeps=0,pd_updates=0,phase_mask_correct=True,phase_runtime_recurrence_max_error=0.,phase_episode_contract_max_error=0.,gravity_quaternion_max_error=0.,raw_q_matches_sensor_evaluation_state=True)
    data={k:[] for k in ['t','pos','vx','vy','vx_world','vy_world','planar_speed','wz_body','wz_world','yaw','heading_rate','lean','roll','cmd','raw_action','q','dq','clip','torque_fraction','left_contact','right_contact','left_height','right_height','phase_error']}
    physics={k:[] for k in ['t','forces','dq','sat','overspeed','abnormal_contact','left_contact','right_contact']};last_pre_physics=None
    with gzip.open(folder/'telemetry.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);v['all_raw_values_finite'] &= finite(row)
            if row.get('model_input') is None:continue
            i=row['control_step'];t=row['nominal_policy_time'];inp=np.array(row['model_input']['values'],np.float32).reshape(10,41);frame=inp[-1]
            raw=np.array(row['model_raw_action'],np.float32);clipped=np.clip(raw,np.array(cfg['clip_actions_lower'],np.float32),np.array(cfg['clip_actions_upper'],np.float32))
            s=row['pre'];o=row['policy_state_float32'];q=np.array(o['q'],np.float32);dq=np.array(o['dq'],np.float32);default=np.array(row['default_q'],np.float32)
            v['chronology'] &= i==v['rows'] and row['simulation_time']>last_t and abs(t-row['simulation_time'])<1e-8;last_t=row['simulation_time'];v['rows']+=1
            if row['reset']:history=[frame.copy() for _ in range(10)];v['reset_steps'].append(i)
            else:history=history[1:]+[frame.copy()]
            if row['terminated']:v['terminal_steps'].append(i)
            v['all_history_frames_exact'] &= np.array_equal(inp,np.stack(history));v['wheels_zero_all_history'] &= bool(np.all(inp[:,[17,18]]==0))
            v['previous_action_exact'] &= np.array_equal(frame[29:39],previous);previous=clipped.copy()
            v['commands_exact'] &= np.array_equal(frame[6:9],np.array(row['planned_command'],np.float32)) and row['planned_command']==row['actual_applied_command']
            v['native_joint_sensor_mapping_exact'] &= np.array_equal(q,np.array(row['sensor_data_before_inference_float64'])[sensors].astype(np.float32)) and row['joint_mapping']==mapping
            if last_pre_physics is not None:v['raw_q_matches_sensor_evaluation_state'] &= np.array_equal(q,np.array(last_pre_physics[0])[qi].astype(np.float32)) and np.array_equal(dq,np.array(last_pre_physics[1])[vi].astype(np.float32))
            pos=(q-default)*np.float32(cfg['dof_pos_scale']);pos[cfg['wheel_indices']]=0
            error(v,'joint_pos_max_error',frame[9:19]-np.clip(pos,-cfg['clip_obs'],cfg['clip_obs']))
            error(v,'joint_vel_max_error',frame[19:29]-dq*np.float32(cfg['dof_vel_scale']))
            error(v,'ang_vel_max_error',frame[:3]-np.array(o['angular_velocity_body'],np.float32)*np.float32(cfg['ang_vel_scale']))
            error(v,'gravity_quaternion_max_error',frame[3:6]-Rof(o['quaternion_wxyz_body_to_world']).T@np.array([0,0,-1.]))
            phase=np.float32(phase+np.float32(np.float32(cfg['policy_period_float32'])*np.float32(cfg['gait_frequency'])))
            while phase>=1:phase=np.float32(phase-1)
            moving=np.linalg.norm(np.array(row['actual_applied_command'],np.float32))>np.float32(.1)
            angle=np.float32(np.float32(2)*np.float32(math.pi)*phase)
            recurrence=np.array([np.sin(angle),np.cos(angle)],np.float32) if moving else np.zeros(2,np.float32)
            expected=np.array([math.sin(2*math.pi*((row['episode_time']*1.25)%1)),math.cos(2*math.pi*((row['episode_time']*1.25)%1))]) if moving else np.zeros(2)
            error(v,'phase_runtime_recurrence_max_error',frame[39:41]-recurrence);error(v,'phase_episode_contract_max_error',frame[39:41]-expected)
            v['phase_mask_correct'] &= bool(np.linalg.norm(frame[39:41])>.99) if moving else bool(np.all(frame[39:41]==0))
            v['no_added_noise'] &= not row['observation_noise_applied'];v['raw_action_clip_steps']+=int(np.any(raw!=clipped))
            qt=default+clipped*np.array(cfg['action_scale'],np.float32);qt[cfg['wheel_indices']]=default[cfg['wheel_indices']]
            vt=np.zeros(10,np.float32);vt[cfg['wheel_indices']]=(clipped*np.array(cfg['action_scale'],np.float32))[cfg['wheel_indices']]
            error(v,'q_target_max_error',qt-row['new_policy_targets']['q']);error(v,'dq_target_max_error',vt-row['new_policy_targets']['dq'])
            for u in row['low_level_updates']:
                candidate=np.array(u['feedforward_tau'],np.float32)+np.array(u['kp'],np.float32)*(np.array(u['q_target'],np.float32)-np.array(u['sensor_q'],np.float32))+np.array(u['kd'],np.float32)*(np.array(u['dq_target'],np.float32)-np.array(u['sensor_dq'],np.float32))
                error(v,'pd_candidate_max_error',candidate-u['torque_candidates_policy_order']);error(v,'pd_bounded_max_error',np.clip(candidate,-limits,limits)-u['bounded_torque_policy_order'])
                low_times.append(u['actual_sim_time']);v['pd_updates']+=1
            last_before=(row['pre']['qpos_mujoco'],row['pre']['qvel_mujoco'])
            for sub in row['physics_substeps']:
                v['physics_substeps']+=1;v['chronology'] &= sub['physics_step']==last_physics+1;last_physics=sub['physics_step']
                force=np.array(sub['actuator_force'])[actmap];velocity=np.array(sub['post_qvel'])[vi];contact={s:False for s in foot};abnormal=False
                for c in sub['contacts']:
                    if c['efc_address']<0:continue
                    g1,g2=c['geom1'],c['geom2']
                    for side,gid in foot.items():contact[side]|=gid in [g1,g2] and (geom[g1]['body_id']==0 or geom[g2]['body_id']==0)
                    if geom[g1]['body_id']==0 or geom[g2]['body_id']==0:
                        robot=g2 if geom[g1]['body_id']==0 else g1
                        abnormal |= robot not in foot.values()
                for k,x in [('t',sub['from_sim_time']),('forces',force),('dq',velocity),('sat',np.abs(force)>=limits*(1-1e-7)),('overspeed',np.abs(velocity)>vmax),('abnormal_contact',abnormal),('left_contact',contact['left']),('right_contact',contact['right'])]:physics[k].append(x)
                last_pre_physics=last_before;last_before=(sub['post_qpos'],sub['post_qvel'])
            R=Rof(s['base_quaternion_wxyz_body_to_world']);roll=math.atan2(R[2,1],R[2,2]);pitch=math.asin(np.clip(-R[2,0],-1,1));yaw=math.atan2(R[1,0],R[0,0]);omega=np.array(s['base_angular_velocity_body']);pg=R.T@np.array(cfg['gravity'])/np.linalg.norm(cfg['gravity'])
            lean=math.degrees(math.asin(np.clip(-pg[0],-1,1)));error(v,'backward_formula_consistency_deg',lean-s['backward_lean_deg'])
            error(v,'gravity_pre_state_max_error',pg-s['projected_gravity_from_raw_pose'])
            ct={side:any(c['efc_address']>=0 and gid in [c['geom1'],c['geom2']] and (geom[c['geom1']]['body_id']==0 or geom[c['geom2']]['body_id']==0) for c in s['contacts']) for side,gid in foot.items()}
            for k,x in [('t',t),('pos',s['base_position_world']),('vx',s['base_origin_linear_velocity_body'][0]),('vy',s['base_origin_linear_velocity_body'][1]),('vx_world',s['base_origin_linear_velocity_world'][0]),('vy_world',s['base_origin_linear_velocity_world'][1]),('planar_speed',np.linalg.norm(s['base_origin_linear_velocity_world'][:2])),('wz_body',omega[2]),('wz_world',s['base_angular_velocity_world'][2]),('yaw',yaw),('heading_rate',(math.sin(roll)*omega[1]+math.cos(roll)*omega[2])/math.cos(pitch)),('lean',lean),('roll',math.degrees(roll)),('cmd',row['command_in_observation']),('q',q),('dq',dq),('raw_action',raw),('clip',np.any(raw!=clipped)),('torque_fraction',np.max(np.abs(np.array([s['actuator_force'] for s in row['physics_substeps']])[:,actmap])/limits)),('left_contact',ct['left']),('right_contact',ct['right']),('left_height',s['feet']['left_foot_site']['site_height_above_plane_m']),('right_height',s['feet']['right_foot_site']['site_height_above_plane_m']),('phase_error',np.max(np.abs(frame[39:41]-expected)))]:data[k].append(x)
            final_state=row['post']
    data={k:np.asarray(x) for k,x in data.items()};physics={k:np.asarray(x) for k,x in physics.items()};data['yaw']=np.degrees(np.unwrap(data['yaw']))
    final_R=Rof(final_state['base_quaternion_wxyz_body_to_world']);final_yaw=math.degrees(math.atan2(final_R[1,0],final_R[0,0]));final_yaw+=360*round((data['yaw'][-1]-final_yaw)/360)
    positions={round(t,8):pos for t,pos in zip(data['t'],data['pos'])};positions[round(final_state['sim_time'],8)]=np.array(final_state['base_position_world'])
    yaws={round(t,8):y for t,y in zip(data['t'],data['yaw'])};yaws[round(final_state['sim_time'],8)]=final_yaw
    result_stats={'result':result,'windows':[],'joint_names':names,'joint_velocity_reference_limits_rad_s':vmax.tolist(),'velocity_limit_source':'snapshot LW.urdf; reporting reference only, not enforced by MJCF','physics_samples':len(physics['t']),'reset_check':'not exercised mid-run'}
    for lo,hi in windows:
        mask=(data['t']>=lo)&(data['t']<hi);pm=(physics['t']>=lo-1e-9)&(physics['t']<hi-1e-9);n=int(mask.sum());w={'window_s':[lo,hi],'samples':n,'expected_samples':round((hi-lo)/.02),'physics_samples':int(pm.sum()),'coverage':'complete' if n==round((hi-lo)/.02) else 'partial','reset_steps':[x for x in v['reset_steps'] if lo<=x*.02<hi]}
        for k in ['vx','vy','vx_world','vy_world','planar_speed','wz_body','wz_world','heading_rate','lean','roll','left_height','right_height']:w[k]=stats(data[k][mask])
        for k,ci in [('vx',0),('vy',1),('wz_body',2),('wz_world',2),('heading_rate',2)]:w[k+'_tracking_error']=stats(data[k][mask]-data['cmd'][mask,ci])
        if n:
            w['yaw_delta_deg']=float(yaws[hi]-yaws[lo]) if hi in yaws and lo in yaws else None
            delta=positions[hi]-positions[lo] if hi in positions and lo in positions else None
            w['net_displacement_world_m']=delta.tolist() if delta is not None else None;w['net_planar_displacement_m']=float(np.linalg.norm(delta[:2])) if delta is not None else None
            pts=np.stack([positions[x] for x in sorted(positions) if lo<=x<=hi]);w['planar_path_length_m']=float(np.linalg.norm(np.diff(pts[:,:2],axis=0),axis=1).sum())
            w['action_clip_step_fraction']=float(data['clip'][mask].mean());w['action_std_by_joint']=np.std(data['raw_action'][mask],axis=0).tolist();w['action_step_difference_rms']=float(np.sqrt(np.mean(np.diff(data['raw_action'][mask],axis=0)**2)))
            w['joint_q_std_rad']=np.std(data['q'][mask],axis=0).tolist();w['joint_dq_rms_rad_s']=np.sqrt(np.mean(data['dq'][mask]**2,axis=0)).tolist()
            w['joint_abs_dq_max_physics_rad_s']=np.max(abs(physics['dq'][pm]),axis=0).tolist();w['joint_overspeed_physics_fraction']=physics['overspeed'][pm].mean(axis=0).tolist()
            w['any_joint_overspeed_physics_fraction']=float(physics['overspeed'][pm].any(axis=1).mean());w['actual_torque_abs_max']=np.max(abs(physics['forces'][pm]),axis=0).tolist()
            w['torque_saturation_physics_fraction_by_joint']=physics['sat'][pm].mean(axis=0).tolist();w['any_torque_saturation_physics_fraction']=float(physics['sat'][pm].any(axis=1).mean())
            w['abnormal_ground_contact_physics_fraction']=float(physics['abnormal_contact'][pm].mean())
            for side in ['left','right']:w[side+'_contact_physics_fraction']=float(physics[side+'_contact'][pm].mean())
        result_stats['windows'].append(w)
    result_stats['whole_run']={'yaw_delta_deg':final_yaw-data['yaw'][0],'final_position_world_m':final_state['base_position_world'],'any_torque_saturation_physics_count':int(physics['sat'].any(axis=1).sum()),'any_joint_overspeed_physics_count':int(physics['overspeed'].any(axis=1).sum()),'abnormal_ground_contact_physics_count':int(physics['abnormal_contact'].sum()),'max_abs_dq_rad_s_by_joint':np.max(abs(physics['dq']),axis=0).tolist()}
    v['actual_pd_intervals_s']=sorted(set(np.round(np.diff(low_times),9)));v['initial_history_checked']=True;v['history_frames_checked']=v['rows']*10
    v['reset_boundary_validation']='initial process only; no mid-run reset performed'
    v['phase_episode_contract_matches']=v['phase_episode_contract_max_error']<1e-5
    validations[cid]=v;statistics[cid]=result_stats;allseries[cid]=data
    print(cid,'5-15',json.dumps(result_stats['windows'][2]['lean']), 'vx',result_stats['windows'][2]['vx']['mean'],'yaw',result_stats['windows'][2]['yaw_delta_deg'],flush=True)
write(out/'statistics.json',statistics);write(out/'input_validation.json',validations)
write(out/'runtime_config.json',{'configs':configs,'signal_sampling':{'pre':'integrated qpos/qvel at policy boundary; geometric/COM quantities re-evaluated on a private mjData copy','policy_state':'unmodified adapter sensor float32 values, copied before actual model call; typically evaluated at preceding physical step start','model_input':'copied directly from TensorView before ONNX inference','physics':'all physical steps; forces/contacts/foot kinematics at from_sim_time, integrated qpos/qvel at to_sim_time','previous_action':'previous model output after existing per-action clipping, before scale/default offset','video':'offline pre-state replay; no closed-loop rendering'},'coordinate_conventions':{'quaternion':'wxyz base_link body-to-world','axes':'body +X forward, +Z up; world gravity negative Z','linear_velocity':'freejoint/base origin qvel[0:3] world, body via R.T; COM velocity stored separately','lean':'degrees(asin(clamp(-unit_gravity_body.x,-1,1)))','roll':'ZYX atan2(R[2,1],R[2,2])','yaw':'ZYX atan2(R[1,0],R[0,0]), unwrapped over each reset-free run'},'reset':plan['reset_check']})
fig,axes=plt.subplots(5,6,figsize=(24,14),sharex=True)
for col,(cid,s) in enumerate(allseries.items()):
    axes[0,col].set_title(cid)
    for rr,(metric,label) in enumerate([('lean','Backward lean (deg)'),('vx','Body-origin vx (m/s)'),('wz_body','Angular velocity (rad/s)'),('yaw','World heading (deg)'),('torque_fraction','Max torque / limit')]):
        axes[rr,col].plot(s['t'],s[metric],lw=.7,label=metric)
        if rr==1:axes[rr,col].plot(s['t'],s['cmd'][:,0],'k--',lw=.7)
        if rr==2:axes[rr,col].plot(s['t'],s['wz_world'],lw=.7,label='world vertical');axes[rr,col].plot(s['t'],s['cmd'][:,2],'k--',lw=.7)
        for t in [2,15]:axes[rr,col].axvline(t,color='gray',lw=.5)
        axes[rr,col].grid(alpha=.2)
        if col==0:axes[rr,col].set_ylabel(label)
    axes[-1,col].set_xlabel('simulation time (s)')
axes[2,0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'overview.png',dpi=140);plt.close(fig)
