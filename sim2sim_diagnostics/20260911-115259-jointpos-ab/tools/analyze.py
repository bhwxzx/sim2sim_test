from pathlib import Path
import gzip,json,math,hashlib,copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path(__file__).resolve().parent.parent
def write(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def stats(a):
    a=np.asarray(a,float)
    if a.size==0:return None
    return {'n':len(a),'mean':float(a.mean()),'std':float(a.std()),'p5':float(np.percentile(a,5)),'p95':float(np.percentile(a,95)),
        'min':float(a.min()),'max':float(a.max()),'rms':float(np.sqrt(np.mean(a*a)))}
def rotation(q):
    w,x,y,z=np.asarray(q,float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
        [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
        [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def rms(a,axis=None):return np.sqrt(np.mean(np.square(a),axis=axis))
def detrended_std(t,x):
    if len(t)<2:return np.zeros(x.shape[1])
    X=np.column_stack((t-t[0],np.ones(len(t))));residual=x-X@np.linalg.lstsq(X,x,rcond=None)[0]
    return np.std(residual,axis=0)
def load_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(l) for l in f]
windows={'stand-a01':[(5,10),(10,30)],'walk-stop-a01':[(5,10),(15,20),(20,22),(22,30),(30,40)]}
results={};validations={};allseries={};configs={};statuses={}
for condition in ['A','B']:
    for cid in windows:
        key=condition+'/'+cid;directory=out/condition/cid
        cfg=json.loads((directory/'effective_config.json').read_text());configs[key]=cfg
        result=json.loads((directory/'result.json').read_text());statuses[key]=result
        raw=load_rows(directory/'telemetry.jsonl.gz')
        rows=[r for r in raw if isinstance(r.get('model_input'),dict) and isinstance(r.get('model_raw_action'),list)]
        v={'rows':len(rows),'history_vectors_checked':0,'latest_input_exact':True,'all_history_frames_exact':True,
            'same_state_pair_only_approved_columns':True,'same_state_pair_matches_independent_calculation':True,
            'previous_action_exact':True,'command_exact':True,'observation_noise_applied':False,'all_input_output_finite':True,
            'joint_names_and_addresses_consistent':True,'model_input_capture_before_forward':True,
            'sensor_time_state_max_error':0.,'pd_formula_max_error':0.,'targets_max_error':0.,'paired_change_counts':[0]*41,
            'initial_reset_rows':[],'max_abs_xfrc_applied':0.,'max_abs_qfrc_applied':0.}
        history=[];previous=np.zeros(10,np.float32)
        sensor_qpos={0.:np.asarray(rows[0]['pre']['qpos_mujoco'])};sensor_qvel={0.:np.asarray(rows[0]['pre']['qvel_mujoco'])}
        for row in rows:
            for p in row['physics_substeps']:
                sensor_qpos[round(p['to_sim_time'],9)]=np.asarray(p['post_qpos']);sensor_qvel[round(p['to_sim_time'],9)]=np.asarray(p['post_qvel'])
        mapping=cfg['ab_experiment']['joint_mapping'];names=[m['joint_name'] for m in mapping]
        qi=[m['mj_qpos_adr'] for m in mapping];vi=[m['mj_dof_adr'] for m in mapping]
        foot={side:next(g['id'] for g in cfg['geoms'] if g['name']==side+'_foot_collision') for side in ['left','right']}
        world={g['id'] for g in cfg['geoms'] if g['body_id']==0}
        data={k:[] for k in ['time','lean','roll','yaw','heading_rate','yaw_rate_body','yaw_rate_world','vx','vx_world','vy',
            'cmd','xy','z','q','dq','action','action_delta_rms','torque_fraction','saturation','left_contact','right_contact']}
        physics=[];lowtimes=[];derived=[];initial_yaw=None
        for index,row in enumerate(rows):
            obs=row['observation_frame'];inp=np.asarray(row['model_input']['values'],np.float32).reshape(10,41)
            q=np.asarray(obs['raw_q_policy_float32'],np.float32);default=np.asarray(obs['default_q_policy_float32'],np.float32)
            dq=np.asarray(obs['dq_policy_float32'],np.float32);scale=np.float32(cfg['dof_pos_scale']);limit=np.float32(cfg['clip_obs'])
            pos=(q-default)*scale;expected_a=pos.copy();expected_a[[8,9]]=0;expected_b=pos.copy();expected_b[[7,9]]=0
            expected_a=np.clip(expected_a,-limit,limit);expected_b=np.clip(expected_b,-limit,limit)
            selected=expected_a if condition=='A' else expected_b
            v['latest_input_exact'] &= bool(np.array_equal(inp[-1,9:19],selected))
            v['observation_noise_applied'] |= obs['noise_applied']
            pa=np.asarray(row['same_state_pair']['A_frame'],np.float32);pb=np.asarray(row['same_state_pair']['B_frame'],np.float32)
            changed=np.flatnonzero(pa!=pb).tolist()
            for c in changed:v['paired_change_counts'][c]+=1
            v['same_state_pair_only_approved_columns'] &= set(changed)<=set([16,17])
            v['same_state_pair_matches_independent_calculation'] &= bool(np.array_equal(pa[9:19],expected_a) and np.array_equal(pb[9:19],expected_b))
            frame=pa if condition=='A' else pb
            if row['reset']:history=[frame.copy() for _ in range(10)];v['initial_reset_rows'].append(row['control_step'])
            else:history=history[1:]+[frame.copy()]
            v['all_history_frames_exact'] &= bool(np.array_equal(inp,np.stack(history)));v['history_vectors_checked']+=10
            v['previous_action_exact'] &= bool(np.array_equal(inp[-1,29:39],previous))
            v['command_exact'] &= bool(np.array_equal(inp[-1,6:9],np.asarray(row['planned_command'],np.float32)))
            rawaction=np.asarray(row['model_raw_action'],np.float32);action=np.clip(rawaction,np.array(cfg['clip_actions_lower'],np.float32),np.array(cfg['clip_actions_upper'],np.float32))
            v['all_input_output_finite'] &= bool(np.isfinite(inp).all() and np.isfinite(rawaction).all())
            v['model_input_capture_before_forward'] &= row['model_input_capture_phase'].startswith('synchronous copy immediately before')
            v['joint_names_and_addresses_consistent'] &= obs['joint_mapping']==mapping
            t_sensor=max(0.,round(row['simulation_time']-.002,9))
            v['sensor_time_state_max_error']=max(v['sensor_time_state_max_error'],float(abs(sensor_qpos[t_sensor][qi].astype(np.float32)-q).max()),float(abs(sensor_qvel[t_sensor][vi].astype(np.float32)-dq).max()))
            qt=default+action*np.array(cfg['action_scale'],np.float32);qt[[8,9]]=default[[8,9]]
            vt=np.zeros(10,np.float32);vt[[8,9]]=(action*np.array(cfg['action_scale'],np.float32))[[8,9]]
            if row.get('new_policy_targets'):
                v['targets_max_error']=max(v['targets_max_error'],float(abs(qt-row['new_policy_targets']['q']).max()),float(abs(vt-row['new_policy_targets']['dq']).max()))
            for update in row['low_level_updates']:
                t=max(0.,round(update['actual_sim_time']-.002,9))
                ps=sensor_qpos[t][qi].astype(np.float32);vs=sensor_qvel[t][vi].astype(np.float32)
                candidate=np.asarray(update['feedforward_tau'],np.float32)+np.asarray(update['kp'],np.float32)*(np.asarray(update['q_target'],np.float32)-ps)+np.asarray(update['kd'],np.float32)*(np.asarray(update['dq_target'],np.float32)-vs)
                v['pd_formula_max_error']=max(v['pd_formula_max_error'],float(abs(candidate-update['torque_candidates_policy_order']).max()))
                lowtimes.append(update['actual_sim_time'])
            s=row['pre'];qw,qx,qy,qz=s['base_quaternion_wxyz_body_to_world'];R=rotation([qw,qx,qy,qz])
            lean=math.degrees(math.asin(np.clip(2*(qx*qz-qw*qy),-1,1)))
            roll=math.atan2(R[2,1],R[2,2]);pitch=-math.radians(lean);yaw=math.atan2(R[1,0],R[0,0])
            omega=np.asarray(s['base_angular_velocity_body']);heading_rate=(math.sin(roll)*omega[1]+math.cos(roll)*omega[2])/math.cos(pitch)
            if initial_yaw is None:initial_yaw=yaw
            contact={}
            for side,gid in foot.items():
                active=[ct for ct in s['contacts'] if gid in [ct['geom1'],ct['geom2']] and {ct['geom1'],ct['geom2']}&world and ct['efc_address']>=0]
                contact[side]={'active':bool(active),'normal_force_sum':sum(ct['force_contact_frame'][0] for ct in active)}
            for moment in ['pre','post']:
                if row.get(moment):
                    v['max_abs_xfrc_applied']=max(v['max_abs_xfrc_applied'],float(np.max(np.abs(row[moment]['xfrc_applied_world']))))
                    v['max_abs_qfrc_applied']=max(v['max_abs_qfrc_applied'],float(np.max(np.abs(row[moment]['qfrc_applied']))))
            subforces=np.asarray([st['actuator_force'] for st in row['physics_substeps']],float)
            lim=np.asarray(cfg['torque_limits']);saturation=np.abs(subforces)>=lim-1e-6
            max_fraction=float(np.max(abs(subforces)/lim)) if subforces.size else None
            for st,fsat in zip(row['physics_substeps'],saturation):physics.append({'t':st['from_sim_time'],'force':st['actuator_force'],'saturation':fsat})
            action_delta=float(rms(action-previous));previous=action
            for k,val in [('time',row['nominal_policy_time']),('lean',lean),('roll',math.degrees(roll)),('yaw',yaw),('heading_rate',heading_rate),
                ('yaw_rate_body',omega[2]),('yaw_rate_world',s['base_angular_velocity_world'][2]),('vx',s['base_origin_linear_velocity_body'][0]),
                ('vx_world',s['base_origin_linear_velocity_world'][0]),('vy',s['base_origin_linear_velocity_body'][1]),('cmd',row['planned_command'][0]),
                ('xy',s['base_position_world'][:2]),('z',s['base_position_world'][2]),('q',q),('dq',dq),('action',action),
                ('action_delta_rms',action_delta),('torque_fraction',max_fraction),('saturation',bool(np.any(saturation))),
                ('left_contact',contact['left']['active']),('right_contact',contact['right']['active'])]:data[k].append(val)
            derived.append({'condition':condition,'case_id':cid,'control_step':row['control_step'],'simulation_time':row['simulation_time'],
                'backward_lean_deg':lean,'base_yaw_rad':yaw,'heading_yaw_rate_rad_s':heading_rate,'base_omega_z_body_rad_s':float(omega[2]),
                'vx_body_error_m_s':float(s['base_origin_linear_velocity_body'][0]-row['planned_command'][0]),
                'contacts':contact,'max_actual_torque_limit_fraction':max_fraction,'any_substep_saturation':bool(np.any(saturation)),
                'gravity_recomputed_from_policy_quaternion':(rotation(obs['quaternion_wxyz_body_to_world']).T@np.array([0,0,-1.])).tolist()})
        with gzip.open(directory/'derived.jsonl.gz','xt',compresslevel=6) as f:
            for row in derived:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
        data={k:np.asarray(values) for k,values in data.items()};data['yaw']=np.degrees(np.unwrap(data['yaw']));data['error']=data['vx']-data['cmd']
        allseries[key]=data
        v['actual_pd_intervals_summary_s']=sorted(set(np.round(np.diff(lowtimes),9).tolist()))
        v['all_checks_pass']=all(v[k] for k in ['latest_input_exact','all_history_frames_exact','same_state_pair_only_approved_columns',
            'same_state_pair_matches_independent_calculation','previous_action_exact','command_exact','all_input_output_finite',
            'joint_names_and_addresses_consistent','model_input_capture_before_forward']) and not v['observation_noise_applied'] and v['sensor_time_state_max_error']==0 and v['pd_formula_max_error']==0 and v['targets_max_error']==0
        validations[key]=v
        t=data['time'];physical_t=np.array([st['t'] for st in physics]);forces=np.asarray([st['force'] for st in physics]);sat=np.asarray([st['saturation'] for st in physics])
        final_state=next((r['post'] for r in reversed(rows) if r.get('post')),rows[-1]['pre'])
        positions={round(r['nominal_policy_time'],8):np.asarray(r['pre']['base_position_world'][:2]) for r in rows};positions[round(final_state['sim_time'],8)]=np.asarray(final_state['base_position_world'][:2])
        endpoint_yaw=math.degrees(math.atan2(rotation(final_state['base_quaternion_wxyz_body_to_world'])[1,0],rotation(final_state['base_quaternion_wxyz_body_to_world'])[0,0]))
        endpoint_yaw+=360*round((data['yaw'][-1]-endpoint_yaw)/360)
        yaw_at={round(ti,8):yi for ti,yi in zip(t,data['yaw'])};yaw_at[round(final_state['sim_time'],8)]=endpoint_yaw
        def displacement(lo,hi):
            if lo not in positions or hi not in positions:return None
            indices=(t>=lo-1e-8)&(t<hi-1e-8)
            segment=np.vstack((data['xy'][indices],positions[hi]));delta=positions[hi]-positions[lo]
            heading=math.radians(yaw_at[lo]);fwd=np.array([math.cos(heading),math.sin(heading)])
            return {'window_s':[lo,hi],'net_world_xy_m':delta.tolist(),'net_xy_m':float(np.linalg.norm(delta)),
                'net_forward_at_window_start_heading_m':float(delta@fwd),'path_length_xy_m':float(np.linalg.norm(np.diff(segment,axis=0),axis=1).sum())}
        summary={'result':result,'windows':[],'joint_order':names,'physics_samples':len(physics),
            'actual_torque_abs_max_by_actuator':np.max(abs(forces),axis=0).tolist(),'saturation_fraction_by_actuator':np.mean(sat,axis=0).tolist(),
            'final_xy_m':positions[round(final_state['sim_time'],8)].tolist(),'yaw_change_entire_run_deg':endpoint_yaw-data['yaw'][0],
            'stopping_drift':{f'{lo}-{hi}':displacement(lo,hi) for lo,hi in [(20,22),(22,30),(30,40),(20,40),(22,40)]} if cid=='walk-stop-a01' else {}}
        for lo,hi in windows[cid]:
            mask=(t>=lo-1e-8)&(t<hi-1e-8);n=int(mask.sum());expected_n=round((hi-lo)/.02)
            w={'window_s':[lo,hi],'samples':n,'expected_samples':expected_n,'coverage':'complete' if n==expected_n else 'partial' if n else 'not_reached',
                'reset_samples':[r['control_step'] for r in rows if r['reset'] and lo<=r['nominal_policy_time']<hi]}
            for k in ['lean','roll','vx','vx_world','vy','error','yaw','yaw_rate_body','yaw_rate_world','heading_rate','z','action_delta_rms']:
                w[k]=stats(data[k][mask])
            if n:
                w['speed_tracking_mae']=float(abs(data['error'][mask]).mean());w['speed_tracking_rmse']=float(rms(data['error'][mask]))
                w['yaw_delta_deg']=float(yaw_at[hi]-yaw_at[lo]) if lo in yaw_at and hi in yaw_at else None
                w['left_contact_fraction']=float(data['left_contact'][mask].mean());w['right_contact_fraction']=float(data['right_contact'][mask].mean())
                w['no_foot_contact_fraction']=float((~data['left_contact'][mask]&~data['right_contact'][mask]).mean())
                q_std=detrended_std(t[mask],data['q'][mask]);action=data['action'][mask];actdiff=np.diff(action,axis=0)
                w['joint_position_std_rad']=np.std(data['q'][mask],axis=0).tolist();w['joint_detrended_std_rad']=q_std.tolist()
                w['leg_joint_detrended_pooled_rms_deg']=float(np.degrees(rms(q_std[:8])))
                w['joint_velocity_rms_rad_s']=rms(data['dq'][mask],axis=0).tolist();w['leg_velocity_pooled_rms_rad_s']=float(rms(data['dq'][mask,:8]))
                w['action_std_by_joint']=np.std(action,axis=0).tolist();w['action_delta_rms_by_joint']=rms(actdiff,axis=0).tolist() if n>1 else None
                w['action_delta_pooled_rms']=float(rms(actdiff)) if n>1 else None
                pmask=(physical_t>=lo-1e-8)&(physical_t<hi-1e-8)
                w['physics_samples']=int(pmask.sum());w['saturation_any_substep_fraction']=float(sat[pmask].any(axis=1).mean())
                w['actuator_saturation_fraction']=sat[pmask].mean(axis=0).tolist();w['actual_torque_abs_max_by_joint']=np.max(abs(forces[pmask]),axis=0).tolist()
                w['net_displacement']=displacement(lo,hi)
            summary['windows'].append(w)
        results[key]=summary

compare=[]
for cid in windows:
    a=results['A/'+cid];b=results['B/'+cid]
    for aw,bw in zip(a['windows'],b['windows']):
        row={'case_id':cid,'window_s':aw['window_s'],'A':aw,'B':bw,'B_minus_A':None}
        if aw['coverage']=='complete' and bw['coverage']=='complete':
            delta={}
            for signal in ['lean','vx','vx_world','yaw_rate_body','heading_rate','roll']:
                delta[signal]={k:bw[signal][k]-aw[signal][k] for k in ['mean','std','p5','p95','rms']}
            for metric in ['speed_tracking_mae','speed_tracking_rmse','yaw_delta_deg','leg_joint_detrended_pooled_rms_deg','leg_velocity_pooled_rms_rad_s','action_delta_pooled_rms','saturation_any_substep_fraction']:
                delta[metric]=bw[metric]-aw[metric]
            if aw['net_displacement'] and bw['net_displacement']:
                delta['net_xy_m']=bw['net_displacement']['net_xy_m']-aw['net_displacement']['net_xy_m']
            row['B_minus_A']=delta
        compare.append(row)
clean=[]
for k,cfg in configs.items():
    x=copy.deepcopy(cfg);x.pop('ab_experiment');clean.append(x)
global_checks={'all_four_actual_configs_equal_excluding_ab_label':all(x==clean[0] for x in clean),
    'four_runs_only':len(results)==4,'all_semantic_checks_pass':all(v['all_checks_pass'] for v in validations.values()),
    'all_cases_completed':all(r['result']['status']=='completed' for r in results.values()),
    'new_A_vs_historical_reference':{}}
for cid in windows:
    current=load_rows(out/'A'/cid/'telemetry.jsonl.gz');previous=load_rows(out.parent/'20260910-111515/cases'/cid/'telemetry.jsonl.gz')
    global_checks['new_A_vs_historical_reference'][cid]={'same_sample_count':len(current)==len(previous),
        'input_and_action_exact':len(current)==len(previous) and all(a['model_input']==b['model_input'] and a['model_raw_action']==b['model_raw_action'] for a,b in zip(current,previous)),
        'raw_qpos_qvel_exact':len(current)==len(previous) and all(a['pre']['qpos_mujoco']==b['pre']['qpos_mujoco'] and a['pre']['qvel_mujoco']==b['pre']['qvel_mujoco'] for a,b in zip(current,previous))}
write(out/'statistics.json',results);write(out/'comparison.json',compare);write(out/'validation.json',{'cases':validations,'global':global_checks})
source_runtime=json.loads((out/'source_reference/runtime_config.json').read_text())
write(out/'runtime_config.json',{'actual_cases':configs,'schedule_and_contract_reference':source_runtime,
    'experiment':json.loads((out/'experiment_plan.json').read_text()),
    'measurement_notes':{'linear_velocity':'base_origin_* at freejoint/body origin; base_com_* at inertial COM; explicit world/body frames',
        'pre':'current integration boundary; policy_state/observation_frame raw_q uses native named sensor evaluation at previous physical substep (except reset)',
        'snapshot':'input/frame/state synchronously copied before underlying model forward; completed row persisted after physics; pending row retained on exception',
        'backward_formula':'degrees(asin(clamp(2*(qx*qz-qw*qy),-1,1))) from raw freejoint wxyz body-to-world',
        'yaw_rate':'body omega_z and world omega_z recorded separately; heading derivative=(sin(roll)*omega_body_y+cos(roll)*omega_body_z)/cos(pitch)',
        'noise':'joint_pos observation noise off for both conditions; this isolates zero-mask semantics, not full noisy Isaac observations',
        'units':'q rad, dq/omega rad/s, velocity m/s, torque N*m; observation inputs are float32 and raw MuJoCo state is float64',
        'oscillation':'per-joint raw std plus residual std after removing a fitted linear trend; pooled RMS over 8 non-wheel joints; action increments over adjacent samples within window',
        'fall':'same diagnostic thresholds as original collector; no retry; time_limit is normal termination',
        'reset':'each process home_leg + original direct Leg FSM Enter; first history filled with own condition frame; no intermediate physical reset in four runs',
        'source_reference_caveat':'previous report order uncertainty is superseded by user verified actual policy-input/action order; source reference describes historical run only'}})
fig,axes=plt.subplots(6,2,figsize=(14,17),sharex='col')
for col,cid in enumerate(windows):
    for condition,color in [('A','#d16c29'),('B','#1679ae')]:
        d=allseries[condition+'/'+cid];t=d['time']
        for row,metric in enumerate(['lean','vx','yaw','yaw_rate_body','action_delta_rms','torque_fraction']):
            axes[row,col].plot(t,d[metric],lw=.7,label=condition,color=color)
    axes[0,col].set_title(cid)
    d=allseries['A/'+cid];axes[1,col].step(d['time'],d['cmd'],where='post',color='black',ls='--',lw=.7,label='command')
    for row,label in enumerate(['Backward lean (deg)','Base-origin vx, body (m/s)','Unwrapped yaw (deg)','Body omega_z (rad/s)','Action delta pooled RMS','Max |actual torque| / limit']):
        ax=axes[row,col];ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
        if cid=='walk-stop-a01':ax.axvspan(10,20,color='green',alpha=.06)
    axes[5,col].set_xlabel('Simulation time (s)')
fig.suptitle('LW Flat AMP-ROA closed-loop A/B | only joint_pos zero semantics differ\nA: wheel positions zero; B: left-foot and left-wheel positions zero | no observation noise',fontsize=12)
fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'overview.png',dpi=150);plt.close(fig)
print(json.dumps(global_checks,indent=2))
for row in compare:
    print(row['case_id'],row['window_s'],'lean A/B/delta',row['A']['lean']['mean'] if row['A']['lean'] else None,
        row['B']['lean']['mean'] if row['B']['lean'] else None,row['B_minus_A']['lean']['mean'] if row['B_minus_A'] else None)
