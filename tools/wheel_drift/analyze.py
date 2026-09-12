"""Full-rate streaming validation and drift statistics for a fixed, authorized matrix."""
import argparse,pathlib,json,gzip,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();out=a.run
plan=json.loads((out/'plan.json').read_text())
def write(path,x):path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
def stat(x):
 x=np.asarray(x,float)
 if not x.size:return None
 return dict(n=len(x),mean=float(x.mean()),std=float(x.std()),rms=float(np.sqrt(np.mean(x*x))),p5=float(np.percentile(x,5)),p95=float(np.percentile(x,95)),min=float(x.min()),max=float(x.max()))
def finite(x):
 if isinstance(x,float):return math.isfinite(x)
 if isinstance(x,list):return all(finite(v) for v in x)
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 return True
def rot(q):
 w,x,y,z=q;return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def err(v,k,d):v[k]=max(v.get(k,0.),float(np.max(np.abs(d))))
statistics={};validation={};configs={};series={}
for spec in plan['cases']:
 cid=spec['case_id'];folder=out/'cases'/cid;result=json.loads((folder/'result.json').read_text())
 if not (folder/'effective_config.json').exists():statistics[cid]={'result':result,'windows':[]};validation[cid]={'unavailable':'startup failed'};continue
 cfg=json.loads((folder/'effective_config.json').read_text());configs[cid]=cfg
 mapping=cfg['joint_mapping_evidence'];qadr=[m['mj_qpos_adr'] for m in mapping];dadr=[m['mj_dof_adr'] for m in mapping];sadr=[m['position_sensor_adr'] for m in mapping]
 actmap=[next(a['id'] for a in cfg['actuators'] if a['trnid'][0]==m['mj_joint_id']) for m in mapping];limits=np.array(cfg['torque_limits'])
 v={'records':0,'all_finite':True,'chronology':True,'shape_dtype':True,'history_exact':True,'previous_action_exact':True,'commands_exact':True,'wheel_positions_zero':True,'noise_off_identity':True,'noise_equation_exact':True,'noise_bounds':True,'noise_mask':True,'sensor_mapping_exact':True,'no_external_force':True,'reset_steps':[],'terminated_steps':[],'physics_steps':0,'pd_updates':0}
 data={k:[] for k in ['t','pos','v_world','v_body','omega_world','omega_body','lean','roll','yaw','cmd','action','q','dq','clip','sat_fraction','torque_max','contacts','noise']}
 history=[];prev=np.zeros(10,np.float32);times=[];last_t=-1;last_phys=0;last_subtime=0;final=None;resets=[];safety={}
 with gzip.open(folder/'telemetry.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line);v['all_finite'] &= finite(row);i=row['control_step'];t=row['simulation_time'];v['records']+=1
   v['chronology'] &= t>last_t and i==v['records']-1 and abs(t-row['nominal_policy_time'])<1e-7;last_t=t
   if row['reset']:v['reset_steps'].append(i);resets.append(t);history=[];prev=np.zeros(10,np.float32)
   if row['terminated']:v['terminated_steps'].append(i)
   for e in row.get('safety_events',[]):safety[json.dumps(e,sort_keys=True)]=e
   if row.get('model_input') is None:continue
   mi=row['model_input'];v['shape_dtype'] &= mi['shape']==[1,195] and mi['dtype']=='float32' and len(mi['values'])==195
   inp=np.array(mi['values'],np.float32).reshape(5,39);frame=inp[-1];history=([frame.copy()]*5 if not history else history[1:]+[frame.copy()]);v['history_exact'] &= np.array_equal(inp,np.stack(history));v['wheel_positions_zero'] &= bool(np.all(inp[:,[17,18]]==0))
   raw=np.array(row['model_raw_action'],np.float32);clipped=np.clip(raw,np.array(cfg['clip_actions_lower'],np.float32),np.array(cfg['clip_actions_upper'],np.float32));v['previous_action_exact'] &= np.array_equal(frame[29:39],prev);prev=clipped
   v['commands_exact'] &= np.array_equal(frame[6:9],np.array(row['planned_command'],np.float32)) and row['planned_command']==row['actual_applied_command']
   n=row['noise_evidence'];clean=np.array(n['clean_deployment_frame'],np.float32);nr=np.array(n['raw_before_noise'],np.float32);noise=np.array(n['raw_additive_noise'],np.float32);scale=np.array(n['scale'],np.float32)
   expected=np.clip(nr+noise,-100,100)*scale
   v['noise_equation_exact'] &= np.array_equal(frame,expected) and np.array_equal(frame,np.array(n['processed_frame'],np.float32))
   bounds=np.zeros(39,np.float32);bounds[:3]=.2;bounds[3:6]=.05;bounds[9:17]=.01;bounds[19:29]=.5
   v['noise_bounds'] &= bool(np.all(abs(noise)<=bounds));v['noise_mask'] &= bool(np.all(noise[bounds==0]==0))
   if spec['condition']!='B':v['noise_off_identity'] &= np.array_equal(frame,clean) and bool(np.all(noise==0)) and not row['observation_noise_applied']
   else:v['noise_equation_exact'] &= row['observation_noise_applied']
   o=row['policy_state_float32'];q=np.array(o['q'],np.float32);dq=np.array(o['dq'],np.float32);default=np.array(row['default_q'],np.float32)
   v['sensor_mapping_exact'] &= row['joint_mapping']==mapping and np.array_equal(q,np.array(row['sensor_data_before_inference_float64'])[sadr].astype(np.float32))
   qp=q-default;qp[cfg['wheel_indices']]=0
   err(v,'clean_joint_position_error',clean[9:19]-np.clip(qp*np.float32(cfg['dof_pos_scale']),-100,100));err(v,'clean_joint_velocity_error',clean[19:29]-np.clip(dq*np.float32(cfg['dof_vel_scale']),-100,100))
   err(v,'clean_angular_velocity_error',clean[:3]-np.array(o['angular_velocity_body'],np.float32)*np.float32(cfg['ang_vel_scale']))
   err(v,'clean_gravity_quaternion_error',clean[3:6]-rot(o['quaternion_wxyz_body_to_world']).T@np.array([0,0,-1.]))
   qt=default+clipped*np.array(cfg['action_scale'],np.float32);qt[cfg['wheel_indices']]=default[cfg['wheel_indices']];vt=np.zeros(10,np.float32);vt[cfg['wheel_indices']]=(clipped*np.array(cfg['action_scale'],np.float32))[cfg['wheel_indices']]
   err(v,'position_target_error',qt-row['new_policy_targets']['q']);err(v,'velocity_target_error',vt-row['new_policy_targets']['dq'])
   for u in row['low_level_updates']:
    times.append(u['actual_sim_time']);v['pd_updates']+=1
    candidate=np.array(u['feedforward_tau'],np.float32)+np.array(u['kp'],np.float32)*(np.array(u['q_target'],np.float32)-np.array(u['sensor_q'],np.float32))+np.array(u['kd'],np.float32)*(np.array(u['dq_target'],np.float32)-np.array(u['sensor_dq'],np.float32))
    err(v,'pd_candidate_error',candidate-u['torque_candidates_policy_order']);err(v,'pd_limit_error',np.clip(candidate,-limits,limits)-u['bounded_torque_policy_order'])
   sat=[];torques=[];contacts=[]
   for sub in row['physics_substeps']:
    v['physics_steps']+=1;v['chronology'] &= sub['physics_step']==last_phys+1 and abs(sub['from_sim_time']-last_subtime)<1e-7;last_phys=sub['physics_step'];last_subtime=sub['to_sim_time']
    force=np.array(sub['actuator_force'])[actmap];torques.append(abs(force));sat.append(abs(force)>=limits*(1-1e-7));contacts.append(len([c for c in sub['contacts'] if c['efc_address']>=0]))
   s=row['pre'];v['no_external_force'] &= not np.any(s['xfrc_applied_world']) and not np.any(s['qfrc_applied'])
   w,x,y,z=s['base_quaternion_wxyz_body_to_world'];R=rot([w,x,y,z]);lean=math.degrees(math.asin(np.clip(2*(x*z-w*y),-1,1)));roll=math.degrees(math.atan2(R[2,1],R[2,2]));yaw=math.atan2(R[1,0],R[0,0]);err(v,'lean_formula_error',lean-s['backward_lean_deg'])
   values={'t':row['nominal_policy_time'],'pos':s['base_position_world'],'v_world':s['base_origin_linear_velocity_world'],'v_body':s['base_origin_linear_velocity_body'],'omega_world':s['base_angular_velocity_world'],'omega_body':s['base_angular_velocity_body'],'lean':lean,'roll':roll,'yaw':yaw,'cmd':frame[6:9],'action':raw,'q':q,'dq':dq,'clip':bool(np.any(raw!=clipped)),'sat_fraction':np.mean(sat,axis=0) if sat else np.zeros(10),'torque_max':np.max(torques,axis=0) if torques else np.zeros(10),'contacts':np.mean(np.array(contacts)>0) if contacts else None,'noise':noise}
   for k,val in values.items():data[k].append(val)
   final=row['post']
 data={k:np.asarray(x) for k,x in data.items()};v['actual_pd_intervals_s']=sorted(set(np.round(np.diff(times),8).tolist()));v['history_frames_checked']=len(data['t'])*5;v['reset_validation']='fresh process plus physical keyframe and policy activation only; no midrun reset tested'
 if not len(data['t']):validation[cid]=v;statistics[cid]={'result':result,'windows':[]};continue
 data['yaw']=np.degrees(np.unwrap(data['yaw']));w,x,y,z=final['base_quaternion_wxyz_body_to_world'];R=rot([w,x,y,z]);fy=math.degrees(math.atan2(R[1,0],R[0,0]));fy+=360*round((data['yaw'][-1]-fy)/360)
 tp=np.r_[data['t'],final['sim_time']];pp=np.vstack([data['pos'],final['base_position_world']]);yp=np.r_[data['yaw'],fy]
 def window(lo,hi):
  available_hi=min(hi,float(tp[-1]));mask=(data['t']>=lo-1e-8)&(data['t']<hi-1e-8);pm=(tp>=lo-1e-8)&(tp<=hi+1e-8)
  crossed=[r for r in resets if lo<r<hi];n=int(mask.sum());win={'window_s':[lo,hi],'samples':n,'expected_samples':round((hi-lo)/.02),'complete':n==round((hi-lo)/.02) and tp[-1]>=hi-1e-7,'reset_times_in_window':[r for r in resets if lo<=r<hi],'cross_reset':bool(crossed)}
  if not n or crossed:return win
  t=tp[pm];pos=pp[pm,:2];delta=pos[-1]-pos[0];slope=np.linalg.lstsq(np.c_[t-t[0],np.ones(len(t))],pos,rcond=None)[0][0]
  win.update({'actual_interval_s':[float(t[0]),float(t[-1])],'xy_net_vector_m':delta.tolist(),'xy_net_m':float(np.linalg.norm(delta)),'xy_fit_slope_m_s':slope.tolist(),'fit_drift_speed_m_s':float(np.linalg.norm(slope)),'max_distance_from_start_m':float(np.linalg.norm(pos-pos[0],axis=1).max()),'xy_range_m':np.ptp(pos,axis=0).tolist(),'xy_path_length_m_50hz':float(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum()),'yaw_change_deg':float(yp[pm][-1]-yp[pm][0])})
  for k,vals in [('speed_xy',np.linalg.norm(data['v_world'][mask,:2],axis=1)),('vx_world',data['v_world'][mask,0]),('vy_world',data['v_world'][mask,1]),('vx_body',data['v_body'][mask,0]),('vy_body',data['v_body'][mask,1]),('vx_tracking_error',data['v_body'][mask,0]-data['cmd'][mask,0]),('backward_lean_deg',data['lean'][mask]),('roll_deg',data['roll'][mask]),('yaw_rate_world_rad_s',data['omega_world'][mask,2]),('yaw_rate_body_rad_s',data['omega_body'][mask,2])]:win[k]=stat(vals)
  win['action_clip_steps']=int(data['clip'][mask].sum());win['torque_saturation_fraction_by_joint']=data['sat_fraction'][mask].mean(axis=0).tolist();win['absolute_torque_max_by_joint']=data['torque_max'][mask].max(axis=0).tolist();win['action_std_by_joint']=data['action'][mask].std(axis=0).tolist();win['joint_position_range_by_joint_rad']=np.ptp(data['q'][mask],axis=0).tolist();win['joint_velocity_rms_by_joint_rad_s']=np.sqrt(np.mean(data['dq'][mask]**2,axis=0)).tolist();win['any_contact_physics_fraction']=float(data['contacts'][mask].mean())
  return win
 windows=[(0,20),(20,40),(40,80),(80,120),(40,120)] if spec['condition']!='C' else [(0,20),(20,40),(40,60),(60,80),(80,100),(100,120)]
 # Additional reset-free 20s position trends aid interpretation without extra simulations.
 stats={'result':result,'condition':spec['condition'],'seed':spec['seed'],'joint_names':[m['joint_name'] for m in mapping],'windows':[window(*w) for w in windows],'twenty_second_trends':[window(k,k+20) for k in range(0,120,20)],'safety_events':list(safety.values()),'noise_raw_mean':data['noise'].mean(axis=0).tolist(),'noise_raw_std':data['noise'].std(axis=0).tolist(),'sampling_hz':50}
 v['counts_match_result']=v['records']==result['control_records'] and v['physics_steps']==result['physics_steps'] and v['pd_updates']==result['low_level_ticks']
 validation[cid]=v;statistics[cid]=stats;series[cid]=data
 print(cid,result['status'],[(w['window_s'],w.get('xy_net_m'),w.get('fit_drift_speed_m_s')) for w in stats['windows']],flush=True)
write(out/'statistics.json',statistics);write(out/'input_validation.json',validation)
write(out/'runtime_config.json',{'configs':configs,'sampling':{'pre':'policy boundary integrated qpos/qvel; geometry refreshed on private mjData copy','policy_state':'actual adapter float32 sensor values, normally preceding physical step start','model_input':'actual TensorView copied immediately before ONNX inference after noise/history','physics_substeps':'all substeps retained, from_sim_time forces/contacts and to_sim_time integrated state','previous_action':'prior raw action after original action clipping, before scale/default','velocity':'base freejoint origin; world qvel[:3], body R.T @ qvel[:3]; COM signals separate','quaternion':'base_link body-to-world wxyz','axes':'body +X forward +Z up, world gravity -Z','pitch':'ZYX pitch asin(2*(qw*qy-qx*qz)); backward lean = negative pitch','noise':'additive uniform to raw observations, term clip then scale, each frame before original history insertion; A/C exact original frame','reset':plan['reset']},'unavailable':['Exact Torch RNG sequence equivalence','Original effective training YAML and modified flat source bytes; flat no-override confirmed by user','Mid-run reset validation','Minimum geometric wheel/foot clearance; site heights are available','Video: not requested for this matrix']})
fig,axes=plt.subplots(8,3,figsize=(19,29));colors={42:'tab:blue',43:'tab:orange',44:'tab:green'}
for col,condition in enumerate('ABC'):
 axes[0,col].set_title(condition+(' training observation noise' if condition=='B' else ' no added noise'))
 for spec in plan['cases']:
  if spec['condition']!=condition or spec['case_id'] not in series:continue
  d=series[spec['case_id']];t=d['t'];c=colors[spec['seed']];label=f"seed {spec['seed']}"
  for row,y in [(0,d['pos'][:,0]),(1,d['pos'][:,1]),(3,d['v_body'][:,0]),(4,d['lean']),(5,d['roll']),(6,d['yaw']),(7,np.linalg.norm(d['v_world'][:,:2],axis=1))]:axes[row,col].plot(t,y,color=c,label=label,lw=.65)
  mask=t>=40;axes[2,col].plot(d['pos'][mask,0],d['pos'][mask,1],color=c,label=label,lw=.8)
  if np.any(mask):axes[2,col].scatter(d['pos'][mask,0][[0,-1]],d['pos'][mask,1][[0,-1]],c=[c,c],marker='x',s=25)
 for row in [0,1,3,4,5,6,7]:
  for boundary in [20,40,60,80,100]:axes[row,col].axvline(boundary,color='gray',alpha=.5,lw=.6)
  axes[row,col].set_xlim(0,120);axes[row,col].set_xlabel('simulation time (s)')
 axes[3,col].plot([0,40,40,60,60,120],[0,0,.5,.5,0,0] if condition=='C' else [0]*6,'k--',lw=.9,label='command vx')
 axes[2,col].set_xlabel('world X (m), t >= 40s');axes[2,col].set_ylabel('world Y (m)');axes[2,col].axis('equal')
 for row in range(8):axes[row,col].grid(alpha=.25);axes[row,col].legend(fontsize=8)
for row,label in [(0,'world X (m)'),(1,'world Y (m)'),(3,'body vx (m/s)'),(4,'backward lean (deg)'),(5,'roll (deg)'),(6,'world heading (deg)'),(7,'world XY speed (m/s)')]:axes[row,0].set_ylabel(label)
fig.tight_layout();fig.savefig(out/'overview.png',dpi=160);plt.close(fig)
