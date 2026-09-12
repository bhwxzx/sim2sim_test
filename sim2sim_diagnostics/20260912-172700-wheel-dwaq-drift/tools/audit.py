import pathlib,json,gzip,hashlib,argparse
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();r=a.run;plan=json.loads((r/'plan.json').read_text());audit={}
for spec in plan['cases']:
 cid=spec['case_id'];cfg=json.loads((r/'cases'/cid/'effective_config.json').read_text());mapping=cfg['joint_mapping_evidence'];qa=[m['mj_qpos_adr'] for m in mapping];da=[m['mj_dof_adr'] for m in mapping]
 vel_sensors=[next(s['adr'] for s in cfg['sensors'] if s['type']==10 and s['objid']==m['mj_joint_id']) for m in mapping]
 rng=np.random.RandomState(spec['seed']);v={'noise_rng_exact':True,'dq_sensor_mapping_exact':True,'q_dq_physics_sensor_timing_exact':True,'ctrl_mapping_exact':True,'first_applied_delay_seconds':[],'series_sha256':{},'body_velocity_transform_max_error':0.0,'sensor_eval_age_range_s':[1e9,-1e9]};hashes={k:hashlib.sha256() for k in ['pose','input','action','first40_pose_input_action']};last_sensor_state=None;frames={};applied=set();initial=None;contact_counts={}
 with gzip.open(r/'cases'/cid/'telemetry.jsonl.gz','rt') as f:
  for line in f:
   row=json.loads(line);o=row['policy_state_float32'];dq=np.array(o['dq'],np.float32);q=np.array(o['q'],np.float32);sensor=np.array(row['sensor_data_before_inference_float64']);v['dq_sensor_mapping_exact'] &= np.array_equal(dq,sensor[vel_sensors].astype(np.float32))
   if last_sensor_state is not None:v['q_dq_physics_sensor_timing_exact'] &= np.array_equal(q,np.array(last_sensor_state[0])[qa].astype(np.float32)) and np.array_equal(dq,np.array(last_sensor_state[1])[da].astype(np.float32))
   age=row['simulation_time']-o['sensor_state_evaluation_time'];v['sensor_eval_age_range_s']=[min(v['sensor_eval_age_range_s'][0],age),max(v['sensor_eval_age_range_s'][1],age)]
   expected=np.zeros(39,np.float32)
   if spec['condition']=='B':
    u=(rng.randint(0,2**32,26,dtype=np.uint32)>>8).astype(np.float32)*np.float32(2**-24);u=np.float32(2)*u-np.float32(1)
    expected[:3]=u[:3]*np.float32(.2);expected[3:6]=u[3:6]*np.float32(.05);expected[9:17]=u[6:14]*np.float32(.01);expected[19:29]=u[16:26]*np.float32(.5)
   v['noise_rng_exact'] &= np.array_equal(expected,np.array(row['noise_evidence']['raw_additive_noise'],np.float32))
   w,x,y,z=row['pre']['base_quaternion_wxyz_body_to_world'];R=np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
   v['body_velocity_transform_max_error']=max(v['body_velocity_transform_max_error'],float(np.max(abs(R.T@np.array(row['pre']['base_origin_linear_velocity_world'])-row['pre']['base_origin_linear_velocity_body']))))
   frames[row['new_policy_targets']['frame']]=row['nominal_policy_time']
   for u in row['low_level_updates']:
    frame=u['policy_output_frame_applied']
    if frame in frames and frame not in applied:v['first_applied_delay_seconds'].append(u['actual_sim_time']-frames[frame]);applied.add(frame)
    for m in mapping:
     actuator=next(a['id'] for a in cfg['actuators'] if a['trnid'][0]==m['mj_joint_id']);v['ctrl_mapping_exact'] &= u['ctrl_mujoco_order'][actuator]==u['bounded_torque_policy_order'][m['policy_index']]
   before=(row['pre']['qpos_mujoco'],row['pre']['qvel_mujoco'])
   for sub in row['physics_substeps']:
    last_sensor_state=before;before=(sub['post_qpos'],sub['post_qvel'])
    for c in sub['contacts']:
     if c['efc_address']>=0:
      names=sorted([cfg['geoms'][c['geom1']]['name'],cfg['geoms'][c['geom2']]['name']]);key=' / '.join(names);contact_counts[key]=contact_counts.get(key,0)+1
   for k,vals in [('pose',row['pre']['qpos_mujoco']),('input',row['model_input']['values']),('action',row['model_raw_action'])]:
    b=np.array(vals,np.float64).tobytes();hashes[k].update(b)
    if row['nominal_policy_time']<40:hashes['first40_pose_input_action'].update(b)
   if initial is None:initial={'state':row['pre'],'q_default_policy_order':row['default_q'],'first_model_frame':row['current_frame']}
 v['first_applied_delay_seconds']=sorted(set(np.round(v['first_applied_delay_seconds'],8).tolist()));v['series_sha256']={k:h.hexdigest() for k,h in hashes.items()};v['contact_pair_physics_point_counts']=contact_counts;v['initial']=initial;audit[cid]=v;print(cid,{k:x for k,x in v.items() if k not in ['series_sha256','contact_pair_physics_point_counts','initial']},flush=True)
comparison={}
for condition in 'AC':comparison[condition+'_across_seeds_bitwise_identical']=all(audit[f'{condition}-seed{s}']['series_sha256']==audit[f'{condition}-seed42']['series_sha256'] for s in [43,44])
comparison['A_C_first40_exact']=all(audit[f'A-seed{s}']['series_sha256']['first40_pose_input_action']==audit[f'C-seed{s}']['series_sha256']['first40_pose_input_action'] for s in [42,43,44])
(r/'additional_audit.json').write_text(json.dumps({'cases':audit,'comparisons':comparison},indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')
