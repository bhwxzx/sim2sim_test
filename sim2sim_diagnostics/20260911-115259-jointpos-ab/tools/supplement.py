from pathlib import Path
import json,gzip,math
import numpy as np
out=Path(__file__).resolve().parent.parent
def load(p):
    with gzip.open(p,'rt') as f:return [json.loads(l) for l in f]
def stats(x):
    x=np.asarray(x,float)
    return {'n':len(x),'mean':float(x.mean()),'std':float(x.std()),'p5':float(np.percentile(x,5)),'p95':float(np.percentile(x,95)),
        'min':float(x.min()),'max':float(x.max()),'rms':float(np.sqrt(np.mean(x*x)))}
def yaw(s):
    w,x,y,z=s['base_quaternion_wxyz_body_to_world'];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
result={'common_pre_obstacle_window_s':[10,15],'common_pre_obstacle':{},'contact_and_failure':{},'divergence':{}}
for condition in ['A','B']:
    cid='walk-stop-a01';p=out/condition/cid
    rows=load(p/'telemetry.jsonl.gz');cfg=json.loads((p/'effective_config.json').read_text());lo,hi=10,15
    selected=[r for r in rows if lo<=r['nominal_policy_time']<hi];end=next(r for r in rows if r['nominal_policy_time']==hi)
    values={'lean':[r['pre']['backward_lean_deg'] for r in selected],
        'vx':[r['pre']['base_origin_linear_velocity_body'][0] for r in selected],
        'roll':[r['pre']['roll_ZYX_deg'] for r in selected],
        'yaw_rate_body':[r['pre']['base_angular_velocity_body'][2] for r in selected]}
    values['error']=[x-r['planned_command'][0] for x,r in zip(values['vx'],selected)]
    angle_delta=np.unwrap([yaw(r['pre']) for r in selected]+[yaw(end['pre'])]);angle_delta=math.degrees(angle_delta[-1]-angle_delta[0])
    q=np.asarray([r['observation_frame']['raw_q_policy_float32'] for r in selected],float)
    t=np.asarray([r['nominal_policy_time'] for r in selected]);X=np.column_stack((t-t[0],np.ones(len(t))))
    residual=q-X@np.linalg.lstsq(X,q,rcond=None)[0];qstd=np.std(residual,axis=0)
    acts=np.asarray([r['model_raw_action'] for r in selected],float);diff=np.diff(acts,axis=0)
    forces=np.asarray([st['actuator_force'] for r in selected for st in r['physics_substeps']]);limits=np.asarray(cfg['torque_limits'])
    sat=np.abs(forces)>=limits-1e-6
    result['common_pre_obstacle'][condition]={k:stats(v) for k,v in values.items()}
    result['common_pre_obstacle'][condition].update({'yaw_change_deg':angle_delta,
        'tracking_mae':float(np.abs(values['error']).mean()),'tracking_rmse':float(np.sqrt(np.mean(np.square(values['error'])))),
        'leg_joint_detrended_pooled_rms_deg':float(np.degrees(np.sqrt(np.mean(np.square(qstd[:8]))))),
        'action_delta_pooled_rms':float(np.sqrt(np.mean(diff*diff))),
        'actuator_force_abs_max':np.max(abs(forces),axis=0).tolist(),'any_saturation_fraction':float(sat.any(axis=1).mean()),
        'net_xy_m':(np.asarray(end['pre']['base_position_world'][:2])-selected[0]['pre']['base_position_world'][:2]).tolist()})
    event=None;saturation_events=[];world_contacts={}
    for r in rows:
        for moment in ['pre','post']:
            s=r[moment]
            if not s:continue
            for ct in s['contacts']:
                if ct['efc_address']<0:continue
                ids=[ct['geom1'],ct['geom2']];world=[i for i in ids if cfg['geoms'][i]['body_id']==0]
                for gid in world:
                    world_contacts[str(gid)]=world_contacts.get(str(gid),0)+1
                    if gid!=0 and event is None:
                        event={'recorded_time':s['sim_time'],'control_step':r['control_step'],'moment':moment,'contact':ct,
                            'geom_descriptions':[cfg['geoms'][i] for i in ids],'base_position_world':s['base_position_world'],
                            'base_backward_lean_deg':s['backward_lean_deg'],'base_velocity_body':s['base_origin_linear_velocity_body'],
                            'note':'first active obstacle contact among recorded pre/post boundaries, not exact substep collision onset'}
        for st in r['physics_substeps']:
            force=np.asarray(st['actuator_force']);indices=np.flatnonzero(abs(force)>=limits-1e-6).tolist()
            if indices:saturation_events.append({'time':st['from_sim_time'],'actuator_indices':indices,'forces':[force[i] for i in indices]})
    terminal=rows[-1]['post'];quat=terminal['base_quaternion_wxyz_body_to_world'];w,x,y,z=quat
    result['contact_and_failure'][condition]={'first_active_obstacle_contact':event,'active_world_geom_occurrences':world_contacts,
        'first_actuator_saturation':saturation_events[0] if saturation_events else None,
        'last_actuator_saturation':saturation_events[-1] if saturation_events else None,'saturated_substeps':len(saturation_events),
        'terminal_time':terminal['sim_time'],'terminal_lean_deg':terminal['backward_lean_deg'],'terminal_roll_deg':terminal['roll_ZYX_deg'],
        'terminal_upright_tilt_deg':math.degrees(math.acos(np.clip(1-2*(x*x+y*y),-1,1))),
        'terminal_position_world':terminal['base_position_world'],'termination_reason':rows[-1]['termination_reason']}
a=result['common_pre_obstacle']['A'];b=result['common_pre_obstacle']['B']
delta={}
for key in ['lean','vx','error','roll','yaw_rate_body']:delta[key]={s:b[key][s]-a[key][s] for s in ['mean','std','p5','p95','rms']}
for key in ['yaw_change_deg','tracking_mae','tracking_rmse','leg_joint_detrended_pooled_rms_deg','action_delta_pooled_rms','any_saturation_fraction']:delta[key]=b[key]-a[key]
result['common_pre_obstacle']['B_minus_A']=delta
for cid in ['stand-a01','walk-stop-a01']:
    a=load(out/'A'/cid/'telemetry.jsonl.gz');b=load(out/'B'/cid/'telemetry.jsonl.gz')
    check={}
    for field in ['model_input','model_raw_action']:
        for x,y in zip(a,b):
            if x[field]!=y[field]:
                check['first_'+field+'_difference']={'step':x['control_step'],'time':x['simulation_time']}
                if field=='model_input':check['first_'+field+'_difference']['changed_410_indices']=np.flatnonzero(np.asarray(x[field]['values'])!=np.asarray(y[field]['values'])).tolist()
                break
    for x,y in zip(a,b):
        if x['pre']['qpos_mujoco']!=y['pre']['qpos_mujoco']:
            check['first_raw_qpos_difference']={'step':x['control_step'],'time':x['simulation_time']};break
    result['divergence'][cid]=check
(out/'supplemental_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
