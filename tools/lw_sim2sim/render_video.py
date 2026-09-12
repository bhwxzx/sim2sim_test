import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,pathlib,json,gzip,subprocess,math
import numpy as np
import mujoco,cv2
p=argparse.ArgumentParser();p.add_argument('--run',type=pathlib.Path,required=True);p.add_argument('--ffmpeg',required=True);a=p.parse_args()
plan=json.loads((a.run/'plan.json').read_text());m=mujoco.MjModel.from_xml_path(plan['scene']);d=mujoco.MjData(m)
m.vis.global_.offwidth=960;m.vis.global_.offheight=720
renderer=mujoco.Renderer(m,height=720,width=960);camera=mujoco.MjvCamera();camera.distance=2.6;camera.elevation=0
for case in plan['cases']:
    cid=case['case_id'];folder=a.run/'cases'/cid;config=json.loads((folder/'effective_config.json').read_text())
    video=folder/'side_view.mp4';assert not video.exists()
    fps=1/(float(config['base_dt_float32'])*config['decimation']);fps=round(fps,6)
    cmd=[a.ffmpeg,'-nostdin','-v','warning','-n','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r',str(fps),'-i','pipe:0','-an','-c:v','libopenh264','-b:v','3500k','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    count=0;checks=[]
    with (folder/'video_console.log').open('x') as log:
        encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            with gzip.open(folder/'telemetry.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line)
                    if 'pre' not in row:continue
                    state=row['pre'];d.qpos[:]=state['qpos_mujoco'];d.qvel[:]=state['qvel_mujoco'];d.ctrl[:]=state['ctrl'];d.time=state['sim_time']
                    mujoco.mj_forward(m,d)
                    w,x,y,z=d.qpos[3:7];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    camera.lookat[:]=[d.qpos[0],d.qpos[1],max(.43,d.qpos[2]-.27)];camera.azimuth=90+math.degrees(yaw)
                    renderer.update_scene(d,camera=camera);frame=renderer.render().copy()
                    if count%50==0:
                        renderer.enable_segmentation_rendering();seg=renderer.render();renderer.disable_segmentation_rendering()
                        ids=seg[:,:,0];types=seg[:,:,1];valid=(types==int(mujoco.mjtObj.mjOBJ_GEOM))&(ids>=0)&(ids<m.ngeom)
                        robot=np.zeros(valid.shape,bool);robot[valid]=m.geom_bodyid[ids[valid]]>0
                        yy,xx=np.nonzero(robot)
                        check={'step':row['control_step'],'robot_pixels':len(xx),'bbox_xyxy':None if not len(xx) else [int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())]}
                        check['robot_present_and_inside_frame']=bool(len(xx) and xx.min()>0 and xx.max()<959 and yy.min()>94 and yy.max()<690)
                        checks.append(check)
                    cv2.rectangle(frame,(0,0),(960,94),(12,18,24),-1)
                    vx,vy,wz=row['planned_command'];texts=[f'{cid} | sim t={d.time:.2f}s | step={row["control_step"]} | reset={row["reset_id"]}',f'cmd [{vx:.1f}, {vy:.1f}, {wz:.1f}] | backward lean={state["backward_lean_deg"]:.2f} deg','Side view, yaw-follow | body +X / FRONT --> | flat ground']
                    for i,text in enumerate(texts):cv2.putText(frame,text,(14,25+29*i),cv2.FONT_HERSHEY_SIMPLEX,.58,(245,245,245),1,cv2.LINE_AA)
                    cv2.putText(frame,f'Offline telemetry replay; physics {config["mujoco_version"]}; renderer {mujoco.__version__}',(14,700),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1,cv2.LINE_AA)
                    encoder.stdin.write(frame.tobytes());count+=1
                    if count in [1,501,1000]:cv2.imwrite(str(folder/f'video_check_{count:04d}.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    if count%500==0:print(cid,count,flush=True)
            encoder.stdin.close();code=encoder.wait();assert code==0
        except Exception:
            encoder.stdin.close();encoder.wait();raise
    reader=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,_=reader.read()
        if not ok:break
        decoded+=1
    measured_fps=reader.get(cv2.CAP_PROP_FPS);reader.release()
    meta={'argv':cmd,'mode':'offline telemetry replay; no mj_step or policy invocation','physics_version':config['mujoco_version'],'renderer_version':mujoco.__version__,'frames':count,'fps':fps,'duration_s':count/fps,'decoded_frames':decoded,'decoded_fps':measured_fps,'validation_pass':decoded==count,'robot_visibility_checks':checks,'robot_visibility_all_sampled_frames_pass':all(x['robot_present_and_inside_frame'] for x in checks),'sampling':'one video frame per actual policy pre-state; 1 Hz segmentation visibility checks','camera':{'elevation_deg':0,'azimuth':'90+base yaw','distance':2.6},'last_frame_episode_time_s':(count-1)/fps}
    (folder/'video_metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(cid,'done',meta['robot_visibility_all_sampled_frames_pass'],flush=True)
renderer.close()
