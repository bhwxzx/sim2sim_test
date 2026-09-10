import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import gzip,json,subprocess,math
import numpy as np
import mujoco,cv2
out=Path(__file__).resolve().parent.parent
xml=out/'config_snapshot/src/rl_sar_zoo/LW_description/mjcf/scene.xml'
ffmpeg='/home/lfr/miniconda3/envs/fluxweave/bin/ffmpeg'
m=mujoco.MjModel.from_xml_path(str(xml));d=mujoco.MjData(m)
# Visual-only offscreen dimensions; never used by the physics collector.
m.vis.global_.offwidth=960;m.vis.global_.offheight=720
r=mujoco.Renderer(m,height=720,width=960)
cam=mujoco.MjvCamera();cam.distance=2.35;cam.elevation=0
for cid in ['stand-a01','walk-stop-a01']:
    directory=out/'cases'/cid;video=directory/'side_view.mp4'
    if video.exists():raise RuntimeError('refuse video overwrite')
    cmd=[ffmpeg,'-nostdin','-v','warning','-n','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','50',
         '-i','pipe:0','-an','-c:v','libopenh264','-b:v','3500k','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    n=0
    with (directory/'video_console.log').open('x') as log:
        p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        try:
            with gzip.open(directory/'telemetry.jsonl.gz','rt') as f:
                for line in f:
                    row=json.loads(line)
                    if 'pre' not in row:continue
                    s=row['pre'];d.qpos[:]=s['qpos_mujoco'];d.qvel[:]=s['qvel_mujoco'];d.ctrl[:]=s['ctrl'];d.time=s['sim_time']
                    # Kinematics/render only. No mj_step and no policy calls.
                    mujoco.mj_forward(m,d)
                    w,x,y,z=d.qpos[3:7];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                    cam.lookat[:]=[d.qpos[0],d.qpos[1],.43];cam.azimuth=90+math.degrees(yaw)
                    r.update_scene(d,camera=cam); frame=r.render().copy()
                    cv2.rectangle(frame,(0,0),(960,94),(12,18,24),-1)
                    for i,text in enumerate([f'{cid} | sim t={d.time:.2f} s | policy step={row["control_step"]}',
                        f'cmd vx={row["planned_command"][0]:.1f} m/s, vy=0, wz=0 | backward lean={s["backward_lean_deg"]:.2f} deg',
                        'Exact side view (yaw-follow), elevation 0 | body +X / FRONT -->']):
                        cv2.putText(frame,text,(14,25+29*i),cv2.FONT_HERSHEY_SIMPLEX,.61,(245,245,245),1,cv2.LINE_AA)
                    cv2.putText(frame,'State replay only; physics: MuJoCo 3.2.7; renderer: 3.7.0',(14,698),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1,cv2.LINE_AA)
                    p.stdin.write(frame.tobytes());n+=1
                    if n==1:cv2.imwrite(str(directory/'side_view_first_frame.png'),cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
                    if n%500==0:print(cid,n,flush=True)
            p.stdin.close();code=p.wait()
            if code:raise RuntimeError(f'encoder exit {code}')
        except Exception:
            p.stdin.close();p.wait();raise
    (directory/'video_metadata.json').write_text(json.dumps({'argv':cmd,'frames':n,'fps':50,'duration_seconds':n/50,
        'renderer_version':mujoco.__version__,'physics_version':'3.2.7','mode':'recorded state replay; no dynamics integration',
        'camera':{'elevation_deg':0,'azimuth_deg':'90 + recorded base yaw','distance':2.35,'lookat':'base x,y and world z=0.43'},
        'frame_source':'telemetry pre state, one frame per policy step'},indent=2)+'\n')
    print(cid,'video done',n,flush=True)
r.close()
