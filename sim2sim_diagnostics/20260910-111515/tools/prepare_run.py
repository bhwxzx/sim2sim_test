from pathlib import Path
import hashlib,json,subprocess,shutil,datetime,shlex,xml.etree.ElementTree as ET
root=Path('/home/lfr/rl_sar'); out=Path(__file__).resolve().parent.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(*args): return subprocess.check_output(args,cwd=root,text=True)
paths=[root/'policy/LW/base.yaml',root/'policy/LW/robot_lab/leg_loco/config.yaml',root/'AGENTS.md',
       root/'docs/LW_QUICK_START_CN.md',root/'docs/LW_BUILD_DEPLOYMENT_CN.md',
       root/'src/rl_sar/src/rl_sim_LW.cpp',root/'src/rl_sar/include/rl_sim_LW.hpp',
       root/'src/rl_sar/fsm_robot/fsm_LW.hpp',
       root/'src/rl_sar/library/thirdparty/mujoco_simulate/mujoco_utils.hpp',
       root/'src/rl_sar/library/thirdparty/mujoco_simulate/simulate.h',
       root/'src/rl_sar/library/thirdparty/mujoco_simulate/simulate.cc']
for directory in ['rl_sdk','observation_buffer','inference_runtime','simulation','safety','vector_math']:
    paths+=list((root/'src/rl_sar/library/core'/directory).glob('*.hpp'))
    paths+=list((root/'src/rl_sar/library/core'/directory).glob('*.cpp'))
xml=root/'src/rl_sar_zoo/LW_description/mjcf/scene.xml'
paths+=[xml,xml.with_name('LW.xml')]
tree=ET.parse(xml.with_name('LW.xml'))
paths += [xml.parent/'assets'/x.attrib['file'] for x in tree.findall('.//asset/mesh')]
files=[]
for p in sorted(set(paths)):
    dest=out/'config_snapshot'/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,dest); files.append({'path':str(p),'snapshot':str(dest.relative_to(out)),'sha256':sha(p),'size_bytes':p.stat().st_size})
archive=Path('/home/lfr/policy_storage/LW/leg_loco/2026-09-04-11-16-35')
shutil.copy2(archive/'archive_manifest.json',out/'config_snapshot/archive_manifest.json')
models=[]
for p in [root/'policy/LW/robot_lab/leg_loco/policy.onnx',archive/'policy.onnx',archive/'policy.pt']:
    models.append({'path':str(p.resolve()),'sha256':sha(p),'size_bytes':p.stat().st_size,'loaded_for_control':p==root/'policy/LW/robot_lab/leg_loco/policy.onnx'})
checkpoint='/home/young/liufengrong/robot_lab/logs/rsl_rl/LW_leg_flat_amp_roa/2026-09-04_11-16-35/model_50000.pt'
baseline={'created_at':datetime.datetime.now().astimezone().isoformat(),'repository':str(root),'head':run('git','rev-parse','HEAD').strip(),
    'git_status_short':run('git','status','--short'),'git_diff_numstat':run('git','diff','--numstat'),
    'git_diff':run('git','diff','--no-ext-diff'),'files':files,'models':models,
    'checkpoint':{'archived_path':checkpoint,'present':Path(checkpoint).is_file(),'sha256_independently_verified':None,
        'archive_declared_sha256':'59ab5245acb17de8ea1b79e924c2f0189765be3c51c85f437b22b8e2423b0e19'},
    'original_executable_not_launched':{'path':str(root/'build/rl_sar/rl_sim_LW'),'sha256':sha(root/'build/rl_sar/rl_sim_LW')},
    'diagnostic_executable':{'path':str(out/'tools/collect'),'sha256':sha(out/'tools/collect')},
    'ldd':run('ldd',str(out/'tools/collect')),
    'runtime_libraries':[]}
for p in [root/'library/mujoco/lib/libmujoco.so.3.2.7',root/'library/inference_runtime/onnxruntime/lib/libonnxruntime.so',Path('/usr/lib/x86_64-linux-gnu/libyaml-cpp.so.0.7.0')]:
    baseline['runtime_libraries'].append({'path':str(p.resolve()),'sha256':sha(p)})
(out/'baseline.json').write_text(json.dumps(baseline,indent=2,ensure_ascii=False)+'\n')
commands=[]
for cid in ['stand-a01','walk-stop-a01']:
    cmd=[str(out/'tools/collect'),str(xml),cid,str(out/'cases'/cid)]
    commands.append({'case_id':cid,'cwd':str(root),'argv':cmd,'shell_command':shlex.join(cmd)+' > '+shlex.quote(str(out/'cases'/cid/'console.log'))+' 2>&1'})
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
print('prepared',out)
