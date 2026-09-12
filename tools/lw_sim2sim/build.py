import argparse,json,pathlib,shlex,subprocess,hashlib
p=argparse.ArgumentParser();p.add_argument('--repo',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
source=pathlib.Path(__file__).with_name('collect.cpp')
database=json.loads((a.repo/'build/rl_sar/compile_commands.json').read_text())
entry=next(x for x in database if x['file'].endswith('/src/rl_sim_LW.cpp'))
original=entry.get('arguments') or shlex.split(entry['command']);includes=[];i=0
while i<len(original):
    token=original[i]
    if token in ['-I','-isystem']:includes.extend(original[i:i+2]);i+=2;continue
    if token.startswith('-I'):includes.append(token)
    i+=1
core=a.repo/'src/rl_sar/library/core';lib=a.repo/'library'
sources=[source,core/'rl_sdk/rl_sdk.cpp',core/'rl_sdk/lw_configuration_validation.cpp',core/'observation_buffer/observation_buffer.cpp',core/'motion_loader/motion_loader_lw.cpp',core/'inference_runtime/inference_runtime.cpp',core/'simulation/lw_mujoco_control_adapter.cpp',core/'simulation/lw_sim_torque_validation.cpp']
libdirs=[lib/'mujoco/lib',lib/'inference_runtime/onnxruntime/lib']
a.output.mkdir(parents=True,exist_ok=True)
cmd=['/usr/bin/c++','-std=c++17','-O2','-DUSE_ONNX','-DPOLICY_DIR=""',*includes,*map(str,sources),*[f'-L{x}' for x in libdirs],'-Wl,-rpath,'+':'.join(map(str,libdirs)),'-lmujoco','-lonnxruntime','-lyaml-cpp','-ltbb','-lz','-lpthread','-o',str(a.output/'collect')]
record={'argv':cmd,'shell_command':shlex.join(cmd),'cwd':str(a.repo),'source_sha256':{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
(a.output/'build_command.json').write_text(json.dumps(record,indent=2)+'\n')
with (a.output/'build.log').open('w') as log:subprocess.run(cmd,cwd=a.repo,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Build complete',a.output/'collect')
