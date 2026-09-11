from pathlib import Path
import shlex, subprocess, json
root=Path('/home/lfr/rl_sar')
out=Path(__file__).resolve().parent.parent
flags=(root/'build/rl_sar/CMakeFiles/rl_sim_LW.dir/flags.make').read_text().splitlines()
inc=shlex.split(next(x.split(' = ',1)[1] for x in flags if x.startswith('CXX_INCLUDES =')))
sources=['rl_sdk/rl_sdk.cpp','rl_sdk/lw_configuration_validation.cpp',
         'observation_buffer/observation_buffer.cpp','motion_loader/motion_loader_lw.cpp',
         'inference_runtime/inference_runtime.cpp','simulation/lw_mujoco_control_adapter.cpp',
         'simulation/lw_sim_torque_validation.cpp']
cmd=['/usr/bin/c++','-std=c++17','-O2','-DUSE_ONNX', '-DPOLICY_DIR="/home/lfr/rl_sar/policy"',
     *inc,str(out/'tools/collect.cpp'),*[str(out/'experimental/rl_sdk.cpp') if s=='rl_sdk/rl_sdk.cpp' else str(root/'src/rl_sar/library/core'/s) for s in sources],
     '-L'+str(root/'library/mujoco/lib'),'-L'+str(root/'library/inference_runtime/onnxruntime/lib'),
     '-Wl,-rpath,'+str(root/'library/mujoco/lib')+':'+str(root/'library/inference_runtime/onnxruntime/lib'),
     '-lmujoco','-lonnxruntime','-lyaml-cpp','-ltbb','-lz','-lpthread','-o',str(out/'tools/collect')]
(out/'tools/build_command.json').write_text(json.dumps({'argv':cmd,'shell_command':shlex.join(cmd)},indent=2)+'\n')
with (out/'tools/build.log').open('w') as log:
    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
print('build exit',result.returncode)
if result.returncode: print((out/'tools/build.log').read_text()[-7000:])
raise SystemExit(result.returncode)
