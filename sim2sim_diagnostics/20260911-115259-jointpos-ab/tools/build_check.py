from pathlib import Path
import json,subprocess,shlex
out=Path(__file__).resolve().parent.parent
cmd=json.loads((out/'tools/build_command.json').read_text())['argv']
cmd=[str(out/'tools/check_frames.cpp') if x==str(out/'tools/collect.cpp') else str(out/'tools/check_frames') if x==str(out/'tools/collect') else x for x in cmd]
(out/'tools/check_build_command.json').write_text(json.dumps({'argv':cmd,'shell_command':shlex.join(cmd)},indent=2)+'\n')
with (out/'tools/check_build.log').open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
if p.returncode:print((out/'tools/check_build.log').read_text())
raise SystemExit(p.returncode)
