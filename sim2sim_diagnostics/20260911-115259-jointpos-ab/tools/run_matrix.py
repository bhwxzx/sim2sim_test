from pathlib import Path
import json,subprocess,time,hashlib,datetime
out=Path(__file__).resolve().parent.parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((out/'frame_checks.json').read_text())['all_checks_pass']
baseline=json.loads((out/'baseline.json').read_text())
assert sha(baseline['model']['path'])==baseline['model']['sha256']
for f in baseline['protected_original_files']:assert sha(f['path'])==f['sha256'],f['path']
commands=json.loads((out/'commands.json').read_text())
for c in commands:
    path=out/c['condition']/c['case_id'];start=time.time()
    with (path/'console.log').open('x') as log:
        p=subprocess.run(c['argv'],cwd=c['cwd'],stdout=log,stderr=subprocess.STDOUT)
    (path/'process.json').write_text(json.dumps({'argv':c['argv'],'cwd':c['cwd'],'exit_code':p.returncode,
        'wall_seconds':time.time()-start,'started_at':datetime.datetime.fromtimestamp(start).astimezone().isoformat()},indent=2)+'\n')
    print(c['condition'],c['case_id'],'exit',p.returncode,flush=True)
    print((path/'result.json').read_text(),flush=True)
    # An individual failed rollout is retained; other prescribed independent
    # cases still run. No parameter changes or automatic retry.
