import argparse,pathlib,json,hashlib,subprocess,datetime,time,shlex
p=argparse.ArgumentParser();p.add_argument('--plan',type=pathlib.Path,required=True);p.add_argument('--collector',type=pathlib.Path,required=True);a=p.parse_args()
plan=json.loads(a.plan.read_text());out=pathlib.Path(plan['output'])
assert hashlib.sha256(pathlib.Path(plan['model']['path']).read_bytes()).hexdigest()==plan['model']['sha256']
assert not (out/'commands.json').exists(), 'Matrix has already been started; no retries'
commands=[]
for case in plan['cases']:
    directory=out/'cases'/case['case_id'];directory.mkdir(parents=True)
    cmd=[str(a.collector),str(a.plan.resolve()),case['case_id'],str(directory)]
    commands.append({'case_id':case['case_id'],'argv':cmd,'cwd':plan['repository'],'stdout_stderr':str(directory/'console.log'),'shell_command':shlex.join(cmd)+' > '+shlex.quote(str(directory/'console.log'))+' 2>&1'})
(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for spec in commands:
    folder=out/'cases'/spec['case_id'];start=time.monotonic();began=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (folder/'console.log').open('x') as log:
        try:
            r=subprocess.run(spec['argv'],cwd=spec['cwd'],stdout=log,stderr=subprocess.STDOUT,timeout=plan['wall_timeout_seconds']+15)
            code=r.returncode
        except subprocess.TimeoutExpired:
            code=124
            if not (folder/'result.json').exists():(folder/'result.json').write_text(json.dumps({'case_id':spec['case_id'],'status':'failed','termination_reason':'supervisor_wall_timeout','requested_sim_duration':120,'final_sim_time':None,'control_records':None})+'\n')
    (folder/'process.json').write_text(json.dumps({**spec,'started_at':began,'wall_seconds':time.monotonic()-start,'exit_code':code},indent=2)+'\n')
    print(spec['case_id'],'exit',code,(folder/'result.json').read_text(),flush=True)
