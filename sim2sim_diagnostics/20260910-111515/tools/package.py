from pathlib import Path
import json,hashlib,gzip,subprocess,tarfile,math,datetime
out=Path(__file__).resolve().parent.parent;root=Path('/home/lfr/rl_sar')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
base=json.loads((out/'baseline.json').read_text());validation=json.loads((out/'validation.json').read_text());delivery=json.loads((out/'delivery_validation.json').read_text())
checks={'original_files_unchanged':all(sha(x['path'])==x['sha256'] for x in base['files']),
    'original_models_unchanged':all(sha(x['path'])==x['sha256'] for x in base['models']),
    'git_status_after':subprocess.check_output(['git','status','--short'],cwd=root,text=True),
    'no_policy_weights_in_output':not any(p.suffix in ['.pt','.onnx','.pth'] for p in out.rglob('*')),
    'case_validation':{cid:validation[cid]['all_checks_pass'] and delivery[cid]['checks_pass'] for cid in validation}}
checks['git_status_unchanged']=checks['git_status_after']==base['git_status_short']
prefix=[]
for cid,n in [('stand-a01',1500),('walk-stop-a01',2000)]:
    with gzip.open(out/'cases'/cid/'telemetry.jsonl.gz','rt') as f:rows=[json.loads(line) for line in f]
    with gzip.open(out/'cases'/cid/'telemetry_derived.jsonl.gz','rt') as f:derived=[json.loads(line) for line in f]
    def finite(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,list):return all(map(finite,value))
        if isinstance(value,dict):return all(map(finite,value.values()))
        return True
    assert len(rows)==len(derived)==n and all(finite(row) for row in rows+derived)
    assert [r['control_step'] for r in rows]==list(range(n))
    assert all(r['control_step']==d['control_step'] and r['case_id']==d['case_id'] for r,d in zip(rows,derived))
    assert all(abs(rows[i]['post']['sim_time']-rows[i+1]['pre']['sim_time'])<1e-10 for i in range(n-1))
    prefix.append([{key:row[key] for key in ['pre','model_input','model_raw_action','new_policy_targets','post']} for row in rows[:500]])
checks['cases_identical_before_command_divergence_10s']=prefix[0]==prefix[1]
checks['all_pass']=checks['original_files_unchanged'] and checks['original_models_unchanged'] and checks['git_status_unchanged'] and checks['no_policy_weights_in_output'] and all(checks['case_validation'].values()) and checks['cases_identical_before_command_divergence_10s']
write(out/'final_checks.json',checks)
assert checks['all_pass'],checks
expected='63ef20be03cbba9602d7008e84f605e62ff38cbe9ed64ce25366fdab9d08a1a7'
manifest={'schema_version':1,'created_at':datetime.datetime.now().astimezone().isoformat(),'output_directory':str(out),
    'report':'report.md','runtime_configuration':'runtime_config.json','statistics':'statistics.json',
    'repository':str(root),'head':base['head'],'collection_mode':'independent deterministic diagnostic reusing repository C++ control implementation; interactive executable not run',
    'model':{'path':base['models'][0]['path'],'sha256':base['models'][0]['sha256'],'expected_sha256':expected,
             'matches':base['models'][0]['sha256']==expected,'type':'ONNX','inputs':[{'name':'obs','shape':[1,410],'dtype':'float32'}],
             'outputs':[{'name':'actions','shape':[1,10],'dtype':'float32'}],'weight_included':False},
    'checkpoint':base['checkpoint'],'scene':next(x for x in base['files'] if x['path'].endswith('/scene.xml')),
    'cases':{},'files':[],
    'self_hash_note':'manifest.json cannot contain its own digest; see manifest.sha256. All other payload files below are individually hashed.',
    'archive':str(out.with_suffix('.tar.gz')),'archive_digest_path':str(out.with_suffix('.tar.gz.sha256'))}
for cid in validation:
    result=json.loads((out/'cases'/cid/'result.json').read_text())
    manifest['cases'][cid]={**result,'telemetry':f'cases/{cid}/telemetry.jsonl.gz','derived_telemetry':f'cases/{cid}/telemetry_derived.jsonl.gz',
        'console':f'cases/{cid}/console.log','video':f'cases/{cid}/side_view.mp4','effective_config':f'cases/{cid}/effective_config.json',
        'video_verified':delivery[cid]['video']['complete']}
for p in sorted(out.rglob('*')):
    if not p.is_file() or p.name in ['manifest.json','manifest.sha256']:continue
    manifest['files'].append({'path':str(p.relative_to(out)),'sha256':sha(p),'size_bytes':p.stat().st_size})
write(out/'manifest.json',manifest)
(out/'manifest.sha256').write_text(sha(out/'manifest.json')+'  manifest.json\n')
archive=out.with_suffix('.tar.gz')
if archive.exists():raise RuntimeError('refuse archive overwrite')
with tarfile.open(archive,'x:gz',compresslevel=6) as tar:tar.add(out,arcname=out.name)
archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
with tarfile.open(archive,'r:gz') as tar:
    archived_manifest=json.load(tar.extractfile(out.name+'/manifest.json'))
    for f in archived_manifest['files']:
        data=tar.extractfile(out.name+'/'+f['path']).read()
        assert hashlib.sha256(data).hexdigest()==f['sha256'],f['path']
print(json.dumps({'output_directory':str(out),'archive':str(archive),'archive_size_bytes':archive.stat().st_size,
    'archive_sha256':sha(archive),'payload_files':len(manifest['files']),'checks':checks,'archive_all_file_digests_verified':True},indent=2))
