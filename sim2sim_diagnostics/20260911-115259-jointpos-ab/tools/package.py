"""Validate and package the completed finite experiment; never runs simulation."""
import datetime
import gzip
import hashlib
import json
import math
import pathlib
import struct
import subprocess
import tarfile

out = pathlib.Path(__file__).resolve().parents[1]
archive = out.with_suffix('.tar.gz')
assert not archive.exists(), 'Refusing to overwrite an existing archive'
assert not (out/'manifest.json').exists(), 'Refusing to overwrite a finalized manifest'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    with path.open('rb') as f:
        return stream_digest(f)

def stream_digest(f):
    h = hashlib.sha256()
    while chunk := f.read(1024*1024):
        h.update(chunk)
    return h.hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, list):
        for child in value: finite(child)

baseline = read(out/'baseline.json')
old = pathlib.Path(baseline['old_package'])
assert digest(old/'manifest.json') == baseline['old_package_manifest_sha256']
for item in baseline['old_files']:
    assert digest(old/item['path']) == item['sha256'], item['path']
for item in baseline['protected_original_files']:
    assert digest(pathlib.Path(item['path'])) == item['sha256'], item['path']
model = baseline['model']
assert digest(pathlib.Path(model['path'])) == model['expected'] == model['sha256']
repo = pathlib.Path('/home/lfr/rl_sar')
head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
status = subprocess.check_output(['git','status','--short'], cwd=repo, text=True)
assert head == baseline['repository_head']
assert status == baseline['repository_status']
transfer_head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=out.parents[1], text=True).strip()
assert transfer_head == baseline['transfer_repository_head']
validation = read(out/'validation.json')
assert validation['global']['all_semantic_checks_pass']
assert read(out/'frame_checks.json')['all_checks_pass']
video = read(out/'video_validation.json')
commands = read(out/'commands.json')
assert len(commands) == 4
expected_cases = {'A/stand-a01':1500,'A/walk-stop-a01':2000,
                  'B/stand-a01':1500,'B/walk-stop-a01':828}
assert set(validation['cases']) == set(expected_cases)
checks = {}
cases = {}
float_command = struct.unpack('f',struct.pack('f',0.4))[0]
for key, expected_rows in expected_cases.items():
    folder = out/key
    result = read(folder/'result.json')
    config = read(folder/'effective_config.json')
    process = read(folder/'process.json')
    records = physics = low_ticks = resets = terminal_rows = clipped = 0
    maximum_action = 0.0
    last = None
    with gzip.open(folder/'telemetry.jsonl.gz','rt') as f:
        for line in f:
            row = json.loads(line)
            finite(row)
            assert row['control_step'] == records
            assert abs(row['nominal_policy_time']-records*.02) < 1e-9
            assert abs(row['simulation_time']-records*.02) < 1e-8
            assert row['condition']+'/'+row['case_id'] == key
            moving = key.endswith('walk-stop-a01') and 500 <= records < 1000
            assert row['planned_command'] == [float_command if moving else 0.0,0.0,0.0]
            assert row['command_in_observation'] == row['planned_command']
            assert row['model_input']['shape'] == [1,410]
            assert len(row['model_input']['values']) == 410
            assert len(row['model_raw_action']) == 10
            maximum_action = max(maximum_action,*map(abs,row['model_raw_action']))
            clipped += sum(abs(x)>100 for x in row['model_raw_action'])
            assert row['reset'] == (records == 0)
            resets += bool(row['reset'])
            terminal_rows += bool(row['terminated'])
            for sub in row['physics_substeps']:
                physics += 1
                assert sub['physics_step'] == physics
                assert abs(sub['to_sim_time']-sub['from_sim_time']-.002) < 1e-10
                assert len(sub['actuator_force']) == len(sub['ctrl']) == 10
            low_ticks += len(row['low_level_updates'])
            records += 1
            last = row
    assert records == expected_rows == result['control_records']
    assert physics == result['physics_steps']
    assert low_ticks == result['low_level_ticks']
    assert resets == result['reset_count'] == 1
    assert terminal_rows == 1 and last['terminated']
    assert abs(last['post']['sim_time']-result['final_sim_time']) < 1e-10
    assert video[key]['passes'] and video[key]['decoded_frames'] == records
    assert clipped == 0
    if key == 'B/walk-stop-a01':
        assert result['status'] == 'terminated_early'
        assert result['termination_reason'] == 'fall_upright_tilt_over_75deg'
        assert process['exit_code'] == 3
    else:
        assert result['status'] == 'completed' and process['exit_code'] == 0
    checks[key] = dict(all_raw_numeric_values_finite=True, chronological_rows=True,
        commands_match_original_schedule=True, rows=records, physics_substeps=physics,
        pd_updates=low_ticks, reset_rows=resets, terminal_rows=terminal_rows,
        raw_action_abs_max=maximum_action, action_clip_events=clipped)
    cases[key] = dict(result=result, process=process,
        telemetry=str((folder/'telemetry.jsonl.gz').relative_to(out)),
        effective_configuration=str((folder/'effective_config.json').relative_to(out)),
        video=str((folder/'side_view.mp4').relative_to(out)),
        console=str((folder/'console.log').relative_to(out)))

final_checks = dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    all_evidence_integrity_checks_pass=True, all_cases_completed=False,
    completed_cases=3, retained_early_termination_cases=1,
    cases=checks, protected_original_file_count=len(baseline['protected_original_files']),
    protected_originals_unchanged=True, old_manifest_and_payload_unchanged=True,
    repository_head=head, repository_status=status,
    transfer_repository_head=transfer_head, no_new_git_commits=True,
    model_sha256_matches=True, video_frames=sum(v['decoded_frames'] for v in video.values()))
write(out/'final_checks.json', final_checks)

# Add concise roll/contact and missing-signal disclosure to the one existing report.
statistics = read(out/'statistics.json')
addition = '\n**补充的 roll 与接触窗口核对。** 接触率为策略边界上具名脚几何与地面/台阶的有效接触占比；它不是每个物理子步的接触占比。下列窗口均无 reset，B 行走15–20 s仅统计已采到部分。\n\n'
addition += '|条件/用例/窗口 s|roll 均值±标准差 °|左脚接触率|右脚接触率|双脚均不接触率|\n|---|---:|---:|---:|---:|\n'
for key, case in statistics.items():
    for window in case['windows']:
        if not window['samples']: continue
        lo,hi = window['window_s']
        r = window['roll']
        label = f'{key} {lo}–{hi}' + ('（部分）' if window['coverage'] != 'complete' else '')
        addition += f"|{label}|{r['mean']:.4f} ± {r['std']:.4f}|{100*window['left_contact_fraction']:.2f}%|{100*window['right_contact_fraction']:.2f}%|{100*window['no_foot_contact_fraction']:.2f}%|\n"
addition += '\n**未获取或不适用的信号。** B 行走终止之后的状态、15–20 s剩余部分和所有停止后窗口未获取；未进行新的Isaac配对运行。训练 checkpoint 原文件在本机不可得，不能独立复核其哈希。ONNX外部接口只有obs/actions，没有显式recurrent state或reset mask可记录；历史作为410维obs的一部分已完整保存。脚接触力只采集策略边界，未采集每个2 ms子步的完整接触求解量；每个物理子步的执行器力/广义力已采集。视频为日志离线回放，没有闭环运行时实时渲染视频。实际中途物理reset未触发，B reset后填充路径只由离线构建检查覆盖，不能称为中途reset闭环验证。\n\n最终数值检查见[final_checks.json](final_checks.json)：全部原始数值有限，逐步时序/命令一致；本次所有原始动作均未触发±100裁剪；正式源码/配置/模型、旧包manifest及其全部文件哈希保持一致。\n'
with (out/'report.md').open('a') as f: f.write(addition)

files = []
for path in sorted(out.rglob('*')):
    assert not path.is_symlink(), path
    if not path.is_file(): continue
    assert path.suffix.lower() not in {'.pt','.pth','.onnx'}, path
    files.append(dict(path=str(path.relative_to(out)),size_bytes=path.stat().st_size,sha256=digest(path)))
scene_path = pathlib.Path(commands[0]['argv'][1])
manifest = dict(schema_version=2, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    output_directory=str(out), report='report.md', runtime_configuration='runtime_config.json',
    model={**model,'type':'ONNX','inputs':[dict(name='obs',shape=[1,410],dtype='float32')],
           'outputs':[dict(name='actions',shape=[1,10],dtype='float32')]},
    training_identity=dict(run='2026-09-04_11-16-35',checkpoint='model_50000.pt',
        checkpoint_local_hash_verified=False, task='RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0'),
    repository=str(repo), head=head,
    scene=dict(path=str(scene_path),sha256=digest(scene_path),dependencies='baseline.json'),
    commands='commands.json', cases=cases, all_cases_completed=False,
    evidence_integrity_checks='final_checks.json', semantic_checks='validation.json',
    files=files, self_hash_note='manifest.json excluded from its own file list; manifest.sha256 contains its SHA-256',
    archive=str(archive), archive_digest_path=str(archive)+'.sha256')
write(out/'manifest.json',manifest)
(out/'manifest.sha256').write_text(digest(out/'manifest.json')+'  manifest.json\n')
print('Validated evidence; archiving',len(files),'payload files',flush=True)
with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
    tar.add(out,arcname=out.name)
expected = {out.name+'/'+item['path']:item['sha256'] for item in files}
expected[out.name+'/manifest.json'] = digest(out/'manifest.json')
expected[out.name+'/manifest.sha256'] = digest(out/'manifest.sha256')
seen = set()
with tarfile.open(archive,'r:gz') as tar:
    for item in tar:
        if not item.isfile(): continue
        assert item.name in expected
        assert stream_digest(tar.extractfile(item)) == expected[item.name],item.name
        seen.add(item.name)
assert seen == set(expected)
archive_sha = digest(archive)
pathlib.Path(str(archive)+'.sha256').write_text(archive_sha+'  '+archive.name+'\n')
print(json.dumps(dict(archive=str(archive),sha256=archive_sha,size_bytes=archive.stat().st_size,
    archive_members_verified=len(seen),output_directory=str(out)),ensure_ascii=False),flush=True)
