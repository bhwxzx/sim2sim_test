from pathlib import Path
import json,hashlib,shutil,difflib,subprocess,datetime,shlex
out=Path(__file__).resolve().parent.parent
old=out.parent/'20260910-111515';root=Path('/home/lfr/rl_sar')
def write(p,j):p.write_text(json.dumps(j,indent=2,ensure_ascii=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
expected='63ef20be03cbba9602d7008e84f605e62ff38cbe9ed64ce25366fdab9d08a1a7'
model=root/'policy/LW/robot_lab/leg_loco/policy.onnx'
assert sha(model)==expected
manifest=json.loads((old/'manifest.json').read_text())
for f in manifest['files']:assert sha(old/f['path'])==f['sha256'],f['path']
shutil.copytree(old/'config_snapshot',out/'config_snapshot',dirs_exist_ok=True)
# Reference only machine-readable metadata and original collector; one new report.
for name in ['commands.json','runtime_config.json','statistics.json','baseline.json','manifest.json']:
    shutil.copy2(old/name,out/'source_reference'/name)
shutil.copy2(old/'tools/collect.cpp',out/'source_reference/collect.cpp')
base=json.loads((old/'baseline.json').read_text())
for f in base['files']:assert sha(f['path'])==f['sha256'],f['path']
write(out/'baseline.json',{'created_at':datetime.datetime.now().astimezone().isoformat(),
    'approval':'User approved minimal independent A/B implementation and exactly four closed-loop runs on 2026-09-11.',
    'old_package':str(old),'old_package_manifest_sha256':sha(old/'manifest.json'),
    'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    'repository_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True),
    'transfer_repository_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=out.parent.parent,text=True).strip(),
    'model':{'path':str(model),'sha256':sha(model),'expected':expected,'matches':True},
    'protected_original_files':base['files'],'old_files':manifest['files'],
    'source_reference_files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted((out/'source_reference').glob('*'))]})

sdk_original=(root/'src/rl_sar/library/core/rl_sdk/rl_sdk.cpp').read_text()
sdk=replace(sdk_original,'void RL::ComputeLWObservationInto(','''// Experiment-only hooks. Declared/defined in the independent collector.
extern bool LWDiagnosticZeroJointPosition(std::size_t index, bool baseline_zero);
extern void LWDiagnosticObserveFrame(const LWPolicyRuntimeConfiguration&,
                                    const Observations<float>&,
                                    const std::vector<float>&);

void RL::ComputeLWObservationInto(''')
start=sdk.index('void RL::ComputeLWObservationInto(');end=sdk.index('std::vector<float> RL::ComputeLWObservation(',start)
frame=sdk[start:end]
frame=replace(frame,'policy_configuration.wheel_mask[i] != 0','LWDiagnosticZeroJointPosition(i, policy_configuration.wheel_mask[i] != 0)')
frame=replace(frame,'        value = clamp(value, -clip, clip);\n    }\n}', '        value = clamp(value, -clip, clip);\n    }\n    LWDiagnosticObserveFrame(policy_configuration, policy_obs, output);\n}')
sdk=sdk[:start]+frame+sdk[end:]
(out/'experimental/rl_sdk.cpp').write_text(sdk)
(out/'experimental/rl_sdk.patch').write_text(''.join(difflib.unified_diff(sdk_original.splitlines(True),sdk.splitlines(True),fromfile='original/rl_sdk.cpp',tofile='experimental/rl_sdk.cpp')))

original=(old/'tools/collect.cpp').read_text();src=original
src=replace(src,'struct CaptureModel : InferenceRuntime::Model {',r'''// Process-local experiment state; no deployment configuration is changed.
namespace experiment {
bool isaac = false;
bool pairing = false;
std::size_t left_foot = 0, right_wheel = 0, left_wheel = 0;
bool names_resolved = false;
const LWPolicyRuntimeConfiguration* frame_config = nullptr;
Observations<float> frame_state;
std::vector<float> actual_frame;
}
bool LWDiagnosticZeroJointPosition(std::size_t index, bool baseline_zero) {
    if (!experiment::isaac) return baseline_zero;
    if (!experiment::names_resolved) throw std::runtime_error("joint names not resolved");
    if (index == experiment::left_foot) return true;
    if (index == experiment::right_wheel) return false;
    return baseline_zero;
}
void LWDiagnosticObserveFrame(const LWPolicyRuntimeConfiguration& cfg,
                              const Observations<float>& obs,
                              const std::vector<float>& frame) {
    if (experiment::pairing) return;
    experiment::frame_config = &cfg;
    experiment::frame_state = obs;
    experiment::actual_frame = frame;
}
struct CaptureModel : InferenceRuntime::Model {''')
src=replace(src,'    int calls=0;','''    int calls=0;
    J* record=nullptr;
    std::function<void(const std::vector<float>&)> before_model;
    std::vector<float> previous_input;''')
src=replace(src,'        inner->forwardInto(x,n,y);',r'''        if(record) {
            (*record)["model_input"]={{"shape",{1,410}},{"dtype","float32"},{"values",input}};
            (*record)["model_input_capture_phase"]="synchronous copy immediately before underlying ONNX forward";
        }
        if(before_model) before_model(input);
        if(input.size()!=410) throw std::runtime_error("input width mismatch");
        if(previous_input.empty()) {
            for(int frame=0;frame<10;++frame)
                if(!std::equal(input.begin()+frame*41,input.begin()+(frame+1)*41,input.begin()+369))
                    throw std::runtime_error("initial history is not repeated current condition frame");
        } else if(!std::equal(input.begin(),input.begin()+369,previous_input.begin()+41)) {
            throw std::runtime_error("history time-major shift mismatch");
        }
        previous_input=input;
        inner->forwardInto(x,n,y);''')
src=replace(src,'        raw.assign(y.data,y.data+y.size); ++calls;','''        raw.assign(y.data,y.data+y.size); ++calls;
        if(record) (*record)["model_raw_action"]=raw;''')
src=replace(src,'    std::vector<int> jids;','''    std::vector<int> jids;
    J joint_mapping_evidence=J::array();''')
src=replace(src,'        PreloadModel(policy_key);',r'''        const std::vector<std::string> expected={"right_hip_joint","left_hip_joint","right_thigh_joint","left_thigh_joint",
            "right_shank_joint","left_shank_joint","right_foot_joint","left_foot_joint","right_wheel_joint","left_wheel_joint"};
        for(std::size_t index=0;index<jids.size();++index) {
            int jid=jids[index];
            std::string jname=name(m,mjOBJ_JOINT,jid);
            if(jid<0 || jname!=expected.at(index)) throw std::runtime_error("policy joint order does not match verified contract");
            int sensor=-1;
            for(int si=0;si<m->nsensor;++si)
                if(m->sensor_type[si]==mjSENS_JOINTPOS && m->sensor_objtype[si]==mjOBJ_JOINT && m->sensor_objid[si]==jid) {
                    if(sensor>=0) throw std::runtime_error("ambiguous joint position sensor");
                    sensor=si;
                }
            if(sensor<0) throw std::runtime_error("missing joint position sensor");
            joint_mapping_evidence.push_back({{"policy_index",index},{"joint_name",jname},{"config_mapping",b.joint_mapping[index]},
                {"mj_joint_id",jid},{"mj_qpos_adr",m->jnt_qposadr[jid]},{"mj_dof_adr",m->jnt_dofadr[jid]},
                {"position_sensor_id",sensor},{"position_sensor_adr",m->sensor_adr[sensor]}});
            if(jname=="left_foot_joint") experiment::left_foot=index;
            if(jname=="right_wheel_joint") experiment::right_wheel=index;
            if(jname=="left_wheel_joint") experiment::left_wheel=index;
        }
        experiment::names_resolved=true;
        PreloadModel(policy_key);''')
src=replace(src,'        preloaded_models_[policy_key]=capture; PreloadLWPolicyContext(policy_key);',r'''        preloaded_models_[policy_key]=capture; PreloadLWPolicyContext(policy_key);
        capture->before_model=[this](const std::vector<float>& input) {
            if(input.size()!=410 || !experiment::frame_config || experiment::actual_frame.size()!=41)
                throw std::runtime_error("frame capture unavailable");
            const auto& cfg=*experiment::frame_config;
            const auto raw_state=experiment::frame_state;
            const auto current_frame=experiment::actual_frame;
            if(!std::equal(current_frame.begin(),current_frame.end(),input.begin()+369))
                throw std::runtime_error("frame reaching model differs from constructed frame");
            std::vector<float> paired_a(41),paired_b(41);
            const bool saved_condition=experiment::isaac;
            experiment::pairing=true;
            try {
                experiment::isaac=false; ComputeLWObservationInto(cfg,raw_state,nullptr,paired_a);
                experiment::isaac=true; ComputeLWObservationInto(cfg,raw_state,nullptr,paired_b);
            } catch(...) {experiment::isaac=saved_condition;experiment::pairing=false;throw;}
            experiment::isaac=saved_condition;experiment::pairing=false;
            std::vector<int> changed;
            for(int i=0;i<41;++i) if(paired_a[i]!=paired_b[i]) {
                changed.push_back(i);
                if(i!=9+int(experiment::left_foot) && i!=9+int(experiment::right_wheel))
                    throw std::runtime_error("same-state observation changed outside approved columns");
            }
            if(current_frame!=(saved_condition?paired_b:paired_a)) throw std::runtime_error("selected observation path mismatch");
            // Confirm raw joint angles are the named adapter's jointpos sensors,
            // not native guessed indices or wheel velocities.
            for(const auto& mapping:joint_mapping_evidence) {
                int i=mapping["policy_index"],adr=mapping["position_sensor_adr"];
                if(raw_state.dof_pos[i]!=static_cast<float>(d->sensordata[adr]))
                    throw std::runtime_error("named joint angle and policy state mismatch");
            }
            if(capture->record) {
                auto& r=*capture->record;
                r["condition"]=saved_condition?"B":"A";
                r["observation_frame"]={{"joint_pos_before_noise",arr(current_frame.data()+9,10)},
                    {"joint_pos_sent",arr(input.data()+378,10)},{"noise_applied",false},
                    {"raw_q_policy_float32",raw_state.dof_pos},{"default_q_policy_float32",cfg.default_dof_pos},
                    {"dq_policy_float32",raw_state.dof_vel},{"joint_mapping",joint_mapping_evidence},
                    {"quaternion_wxyz_body_to_world",raw_state.base_quat},
                    {"commands",raw_state.commands},{"previous_action",raw_state.actions},
                    {"gait_phase",raw_state.gait_phase}};
                r["same_state_pair"]={{"A_frame",paired_a},{"B_frame",paired_b},{"changed_columns",changed},
                    {"allowed_columns",{9+experiment::left_foot,9+experiment::right_wheel}},
                    {"stage","complete per-frame observation builder, scale/clip included, before history; no additional inference or physics"}};
            }
        };''')
src=replace(src,'        return j;\n    }\n    J state()',r'''        j["ab_experiment"]={{"condition",experiment::isaac?"B":"A"},{"noise_applied",false},
            {"joint_mapping",joint_mapping_evidence},{"selection_scope","joint-position zero predicate only"},
            {"application_stage","per-frame DofPosition before scaling/clipping and history insertion"}};
        return j;
    }
    J state()''')
src=replace(src,'        J j={{"sim_time",d->time}',r'''        std::vector<double> origin_body(3);
        for(int i=0;i<3;++i) {origin_body[i]=0;for(int k=0;k<3;++k)origin_body[i]+=R[3*k+i]*d->qvel[k];}
        const auto* qw=d->qpos+3;
        const double backward=std::asin(std::clamp(2*(qw[1]*qw[3]-qw[0]*qw[2]),-1.0,1.0))*180/M_PI;
        J j={{"sim_time",d->time}''')
src=replace(src,'{"base_linear_velocity_world",arr(vw+3,3)},{"base_linear_velocity_body",arr(vb+3,3)},','''{"base_com_linear_velocity_world",arr(vw+3,3)},{"base_com_linear_velocity_body",arr(vb+3,3)},
            {"base_origin_linear_velocity_world",arr(d->qvel,3)},{"base_origin_linear_velocity_body",origin_body},''')
src=replace(src,'{"backward_lean_deg",-pitch*180/M_PI}','{"backward_lean_deg",backward}')
src=replace(src,'if(argc!=4){std::cerr<<"usage: collect XML CASE_ID OUTPUT_DIR\\n";return 2;}', 'if(argc!=5){std::cerr<<"usage: collect XML CASE_ID OUTPUT_DIR A|B\\n";return 2;}')
src=replace(src,'    gzFile log=nullptr; J result=', '''    const std::string condition=argv[4];
    if(condition!="A" && condition!="B") {std::cerr<<"condition must be A or B\\n";return 2;}
    experiment::isaac=condition=="B";
    J pending_row;
    gzFile log=nullptr; J result=''')
src=replace(src,'            J row={{"case_id",cid}', '            pending_row=J{{"case_id",cid}')
src=replace(src,'{"pre",h.state()},{"low_level_updates",J::array()},{"physics_substeps",J::array()}};', '''{"pre",h.state()},{"low_level_updates",J::array()},{"physics_substeps",J::array()},
                {"condition",condition},{"model_input",nullptr},{"model_raw_action",nullptr},{"new_policy_targets",nullptr},{"post",nullptr}};
            J& row=pending_row;
            h.capture->record=&row;''')
src=replace(src,'            row["safety_events"]=h.safety_events;emit(row);++step;', '            row["safety_events"]=h.safety_events;emit(row);pending_row=J();h.capture->record=nullptr;++step;')
src=replace(src,'        result={{"case_id",cid},{"status",why.empty()', '        result={{"condition",condition},{"case_id",cid},{"status",why.empty()')
src=replace(src,'        if(log){std::string s=J(', '''        if(log && !pending_row.is_null()) {
            pending_row["terminated"]=true;pending_row["termination_reason"]=std::string("program_exception: ")+e.what();
            std::string line=pending_row.dump()+"\\n";gzwrite(log,line.data(),line.size());
        }
        if(log){std::string s=J(''')
(out/'tools/collect.cpp').write_text(src)
(out/'experimental/collect.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),src.splitlines(True),fromfile='old/collect.cpp',tofile='new/collect.cpp')))

build=(old/'tools/build_collect.py').read_text()
build=replace(build,"str(root/'src/rl_sar/library/core'/s) for s in sources", "str(out/'experimental/rl_sdk.cpp') if s=='rl_sdk/rl_sdk.cpp' else str(root/'src/rl_sar/library/core'/s) for s in sources")
(out/'tools/build_collect.py').write_text(build)

commands=[]
for cond in ['A','B']:
    for cid in ['stand-a01','walk-stop-a01']:
        case=out/cond/cid
        argv=[str(out/'tools/collect'),str(root/'src/rl_sar_zoo/LW_description/mjcf/scene.xml'),cid,str(case),cond]
        commands.append({'condition':cond,'case_id':cid,'cwd':str(root),'argv':argv,
            'shell_command':shlex.join(argv)+' > '+shlex.quote(str(case/'console.log'))+' 2>&1'})
write(out/'commands.json',commands)
write(out/'experiment_plan.json',{'matrix':commands,'seed':42,'added_observation_noise':False,
    'command_source':{'commands_json':str(old/'commands.json'),'schedule_implementation':str(old/'tools/collect.cpp'),
        'verified_from_old_telemetry':True},
    'schedules':{'stand-a01':[[0,30,[0,0,0]]],'walk-stop-a01':[[0,10,[0,0,0]],[10,20,[0.4,0,0]],[20,40,[0,0,0]]]},
    'four_runs_only':True,'on_failure':'preserve logs, stop the affected case, no tuning or retry',
    'unchanged':'model, XML, physics (including freejoint frictionloss=.2), action mapping, PD, scaling, normalization, history algorithm, seed, reset/FSM path, scheduling and action delay'})
print(out)
