#include "lw_runtime_core.hpp"
#include "lw_mujoco_control_adapter.hpp"
#include "fsm_LW.hpp"
#include <nlohmann/json.hpp>
#include <zlib.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>

// Independent diagnostic executable. All policy/control computations remain
// in the repository sources linked by build_collect.py.
using J = nlohmann::json;
template<class T> J arr(const T* p, int n) { return std::vector<T>(p,p+n); }
std::string name(const mjModel* m, int type, int id) {
    const char* n=mj_id2name(m,type,id); return n ? n : "";
}
void write_json(const std::string& p,const J& j) {
    std::ofstream f(p); f << j.dump(2) << '\n';
    if (!f) throw std::runtime_error("write failed: "+p);
}
// Process-local experiment state; no deployment configuration is changed.
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
struct CaptureModel : InferenceRuntime::Model {
    std::shared_ptr<InferenceRuntime::Model> inner;
    std::vector<float> input, raw;
    int calls=0;
    J* record=nullptr;
    std::function<void(const std::vector<float>&)> before_model;
    std::vector<float> previous_input;
    explicit CaptureModel(std::shared_ptr<InferenceRuntime::Model> m):inner(m) {}
    bool load(const std::string& p) override { return inner->load(p); }
    bool is_loaded() const override { return inner->is_loaded(); }
    std::string get_model_type() const override { return inner->get_model_type(); }
    const std::vector<InferenceRuntime::TensorMetadata>& input_metadata() const override { return inner->input_metadata(); }
    const std::vector<InferenceRuntime::TensorMetadata>& output_metadata() const override { return inner->output_metadata(); }
    void forwardInto(const InferenceRuntime::TensorView* x, std::size_t n, InferenceRuntime::MutableTensorView y) override {
        if (n!=1) throw std::runtime_error("unexpected input count");
        input.assign(x[0].data,x[0].data+x[0].size);
        if(record) {
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
        inner->forwardInto(x,n,y);
        raw.assign(y.data,y.data+y.size); ++calls;
        if(record) (*record)["model_raw_action"]=raw;
    }
};
J metadata(const std::vector<InferenceRuntime::TensorMetadata>& ms) {
    J j=J::array(); for(auto& m:ms) j.push_back({{"name",m.name},{"shape",m.shape},{"dtype",m.element_type==InferenceRuntime::TensorElementType::Float32?"float32":"unknown"}}); return j;
}

struct Diagnostic : RL {
    mjModel* m=nullptr; mjData* d=nullptr; mjData* scratch=nullptr;
    LWRuntimeCore core;
    std::unique_ptr<LWMuJoCoControlAdapter> adapter;
    std::shared_ptr<CaptureModel> capture;
    std::vector<float> candidates=std::vector<float>(10), bounded=std::vector<float>(10);
    std::vector<int> jids;
    J joint_mapping_evidence=J::array();
    J safety_events=J::array();
    const std::string policy_key="LW/robot_lab/leg_loco";
    explicit Diagnostic(const std::string& xml) {
        char error[4096]{}; m=mj_loadXML(xml.c_str(),nullptr,error,sizeof(error));
        if(!m) throw std::runtime_error(error);
        d=mj_makeData(m); scratch=mj_makeData(m);
        if(!d || !scratch) throw std::runtime_error("mj_makeData failed");
        robot_name="LW"; ang_vel_axis="body";
        SetPolicyRoot("/home/lfr/rl_sar/policy"); ReadYaml(robot_name,"base.yaml");
        SetLWBaseRuntimeConfiguration(LWValidatedBaseConfiguration(params.config_node,ResolvePolicyPath("LW/base.yaml")));
        const auto& b=GetLWBaseRuntimeConfiguration();
        adapter=std::make_unique<LWMuJoCoControlAdapter>(*m,b.joint_names,b.joint_mapping);
        for(int i:b.joint_mapping) jids.push_back(mj_name2id(m,mjOBJ_JOINT,b.joint_names[i].c_str()));
        const std::vector<std::string> expected={"right_hip_joint","left_hip_joint","right_thigh_joint","left_thigh_joint",
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
        PreloadModel(policy_key);
        capture=std::make_shared<CaptureModel>(preloaded_models_.at(policy_key));
        preloaded_models_[policy_key]=capture; PreloadLWPolicyContext(policy_key);
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
        };
        InitJointNum(b.num_dofs); InitControl(); control.gait_frequency=b.gait_command[0];
        core.bind(*this,[this](const LWSafetyDecision& dec,const std::string& reason){
            safety_events.push_back({{"sim_time",d->time},{"action",LWSafetyActionName(dec.action)},{"reason",reason}});
            std::cerr << "Safety " << LWSafetyActionName(dec.action) << " " << reason << '\n';
        });
        fsm=*FSMManager::GetInstance().CreateFSM("LW",this);
    }
    ~Diagnostic() { if(scratch)mj_deleteData(scratch); if(d)mj_deleteData(d); if(m)mj_deleteModel(m); }
    std::vector<float> Forward() override { return core.forward(); }
    void GetState(RobotState<float>* s) override {
        if(!adapter->ReadState(*d,s->motor_state.q,s->motor_state.dq,s->motor_state.tau_est,s->imu.quaternion,s->imu.gyroscope)) throw std::runtime_error("state read failed");
    }
    void SetCommand(const RobotCommand<float>* c) override {
        auto& a=c->motor_command;
        auto v=adapter->ApplyCommand(*d,a.q,a.dq,a.tau,a.kp,a.kd,GetLWBaseRuntimeConfiguration().torque_limits,candidates,bounded);
        if(!v.valid()) throw std::runtime_error("ApplyCommand rejected torque");
    }
    void HandleLWPolicyOutputFault(LWPolicyOutputStatus s) noexcept override { core.handlePolicyOutputFault(s); }
    void reset() {
        std::srand(42);
        int k=mj_name2id(m,mjOBJ_KEY,"home_leg"); if(k<0)throw std::runtime_error("home_leg missing");
        mj_resetDataKeyframe(m,d,k); mj_forward(m,d); GetState(&robot_state);
    }
    void activate() {
        // Same LegLocomotion Enter() as normal program; no interpolation or
        // alternate posture is applied to the home_leg reset state.
        fsm.RequestStateChange("RLFSMStateRLLocomotion_Leg");
        StateController(&robot_state,&robot_command,false);
        core.publishInitialPolicyInput();
    }
    J config() {
        const auto& b=GetLWBaseRuntimeConfiguration();
        auto def=GetLWPolicyDefinition(policy_key); const auto& c=def->runtime;
        J j={{"mujoco_version",mj_versionString()},{"onnxruntime_version",OrtGetApiBase()->GetVersionString()},
            {"physics_timestep",m->opt.timestep},{"base_dt_float32",b.dt},{"decimation",b.decimation},{"policy_period_float32",c.period_seconds},
            {"model_path",ResolvePolicyPath(policy_key+"/policy.onnx")},{"model_type",capture->get_model_type()},
            {"model_inputs",metadata(capture->input_metadata())},{"model_outputs",metadata(capture->output_metadata())},
            {"gravity",arr(m->opt.gravity,3)},{"wind",arr(m->opt.wind,3)},{"integrator",m->opt.integrator},
            {"solver",m->opt.solver},{"iterations",m->opt.iterations},{"tolerance",m->opt.tolerance},{"cone",m->opt.cone},
            {"impratio",m->opt.impratio},{"density",m->opt.density},{"viscosity",m->opt.viscosity},
            {"enableflags",m->opt.enableflags},{"disableflags",m->opt.disableflags},
            {"nq",m->nq},{"nv",m->nv},{"nu",m->nu},{"nbody",m->nbody},
            {"joint_names_config",b.joint_names},{"joint_mapping",b.joint_mapping},{"policy_joint_ids",jids},
            {"rl_kp",c.rl_kp},{"rl_kd",c.rl_kd},{"default_dof_pos",c.default_dof_pos},
            {"action_scale",c.action_scale},{"clip_actions_lower",c.clip_actions_lower},{"clip_actions_upper",c.clip_actions_upper},
            {"torque_limits",c.torque_limits},{"wheel_indices",c.wheel_indices},
            {"ang_vel_scale",c.ang_vel_scale},{"dof_pos_scale",c.dof_pos_scale},{"dof_vel_scale",c.dof_vel_scale},
            {"commands_scale",c.commands_scale},{"clip_obs",c.clip_obs},{"history_indices",c.observations_history},
            {"history_priority",c.observations_history_priority},{"gait_frequency",control.gait_frequency},
            {"merged_yaml",YAML::Dump(def->params.config_node)},{"initial_qpos",arr(d->qpos,m->nq)},
            {"initial_qvel",arr(d->qvel,m->nv)},{"initial_ctrl",arr(d->ctrl,m->nu)}};
        for(int i=0;i<m->nbody;++i) j["bodies"].push_back({{"id",i},{"name",name(m,mjOBJ_BODY,i)},{"parent_id",m->body_parentid[i]},
            {"pos_parent",arr(m->body_pos+3*i,3)},{"quat_parent_wxyz",arr(m->body_quat+4*i,4)},
            {"mass",m->body_mass[i]},{"com_body",arr(m->body_ipos+3*i,3)},
            {"principal_inertia",arr(m->body_inertia+3*i,3)},{"inertial_quat_wxyz",arr(m->body_iquat+4*i,4)}});
        for(int i=0;i<m->njnt;++i) {int a=m->jnt_dofadr[i]; int n=m->jnt_type[i]==mjJNT_FREE?6:1;
            j["joints"].push_back({{"id",i},{"name",name(m,mjOBJ_JOINT,i)},{"type",m->jnt_type[i]},{"body_id",m->jnt_bodyid[i]},
                {"qpos_adr",m->jnt_qposadr[i]},{"dof_adr",a},{"axis",arr(m->jnt_axis+3*i,3)},
                {"limited",bool(m->jnt_limited[i])},{"range",arr(m->jnt_range+2*i,2)},
                {"stiffness",m->jnt_stiffness[i]},{"damping",arr(m->dof_damping+a,n)},
                {"armature",arr(m->dof_armature+a,n)},{"frictionloss",arr(m->dof_frictionloss+a,n)}});
        }
        for(int i=0;i<m->nu;++i) j["actuators"].push_back({{"id",i},{"name",name(m,mjOBJ_ACTUATOR,i)},
            {"trntype",m->actuator_trntype[i]},{"trnid",arr(m->actuator_trnid+2*i,2)},
            {"dyntype",m->actuator_dyntype[i]},{"gaintype",m->actuator_gaintype[i]},{"biastype",m->actuator_biastype[i]},
            {"gainprm",arr(m->actuator_gainprm+10*i,10)},{"biasprm",arr(m->actuator_biasprm+10*i,10)},
            {"gear",arr(m->actuator_gear+6*i,6)},{"ctrllimited",bool(m->actuator_ctrllimited[i])},{"ctrlrange",arr(m->actuator_ctrlrange+2*i,2)},
            {"forcelimited",bool(m->actuator_forcelimited[i])},{"forcerange",arr(m->actuator_forcerange+2*i,2)}});
        for(int i=0;i<m->ngeom;++i) j["geoms"].push_back({{"id",i},{"name",name(m,mjOBJ_GEOM,i)},{"body_id",m->geom_bodyid[i]},
            {"type",m->geom_type[i]},{"pos_body",arr(m->geom_pos+3*i,3)},{"quat_body_wxyz",arr(m->geom_quat+4*i,4)},
            {"size",arr(m->geom_size+3*i,3)},{"friction",arr(m->geom_friction+3*i,3)},{"contype",m->geom_contype[i]},
            {"conaffinity",m->geom_conaffinity[i]},{"condim",m->geom_condim[i]},{"solref",arr(m->geom_solref+2*i,2)},
            {"solimp",arr(m->geom_solimp+5*i,5)},{"mesh_id",m->geom_dataid[i]}});
        for(int i=0;i<m->nsite;++i) j["sites"].push_back({{"id",i},{"name",name(m,mjOBJ_SITE,i)},{"body_id",m->site_bodyid[i]},
            {"pos_body",arr(m->site_pos+3*i,3)},{"quat_body_wxyz",arr(m->site_quat+4*i,4)}});
        for(int i=0;i<m->nsensor;++i) j["sensors"].push_back({{"id",i},{"name",name(m,mjOBJ_SENSOR,i)},
            {"type",m->sensor_type[i]},{"objtype",m->sensor_objtype[i]},{"objid",m->sensor_objid[i]},
            {"adr",m->sensor_adr[i]},{"dim",m->sensor_dim[i]},{"noise",m->sensor_noise[i]},{"cutoff",m->sensor_cutoff[i]}});
        j["ab_experiment"]={{"condition",experiment::isaac?"B":"A"},{"noise_applied",false},
            {"joint_mapping",joint_mapping_evidence},{"selection_scope","joint-position zero predicate only"},
            {"application_stage","per-frame DofPosition before scaling/clipping and history insertion"}};
        return j;
    }
    J state() {
        // Refresh only a copy: keep native mj_step sensor evaluation timing.
        mj_copyData(scratch,m,d); mj_forward(m,scratch);
        const int base=mj_name2id(m,mjOBJ_BODY,"base_link");
        const mjtNum* R=scratch->xmat+9*base;
        mjtNum vw[6],vb[6]; mj_objectVelocity(m,scratch,mjOBJ_BODY,base,vw,0); mj_objectVelocity(m,scratch,mjOBJ_BODY,base,vb,1);
        std::vector<double> pg(3), q,dq,qa;
        double gn=mju_norm3(m->opt.gravity);
        for(int i=0;i<3;++i) {pg[i]=0;for(int k=0;k<3;++k)pg[i]+=R[3*k+i]*m->opt.gravity[k]/gn;}
        for(int id:jids) {q.push_back(d->qpos[m->jnt_qposadr[id]]);dq.push_back(d->qvel[m->jnt_dofadr[id]]);qa.push_back(d->qfrc_actuator[m->jnt_dofadr[id]]);}
        double pitch=std::asin(std::clamp(-R[6],-1.0,1.0)); double roll=std::atan2(R[7],R[8]);
        std::vector<double> origin_body(3);
        for(int i=0;i<3;++i) {origin_body[i]=0;for(int k=0;k<3;++k)origin_body[i]+=R[3*k+i]*d->qvel[k];}
        const auto* qw=d->qpos+3;
        const double backward=std::asin(std::clamp(2*(qw[1]*qw[3]-qw[0]*qw[2]),-1.0,1.0))*180/M_PI;
        J j={{"sim_time",d->time},{"qpos_mujoco",arr(d->qpos,m->nq)},{"qvel_mujoco",arr(d->qvel,m->nv)},
            {"base_position_world",arr(scratch->xpos+3*base,3)},{"base_quaternion_wxyz_body_to_world",arr(d->qpos+3,4)},
            {"base_com_linear_velocity_world",arr(vw+3,3)},{"base_com_linear_velocity_body",arr(vb+3,3)},
            {"base_origin_linear_velocity_world",arr(d->qvel,3)},{"base_origin_linear_velocity_body",origin_body},
            {"base_angular_velocity_world",arr(vw,3)},{"base_angular_velocity_body",arr(vb,3)},
            {"projected_gravity_from_raw_pose",pg},{"pitch_ZYX_rad",pitch},{"backward_lean_deg",backward},{"roll_ZYX_deg",roll*180/M_PI},
            {"q_policy_order",q},{"dq_policy_order",dq},{"ctrl",arr(d->ctrl,m->nu)},
            {"actuator_force_last_dynamics",arr(d->actuator_force,m->nu)},
            {"joint_actuator_generalized_force_last_dynamics_policy_order",qa},
            {"xfrc_applied_world",arr(d->xfrc_applied,6*m->nbody)},{"qfrc_applied",arr(d->qfrc_applied,m->nv)},
            {"contact_evaluation","mj_forward on copied data at this qpos/qvel"},{"contacts",J::array()}};
        for(int i=0;i<scratch->ncon;++i) {auto& c=scratch->contact[i];mjtNum f[6]{};mj_contactForce(m,scratch,i,f);
            j["contacts"].push_back({{"geom1",c.geom1},{"geom2",c.geom2},{"distance",c.dist},{"position_world",arr(c.pos,3)},
                {"frame_world_rows",arr(c.frame,9)},{"force_contact_frame",arr(f,6)},{"efc_address",c.efc_address}});
        }
        for(const char* s:{"left_foot_site","right_foot_site"}) {int id=mj_name2id(m,mjOBJ_SITE,s);j["foot_sites_world"][s]=id>=0?arr(scratch->site_xpos+3*id,3):J(nullptr);}
        return j;
    }
    J command() {auto& c=robot_command.motor_command;return {{"q_target",c.q},{"dq_target",c.dq},{"feedforward_tau",c.tau},{"kp",c.kp},{"kd",c.kd},{"torque_candidates_policy_order",candidates},{"bounded_torque_policy_order",bounded},{"ctrl_mujoco_order",arr(d->ctrl,m->nu)}};}
    std::string stop_reason() {
        for(int i=0;i<m->nq;++i) if(!std::isfinite(d->qpos[i]))return "nonfinite_qpos";
        for(int i=0;i<m->nv;++i) if(!std::isfinite(d->qvel[i]))return "nonfinite_qvel";
        for(int i=0;i<mjNWARNING;++i)if(d->warning[i].number>0)return "mujoco_warning_"+std::to_string(i);
        if(core.terminalLatched() || core.controlledFallbackLatched())return "runtime_safety_termination";
        if(d->qpos[2]<0.25)return "fall_base_height_below_0.25m";
        // Attitude check evaluated from the current freejoint, including pitch
        // beyond the Euler +/-90 degree branch, using upright-axis tilt.
        const auto* q=d->qpos+3;
        if(1-2*(q[1]*q[1]+q[2]*q[2])<std::cos(75*M_PI/180))return "fall_upright_tilt_over_75deg";
        int bc=mj_name2id(m,mjOBJ_GEOM,"base_collision");
        for(int i=0;i<d->ncon;++i)if(d->contact[i].geom1==bc || d->contact[i].geom2==bc)return "fall_base_collision_contact";
        return "";
    }
};

int main(int argc,char**argv) {
    if(argc!=5){std::cerr<<"usage: collect XML CASE_ID OUTPUT_DIR A|B\n";return 2;}
    std::string xml=argv[1],cid=argv[2],out=argv[3];
    const std::string condition=argv[4];
    if(condition!="A" && condition!="B") {std::cerr<<"condition must be A or B\n";return 2;}
    experiment::isaac=condition=="B";
    J pending_row;
    gzFile log=nullptr; J result={{"case_id",cid},{"status","failed"},{"control_records",0}};
    try {
        if(cid!="stand-a01" && cid!="walk-stop-a01")throw std::runtime_error("unknown case");
        if(std::filesystem::exists(out+"/telemetry.jsonl.gz"))throw std::runtime_error("refusing to overwrite telemetry");
        Diagnostic h(xml);h.reset(); write_json(out+"/effective_config.json",h.config());
        log=gzopen((out+"/telemetry.jsonl.gz").c_str(),"wb1");if(!log)throw std::runtime_error("gzopen failed");
        auto emit=[&](const J& j){std::string s=j.dump()+"\n";if(gzwrite(log,s.data(),s.size())!=int(s.size()))throw std::runtime_error("gzwrite failed");};
        const double dt=h.m->opt.timestep;
        // Scheduling periods are read from YAML as doubles; the runtime retains
        // its own float32 periods for gait/state computations.
        double pd=h.params.config_node["dt"].as<double>();int dec=h.params.Get<int>("decimation");double period=pd*dec;
        double duration=cid=="stand-a01"?30:40;
        h.activate();
        int low_tick=0,physical=0,step=0; std::string why;
        while(h.d->time<duration-1e-10) {
            double boundary=step*period;
            pending_row=J{{"case_id",cid},{"control_step",step},{"simulation_time",h.d->time},
                {"nominal_policy_time",boundary},{"reset",step==0},{"reset_id",0},{"terminated",false},
                {"pre",h.state()},{"low_level_updates",J::array()},{"physics_substeps",J::array()},
                {"condition",condition},{"model_input",nullptr},{"model_raw_action",nullptr},{"new_policy_targets",nullptr},{"post",nullptr}};
            J& row=pending_row;
            h.capture->record=&row;
            float planned=cid=="walk-stop-a01" && boundary>=10-1e-10 && boundary<20-1e-10?0.4f:0.0f;
            row["planned_command"]={planned,0.0f,0.0f};
            auto low=[&](){
                h.core.runControlCycle({[&](){h.control.x=planned;h.control.y=0;h.control.yaw=0;}, {}, [](){return true;}, [](){return true;}, {}, {}});
                J u=h.command();u["nominal_time"]=low_tick*pd;u["actual_sim_time"]=h.d->time;u["low_level_tick"]=low_tick++;
                auto a=h.LoadLWPolicyOutput();u["policy_output_frame_applied"]=a?J(a->frame):J(nullptr);
                row["low_level_updates"].push_back(u);
            };
            low();
            auto& s=h.robot_state;
            row["policy_state_float32"]={{"q",s.motor_state.q},{"dq",s.motor_state.dq},{"quaternion_wxyz_body_to_world",s.imu.quaternion},
                {"angular_velocity_body",s.imu.gyroscope},{"sensor_state_evaluation_time",std::max(0.0,h.d->time-dt)}};
            int calls=h.capture->calls;
            h.core.runInferenceCycle(false);
            if(h.capture->calls!=calls+1)throw std::runtime_error("policy inference missing");
            row["model_input"]={{"shape",{1,410}},{"dtype","float32"},{"values",h.capture->input}};
            row["model_raw_action"]=h.capture->raw;
            row["command_in_observation"]=arr(h.capture->input.data()+369+6,3);
            row["projected_gravity_in_observation"]=arr(h.capture->input.data()+369+3,3);
            row["previous_action_in_observation"]=arr(h.capture->input.data()+369+29,10);
            auto p=h.LoadLWPolicyOutput();if(!p)throw std::runtime_error("output not published");
            row["new_policy_targets"]={{"q",p->dof_pos},{"dq",p->dof_vel},{"pd_torque_at_inference",p->dof_tau},{"frame",p->frame},{"source_input_sequence",p->source_input_sequence}};
            const double end=std::min(duration,(step+1)*period);
            while(h.d->time<end-1e-10) {
                if(low_tick*pd<=h.d->time+1e-10)low();
                double before=h.d->time;
                mj_step(h.m,h.d);++physical;
                J sub={{"physics_step",physical},{"from_sim_time",before},{"to_sim_time",h.d->time},
                    {"ctrl",arr(h.d->ctrl,h.m->nu)},{"actuator_force",arr(h.d->actuator_force,h.m->nu)},
                    {"qfrc_actuator",arr(h.d->qfrc_actuator,h.m->nv)},
                    {"post_qpos",arr(h.d->qpos,h.m->nq)},{"post_qvel",arr(h.d->qvel,h.m->nv)}};
                row["physics_substeps"].push_back(sub);
                why=h.stop_reason(); if(h.d->time<=before)why="unexpected_mujoco_time_reset";
                if(!why.empty())break;
            }
            row["post"]=h.state();row["terminated"]=!why.empty() || h.d->time>=duration-1e-10;
            row["termination_reason"]=why.empty()?(row["terminated"].get<bool>()?J("time_limit"):J(nullptr)):J(why);
            row["safety_events"]=h.safety_events;emit(row);pending_row=J();h.capture->record=nullptr;++step;
            if(step%250==0)std::cout<<"case="<<cid<<" step="<<step<<" time="<<h.d->time<<std::endl;
            if(!why.empty())break;
        }
        result={{"condition",condition},{"case_id",cid},{"status",why.empty()?"completed":"terminated_early"},{"termination_reason",why.empty()?"time_limit":why},
            {"control_records",step},{"low_level_ticks",low_tick},{"physics_steps",physical},{"final_sim_time",h.d->time},
            {"requested_sim_duration",duration},{"seed",42},{"seed_method","std::srand(42) before home_leg reset; no explicit policy or physics randomization"},
            {"reset_count",1},{"safety_events",h.safety_events}};
        if(gzclose(log)!=Z_OK)throw std::runtime_error("gzclose failed");log=nullptr;
        write_json(out+"/result.json",result);std::cout<<result.dump(2)<<std::endl;
        return why.empty()?0:3;
    }catch(const std::exception&e) {
        if(log && !pending_row.is_null()) {
            pending_row["terminated"]=true;pending_row["termination_reason"]=std::string("program_exception: ")+e.what();
            std::string line=pending_row.dump()+"\n";gzwrite(log,line.data(),line.size());
        }
        if(log){std::string s=J({{"event","program_exception"},{"case_id",cid},{"terminated",true},{"reason",e.what()}}).dump()+"\n";gzwrite(log,s.data(),s.size());gzclose(log);}
        result["exception"]=e.what();write_json(out+"/result.json",result);std::cerr<<e.what()<<std::endl;return 1;
    }
}
