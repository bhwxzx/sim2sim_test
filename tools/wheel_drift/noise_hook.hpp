#ifndef LW_DIAGNOSTIC_NOISE_HPP
#define LW_DIAGNOSTIC_NOISE_HPP
// Linked only by the independent diagnostic executable. Production sources stay unchanged.
struct DiagnosticNoise {
    bool enabled=false; int seed=42; std::mt19937 rng;
    nlohmann::json last;
    void reset(int s,bool e){seed=s;enabled=e;rng.seed(static_cast<unsigned>(s));last=nullptr;}
    float draw(float amplitude){float u=static_cast<float>(rng()>>8)*0x1p-24f;return (2.0f*u-1.0f)*amplitude;}
    nlohmann::json contract()const{return {{"enabled",enabled},{"rng","mt19937(seed), top 24 bits / 2^24, independent from Torch RNG"},{"raw_uniform_half_ranges",{0.2,0.05,0.01,0.5}},{"fields",{"angular_velocity","projected_gravity","nonwheel_joint_position","all_joint_velocity"}},{"order","raw field -> selective additive noise -> term clip [-100,100] -> scale -> original history"},{"wheel_position_noise",false},{"commands_action_noise",false},{"initial_history","same first processed frame replicated by original resetAll"}};}
} diagnostic_noise;
void LWDiagnosticNoise(const LWPolicyRuntimeConfiguration& cfg,const Observations<float>& obs,std::vector<float>& frame){
    if(frame.size()!=39)throw std::runtime_error("diagnostic expects a 39-column DWAQ frame");
    const auto clean=frame;
    std::vector<float> raw(39),scale(39,1),noise(39,0),noiseless_training(39);
    raw=clean;
    for(int i=0;i<3;++i){raw[i]=obs.ang_vel[i];scale[i]=cfg.ang_vel_scale;}
    for(int i=0;i<10;++i){raw[9+i]=cfg.wheel_mask[i]?0.0f:obs.dof_pos[i]-cfg.default_dof_pos[i];scale[9+i]=cfg.dof_pos_scale;raw[19+i]=obs.dof_vel[i];scale[19+i]=cfg.dof_vel_scale;}
    for(int i=0;i<39;++i)noiseless_training[i]=std::clamp(raw[i],-100.0f,100.0f)*scale[i];
    // A/B must not acquire an additional clipping-order difference. Stop and retain evidence if it becomes material.
    if(clean!=noiseless_training)throw std::runtime_error("training/deployment noiseless clipping differs at current state");
    if(diagnostic_noise.enabled){
        for(int i=0;i<3;++i)noise[i]=diagnostic_noise.draw(.2f);
        for(int i=3;i<6;++i)noise[i]=diagnostic_noise.draw(.05f);
        for(int i=0;i<10;++i){const float n=diagnostic_noise.draw(.01f);if(!cfg.wheel_mask[i])noise[9+i]=n;}
        for(int i=19;i<29;++i)noise[i]=diagnostic_noise.draw(.5f);
        for(int i=0;i<39;++i)frame[i]=std::clamp(raw[i]+noise[i],-100.0f,100.0f)*scale[i];
    }
    diagnostic_noise.last={{"clean_deployment_frame",clean},{"raw_before_noise",raw},{"raw_additive_noise",noise},{"scale",scale},{"processed_frame",frame},{"noiseless_training_path_matches_deployment",true}};
}
#endif
