// Offline observation-contract check. No mj_step, no closed-loop rollout.
#define main collector_entry_unused
#include "collect.cpp"
#undef main

int main(int argc,char** argv) {
    if(argc!=3) return 2;
    try {
        Diagnostic h(argv[1]);h.reset();
        const auto& cfg=h.GetLWPolicyDefinition(h.policy_key)->runtime;
        J result={{"type","offline synthetic frame check; no physics stepping or policy activation"},
                  {"physics_steps",0},{"joint_mapping",h.joint_mapping_evidence},{"frames",J::array()}};
        ObservationBuffer ha(1,{3,3,3,10,10,10,2},10,"time"),hb(1,{3,3,3,10,10,10,2},10,"time");
        std::vector<float> last_a,last_b;
        for(int k=0;k<16;++k) {
            Observations<float> obs;
            obs.ang_vel={.1f,.2f,.3f};obs.gravity_vec={0,0,-1};obs.commands={.4f,0,0};obs.base_quat={1,0,0,0};
            obs.dof_pos=cfg.default_dof_pos;obs.dof_vel.assign(10,-7.0f);obs.actions.assign(10,.25f);obs.gait_phase={0,1};
            for(int i=0;i<10;++i)obs.dof_pos[i]+=(k+1)*.015f*(i+1);
            // Distinct angle and speed; right wheel cannot pass via dq substitution.
            obs.dof_pos[experiment::right_wheel]=.125f*(k+1);
            std::vector<float> a(41),b(41),ia(410),ib(410);
            experiment::isaac=false;h.ComputeLWObservationInto(cfg,obs,nullptr,a);
            experiment::isaac=true;h.ComputeLWObservationInto(cfg,obs,nullptr,b);
            if(a[16]!=(obs.dof_pos[7]-cfg.default_dof_pos[7])*cfg.dof_pos_scale || a[17]!=0 || a[18]!=0
               || b[16]!=0 || b[17]!=(obs.dof_pos[8]-cfg.default_dof_pos[8])*cfg.dof_pos_scale || b[18]!=0)
                throw std::runtime_error("A/B joint position semantics failed");
            for(int i=0;i<41;++i)if(i!=16 && i!=17 && a[i]!=b[i])throw std::runtime_error("other observation column changed");
            const bool reset=k==0 || k==9;
            if(reset) {ha.resetAll(a);hb.resetAll(b);}
            ha.insert(a);hb.insert(b);ha.getObsInto(cfg.observations_history,ia);hb.getObsInto(cfg.observations_history,ib);
            if(reset) {
                for(int j=0;j<10;++j)if(!std::equal(ia.begin()+j*41,ia.begin()+(j+1)*41,a.begin())
                    || !std::equal(ib.begin()+j*41,ib.begin()+(j+1)*41,b.begin()))throw std::runtime_error("history reset fill failed");
            } else if(!std::equal(ia.begin(),ia.begin()+369,last_a.begin()+41)
                       || !std::equal(ib.begin(),ib.begin()+369,last_b.begin()+41))throw std::runtime_error("history shift failed");
            last_a=ia;last_b=ib;
            result["frames"].push_back({{"probe_step",k},{"reset",reset},{"q",obs.dof_pos},{"dq",obs.dof_vel},
                {"default_q",cfg.default_dof_pos},{"A_input",ia},{"B_input",ib}});
        }
        result["all_checks_pass"]=true;
        write_json(argv[2],result);std::cout<<"offline frame/history checks passed\n";return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
