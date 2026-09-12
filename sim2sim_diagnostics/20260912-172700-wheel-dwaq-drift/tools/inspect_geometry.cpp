#include <mujoco/mujoco.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <limits>
#include <stdexcept>
using J=nlohmann::json;
int main(int argc,char** argv){if(argc!=3)return 2;char err[4096]{};mjModel* m=mj_loadXML(argv[1],nullptr,err,sizeof(err));if(!m)throw std::runtime_error(err);mjData*d=mj_makeData(m);int key=mj_name2id(m,mjOBJ_KEY,"home_wheel");mj_resetDataKeyframe(m,d,key);mj_forward(m,d);J j={{"mujoco_version",mj_versionString()},{"physics_steps",0},{"method","minimum world Z of compiled collision mesh vertices at unchanged home_wheel; geometric clearance, excludes contact margin"}};for(int g=0;g<m->ngeom;++g){if(m->geom_type[g]!=mjGEOM_MESH || m->geom_contype[g]==0)continue;int id=m->geom_dataid[g];double low=std::numeric_limits<double>::infinity();for(int v=m->mesh_vertadr[id];v<m->mesh_vertadr[id]+m->mesh_vertnum[id];++v){double z=d->geom_xpos[3*g+2];for(int k=0;k<3;++k)z+=d->geom_xmat[9*g+6+k]*m->mesh_vert[3*v+k];low=std::min(low,z);}const char* name=mj_id2name(m,mjOBJ_GEOM,g);j["collision_mesh_min_z_m"][name?name:"unnamed"]=low;}std::ofstream f(argv[2]);f<<j.dump(2)<<'\n';mj_deleteData(d);mj_deleteModel(m);}
