#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include "SemanticGraph.hpp"
void zfill(std::string& in_str,int len){
        while (in_str.size() < len)
        {
            in_str = "0" + in_str;
        }
}
int main(int argc,char** argv){

    std::string conf_file="../config/config_kitti_graph.yaml";
    std::string out_file="../out/demo_pair_pose.txt";
    std::ofstream f_out(out_file,std::ofstream::trunc);
    if(argc>1){
        conf_file=argv[1];
    }
    auto data_cfg = YAML::LoadFile(conf_file);
    auto cloud_file1=data_cfg["eval_pair"]["cloud_file1"].as<std::string>();
    auto cloud_file2=data_cfg["eval_pair"]["cloud_file2"].as<std::string>();
    auto label_file1=data_cfg["eval_pair"]["label_file1"].as<std::string>();
    auto label_file2=data_cfg["eval_pair"]["label_file2"].as<std::string>();
    SGLC::SemanticGraph SemGraph(conf_file);

    auto start_time = std::chrono::steady_clock::now();
    auto transform = SemGraph.loop_poses_estimation(cloud_file1,cloud_file2,label_file1,label_file2);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout<<"transform:"<<transform<<std::endl;
    for(int i=0;i<3;i++){
            for(int j=0;j<4;j++){
                if(i==2 && j==3) f_out<< transform(i,j)<<std::endl;
                else f_out<< transform(i,j)<<" ";
                
            }
    }
    std::cout<<"===================================="<<std::endl;
    std::cout<<"total time (including load pc, gen des, registration):"<<duration<<"ms"<<std::endl;
    std::cout<<"frame_count:"<<SemGraph.frame_count<<std::endl;
    std::cout<<"icp average_time:"<<SemGraph.total_time/SemGraph.frame_count<<std::endl;
    return 0;

}