#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <pcl/io/pcd_io.h>
#include "SemanticGraph.hpp"


std::ostream& blue(std::ostream& os) {
    return os << "\033[1;36m";
}

std::ostream& reset(std::ostream& os) {
    return os << "\033[0m";
}

int main(){
    std::string conf_file="../config/config_kitti_graph.yaml";
    auto data_cfg = YAML::LoadFile(conf_file);
    auto cloud_file1=data_cfg["eval_pair"]["cloud_file1"].as<std::string>();
    auto cloud_file2=data_cfg["eval_pair"]["cloud_file2"].as<std::string>();
    auto label_file1=data_cfg["eval_pair"]["label_file1"].as<std::string>();
    auto label_file2=data_cfg["eval_pair"]["label_file2"].as<std::string>();
    SGLC::SemanticGraph SemGraph(conf_file);
    
    std::cout<<"cloud_file1:"<<blue<<cloud_file1<<reset<<std::endl;
    std::cout<<"cloud_file2:"<<blue<<cloud_file2<<reset<<std::endl;

    auto start_time = std::chrono::steady_clock::now();
    auto score=SemGraph.loop_pairs_similarity(cloud_file1,cloud_file2,label_file1,label_file2);
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    std::cout<<" similarity:"<<blue<<score<<reset<<std::endl;
    std::cout<<"       time:"<<blue<<duration<<" ms"<<reset<<std::endl;
    return 0;
}