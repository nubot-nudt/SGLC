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
    std::string conf_file="../config/config_kitti360_graph.yaml";
    if(argc>1){
        conf_file=argv[1];
    }
    auto data_cfg = YAML::LoadFile(conf_file);
    auto cloud_path= data_cfg["eval_poses"]["cloud_path"].as<std::string>();
    auto label_path= data_cfg["eval_poses"]["label_path"].as<std::string>();
    auto loop_poses_file = data_cfg["eval_poses"]["loop_poses_file"].as<std::string>();
    auto out_file =  data_cfg["eval_poses"]["out_file"].as<std::string>();
    auto file_name_length=data_cfg["file_name_length"].as<int>();
    SGLC::SemanticGraph SemGraph(conf_file);
    int num = 1;
    float time_count = 0.0;
    std::ifstream f_pairs(loop_poses_file);
    std::ofstream f_out(out_file);
    while (1)
    {
        
        std::string sequ1, sequ2;
        int label;
        f_pairs >> sequ1;
        f_pairs >> sequ2;
        f_out << sequ1 << " " << sequ2<<" ";
        float poses;
        for(int i=0; i<12; i++){
            f_pairs >> poses;
        }
        if (sequ1.empty() || sequ2.empty())
        {
            break;
        }
        std::string label1 = sequ1;
        std::string label2 = sequ2;
        zfill(sequ1,file_name_length);
        zfill(sequ2,file_name_length);
        zfill(label1,6);
        zfill(label2,6);
        std::string cloud_file1, cloud_file2, sem_file1, sem_file2;
        cloud_file1 = cloud_path + sequ1;
        cloud_file1 = cloud_file1 + ".bin";
        cloud_file2 = cloud_path + sequ2;
        cloud_file2 = cloud_file2 + ".bin";
        sem_file1 = label_path + label1;
        sem_file1 = sem_file1 + ".label";
        sem_file2 = label_path + label2;
        sem_file2 = sem_file2 + ".label";


        auto start_time = std::chrono::steady_clock::now();
        auto transform = SemGraph.loop_poses_estimation(cloud_file1,cloud_file2,sem_file1,sem_file2);
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        time_count = time_count + duration;
        for(int i=0;i<3;i++){
            for(int j=0;j<4;j++){
                if(i==2 && j==3) f_out<< transform(i,j)<<std::endl;
                else f_out<< transform(i,j)<<" ";
                
            }
        }
        num++;
        std::cout<<num<<": "<<duration<<"ms"<<std::endl;
    }
    float average_time = time_count/num;
    std::cout<<"average_time:"<<average_time<<std::endl;
    std::cout<<"===================================="<<std::endl;
    std::cout<<"frame_count:"<<SemGraph.frame_count<<std::endl;
    std::cout<<"icp average_time:"<<SemGraph.total_time/SemGraph.frame_count<<std::endl;
    return 0;
}