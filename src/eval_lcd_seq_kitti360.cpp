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
    auto cloud_path=data_cfg["eval_seq"]["cloud_path"].as<std::string>();
    auto label_path=data_cfg["eval_seq"]["label_path"].as<std::string>();
    auto pairs_file=data_cfg["eval_seq"]["pairs_file"].as<std::string>();
    auto out_file=data_cfg["eval_seq"]["out_file"].as<std::string>();
    auto file_name_length=data_cfg["file_name_length"].as<int>();

    SGLC::SemanticGraph SemGraph(conf_file);
    std::ifstream f_pairs(pairs_file);
    std::ofstream f_out(out_file);
    int num = 1;
    float time_count = 0.0;
    while (1)
    {
        std::string sequ1, sequ2;
        int label;
        f_pairs >> sequ1;
        f_pairs >> sequ2;
        f_pairs >> label;
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
        auto score_graph = SemGraph.loop_pairs_similarity(cloud_file1,cloud_file2,sem_file1,sem_file2); 
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        time_count = time_count + duration;
        f_out << score_graph << " " << label << std::endl;
        std::cout<<num<<" "<<score_graph<<" "<<label<<" "<<duration<<"ms"<<std::endl;
        num++;
    }
    float average_time = time_count/num;
    std::cout<<"average time:"<<SemGraph.total_time/SemGraph.frame_count<<std::endl;
    std::cout<<"frame_count:"<<SemGraph.frame_count<<std::endl;
    return 0;
}