#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>
#include <iterator>
#include "SemanticGraph.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "nanoflann.hpp"


using InvDesTree = KDTreeVectorOfVectorsAdaptor< std::vector<std::vector<float>>, float >;


void zfill(std::string &in_str, size_t len)
{
    while (in_str.size() < len)
    {
        in_str = "0" + in_str;
    }
}
bool cmp(std::tuple<float,int>& score1,std::tuple<float,int>& score2){
    return std::get<0>(score1)>std::get<0>(score2);
}

int main(int argc, char **argv)
{
    std::string conf_file = "../config/config_ford_campus.yaml";
    if (argc > 1)
    {
        conf_file = argv[1];
    }
    auto data_cfg = YAML::LoadFile(conf_file);
    SGLC:: SemanticGraph SemGraph(conf_file);
    // SGLC::SemanticGraph_MatchNode SemGraph(conf_file);
    auto file_name_length=data_cfg["file_name_length"].as<int>();
    auto cloud_path = data_cfg["eval_overlap"]["cloud_path"].as<std::string>();
    auto label_path = data_cfg["eval_overlap"]["label_path"].as<std::string>();
    auto out_file = data_cfg["eval_overlap"]["out_pair_file"].as<std::string>();
    auto out_file_verifile = data_cfg["eval_overlap"]["out_pair_veri_file"].as<std::string>();

    std::ofstream f_out(out_file);
    std::ofstream f_out_veri(out_file_verifile);

    std::vector<std::vector<float>> global_des_vec;

    int dim = 231; // descriptors dimension 231 : 180(edge:edge_th/edge_sub_interval)*label_category) + 3 (node_category) + 48 (dimension of background descriptors)
    int candidate_scan_num = 50;  // the number of candidate scan, 50 for evaluating recall@1%; for SLAM in actual applications, candidate_scan_num=5~7 is enough.
    int frame_idx = 0;
    while (1)
    {
        std::string sequ = std::to_string(frame_idx);
        zfill(sequ, file_name_length);
        std::string cloud_file, sem_file;
        cloud_file = cloud_path + sequ;
        cloud_file = cloud_file + ".bin";
        sem_file = label_path + sequ;
        sem_file = sem_file + ".label";
        auto start_time = std::chrono::steady_clock::now();
        auto frame_global_desp = SemGraph.gen_scan_descriptors(cloud_file,sem_file);
        global_des_vec.emplace_back(frame_global_desp);
        if(frame_idx>100){
            int search_results_num = frame_idx - 100;
            if(search_results_num>candidate_scan_num) search_results_num = candidate_scan_num;

            std::vector<std::vector<float>> global_des_search;
            std::vector<std::pair<int,double>> candidate_idx_dis;

            global_des_search.assign( global_des_vec.begin(), global_des_vec.begin() + frame_idx - 100 );

            // build kd tree
            std::unique_ptr<InvDesTree> global_des_tree;
            global_des_tree = std::make_unique<InvDesTree>(dim /* dim */, global_des_search, 10 /* max leaf */ );

            std::vector<size_t> candidate_indexes( search_results_num );
            std::vector<float> out_dists_sqr( search_results_num );
            nanoflann::KNNResultSet<float> knnsearch_result( search_results_num );
            knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );

            // query(current) search
            global_des_tree->index->findNeighbors( knnsearch_result, &global_des_vec[frame_idx][0] /* query */, nanoflann::SearchParameters(10) ); 

            for ( int m = 0; m < search_results_num; m++ )
            {
                f_out<<frame_idx<<" "<<candidate_indexes[m]<<" "<<out_dists_sqr[m]<<std::endl;
                candidate_idx_dis.emplace_back(std::make_pair(candidate_indexes[m],out_dists_sqr[m]));
            }
            if(search_results_num<candidate_scan_num){
                for (int m = 0 ; m < candidate_scan_num-search_results_num; m++){
                    f_out<<frame_idx<<" "<<-1<<" "<<2<<std::endl;  // max distance: 2 for two normalized vector
                    candidate_idx_dis.emplace_back(std::make_pair(-1,2)); // max distance: 2 for two normalized vector
                }
            }

            // geometry verification
            for(int m=0;m<candidate_idx_dis.size();++m){
                std::string sequ2 = std::to_string(candidate_idx_dis[m].first);
                zfill(sequ2, file_name_length);
                std::string cloud_file2, sem_file2;
                cloud_file2 = cloud_path + sequ2;
                cloud_file2 = cloud_file2 + ".bin";
                sem_file2 = label_path + sequ2;
                sem_file2 = sem_file2 + ".label";

                double score = 1;
                if(candidate_idx_dis[m].second<2) score=SemGraph.loop_pairs_similarity(cloud_file,cloud_file2,sem_file,sem_file2);
                if(score==0){
                    candidate_idx_dis[m].second = 2; 
                }
                f_out_veri << frame_idx<<" "<<  candidate_idx_dis[m].first << " " << candidate_idx_dis[m].second  << std::endl;
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        std::cout<<"frame_idx:"<<frame_idx<<",time cost:"<<duration<<"ms"<<std::endl;
        frame_idx++;
    }
}