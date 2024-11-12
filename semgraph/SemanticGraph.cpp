// This file is covered by the LICENSE file in the root of this project.
// developed by Neng Wang, <neng.wang@hotmail.com>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>


#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <random>
#include <cmath>
#include "SemanticGraph.hpp"

namespace SGLC{
    inline double square(double x) { return x * x; }
    struct ResultTuple {
        ResultTuple() {
            JTJ.setZero();
            JTr.setZero();
        }

        ResultTuple operator+(const ResultTuple &other) {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            return *this;
        }

        Eigen::Matrix<double, 6, 6> JTJ;
        Eigen::Matrix<double, 6, 1> JTr;
    };

    SemanticGraph::SemanticGraph(std::string conf_file)
    {
        auto data_cfg = YAML::LoadFile(conf_file);
        show = data_cfg["show"].as<bool>();
        cluster_view = data_cfg["cluster_view"].as<bool>();
        remap = data_cfg["remap"].as<bool>();

        // cluster 
        deltaA = data_cfg["deltaA"].as<double>(); 
        deltaR = data_cfg["deltaR"].as<double>(); 
        deltaP = data_cfg["deltaP"].as<double>(); 


        edge_th = data_cfg["edge_th"].as<double>();
        sub_interval = (edge_th / data_cfg["edge_sub_interval"].as<double>());
        graph_node_dimension = data_cfg["graph_node_dimension"].as<int>();
        box_diff_th = data_cfg["box_diff_th"].as<double>();
        edge_local_triangle = data_cfg["edge_local_triangle"].as<double>();
        true_node_corres_th = data_cfg["true_node_corres_th"].as<int>();

        graph_sim_th = data_cfg["graph_sim_th"].as<double>();
        back_sim_th = data_cfg["back_sim_th"].as<double>();

        back_sim_th_without_node = data_cfg["back_sim_th_without_node"].as<double>();
        inlier_th = data_cfg["inlier_th"].as<double>();

        if (show)
        {
            viewer.reset(new pcl::visualization::CloudViewer("viewer"));
        }
        auto color_map = data_cfg["color_map"];
        learning_map = data_cfg["learning_map"];
        label_map.resize(260);
        for (auto it = learning_map.begin(); it != learning_map.end(); ++it)
        {
            label_map[it->first.as<int>()] = it->second.as<int>();
        }
        YAML::const_iterator it;
        for (it = color_map.begin(); it != color_map.end(); ++it)
        {
            // Get label and key
            int key = it->first.as<int>(); // <- key
            Color color = std::make_tuple(
                static_cast<u_char>(color_map[key][0].as<unsigned int>()),
                static_cast<u_char>(color_map[key][1].as<unsigned int>()),
                static_cast<u_char>(color_map[key][2].as<unsigned int>()));
            _color_map[key] = color;
        }
        auto learning_class = data_cfg["learning_map_inv"];
        for (it = learning_class.begin(); it != learning_class.end(); ++it)
        {
            int key = it->first.as<int>(); // <- key
            _argmax_to_rgb[key] = _color_map[learning_class[key].as<unsigned int>()];
        }
    }

    SemanticGraph::~SemanticGraph()
    {

    }

    SGLC::Graph SemanticGraph::build_graph(std::vector<SGLC::Bbox> cluster_boxes, double edge_dis_th,double subinterval){
        SGLC::Graph frame_graph;
        float sub_interval_value = edge_dis_th/subinterval;
        int N = cluster_boxes.size();
        Eigen::MatrixXf AdjacencyMatrix = Eigen::MatrixXf::Zero(N,N);
        Eigen::MatrixXf EdgeMatrix = Eigen::MatrixXf::Zero(N,N);
        Eigen::MatrixXf NodeEmbeddings = Eigen::MatrixXf::Zero(N,subinterval*3+graph_node_dimension);
        Eigen::MatrixXf NodeEmbeddings_Local = Eigen::MatrixXf::Zero(N,subinterval*3);

        for(size_t i = 0; i < N; i++){
            frame_graph.node_labels.emplace_back(cluster_boxes[i].label);
            if(cluster_boxes[i].label==1) frame_graph.node_stable.emplace_back(0); 
            else frame_graph.node_stable.emplace_back(1);
            frame_graph.node_centers.emplace_back(cluster_boxes[i].center);
            frame_graph.node_dimensions.emplace_back(cluster_boxes[i].dimension);
            std::vector<std::pair<int,double>> vertex_edges;
            for(size_t j = 0; j< N; j++){
                double edge = (cluster_boxes[i].center - cluster_boxes[j].center).norm();
                if(edge<edge_dis_th){
                    vertex_edges.emplace_back(std::make_pair(cluster_boxes[i].label,edge));
                    AdjacencyMatrix(i,j) = 1;
                    EdgeMatrix(i,j) = edge;
                    if(j>=i+1){                                          // only count once
                        frame_graph.edges.emplace_back(std::make_pair(i,j));
                        frame_graph.edge_value.emplace_back(edge);
                        frame_graph.edge_weights.emplace_back((edge_dis_th-edge)/edge_dis_th); //[0,edge_dis_th]->[1,0]
                    }
                }
            }
            // build vertes desc
            for(size_t m=0;m<vertex_edges.size();m++){
                if(vertex_edges[m].first == 1){ // x - car
                    frame_graph.car_num = frame_graph.car_num+1;
                    NodeEmbeddings_Local(i,int(vertex_edges[m].second/sub_interval_value))++;
                }
                else if(vertex_edges[m].first == 2){ // x - truck
                    frame_graph.trunk_num = frame_graph.trunk_num+1;
                    NodeEmbeddings_Local(i,subinterval+int(vertex_edges[m].second/sub_interval_value))++;
                }
                else if(vertex_edges[m].first == 3){ // x - pole
                    frame_graph.pole_like_num = frame_graph.pole_like_num+1;
                    NodeEmbeddings_Local(i,subinterval*2+int(vertex_edges[m].second/sub_interval_value))++;
                }
            }
        }

        if(frame_graph.node_labels.size()==0) return  frame_graph;
        // only 0.0xms -> 0.1ms 
        Eigen::MatrixXf NodeEmbeddings_Global= matrix_decomposing(AdjacencyMatrix,graph_node_dimension);
        NodeEmbeddings_Local = NodeEmbeddings_Local.array().colwise()/NodeEmbeddings_Local.rowwise().norm().array();
        NodeEmbeddings_Global = NodeEmbeddings_Global.array().colwise()/NodeEmbeddings_Global.rowwise().sum().array();
        NodeEmbeddings.leftCols(subinterval*3) = NodeEmbeddings_Local;
        NodeEmbeddings.rightCols(graph_node_dimension) = NodeEmbeddings_Global;

        for(size_t i=0;i<N;i++){
            Eigen::MatrixXf evec_sort_row = NodeEmbeddings.row(i);
            std::vector<float> node_desf(evec_sort_row.data(),evec_sort_row.data()+evec_sort_row.size());
            frame_graph.node_desc.emplace_back(node_desf);
        }

        frame_graph.edge_matrix = EdgeMatrix; // for subsequent correspondences pruning

        return frame_graph;
    }

    /*
        Decomposing Matrix
        input: matrix 
        retun: eigen vector
    */
    Eigen::MatrixXf SemanticGraph::matrix_decomposing(Eigen::MatrixXf MatrixInput,int Dimension){ 

        // decomposing adjacency matrix
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> es;
        es.compute(MatrixInput);
        Eigen::MatrixXcf evecs = es.eigenvectors();           
        Eigen::MatrixXf evecs_abs = evecs.real().cwiseAbs();   

        Eigen::MatrixXcf evals = es.eigenvalues();             
        Eigen::MatrixXf abs_evals = evals.real().cwiseAbs();

        // sort to get max->min eigen value
        std::vector<float> vec_evals(&abs_evals(0, 0),abs_evals.data()+abs_evals.size()); // Eigen::MatrixXf --> std::vector
        std::vector<int> indices(vec_evals.size());
        std::iota(indices.begin(), indices.end(), 0);  // sort: get d eigen vector corresponding to the d largest eigen value
        std::sort(indices.begin(), indices.end(), [&vec_evals](int i, int j) { return vec_evals[i] > vec_evals[j]; });

        // get node vector
        Eigen::MatrixXf evecs_sort = Eigen::MatrixXf::Zero(vec_evals.size(),Dimension);
        int iter = std::min(Dimension,static_cast<int>(indices.size()));
        for(size_t i=0;i<iter;i++){
            evecs_sort.col(i) = evecs_abs.col(indices[i]);
        }

        return evecs_sort;
    }

    
    std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> SemanticGraph::load_cloud(std::string file_cloud, std::string file_label){
        std::ifstream in_label(file_label, std::ios::binary);
        if (!in_label.is_open()) {
            std::cerr << "No file:" << file_label << std::endl;
            exit(-1);
        }
        in_label.seekg(0, std::ios::end);
        uint32_t num_points = in_label.tellg() / sizeof(uint32_t);
        in_label.seekg(0, std::ios::beg);
        std::vector<uint32_t> values_label(num_points);
        in_label.read((char*)&values_label[0], num_points * sizeof(uint32_t));
        std::ifstream in_cloud(file_cloud, std::ios::binary);
        std::vector<float> values_cloud(4 * num_points);
        in_cloud.read((char*)&values_cloud[0], 4 * num_points * sizeof(float));
  

        std::vector<Eigen::Vector3d> pc_out(num_points);
        std::vector<int> label_out(num_points);

        for (uint32_t i = 0; i < num_points; ++i) {
            uint32_t sem_label;
            if (remap) {
                sem_label = label_map[(int)(values_label[i] & 0x0000ffff)];
            } 
            else {
                sem_label = values_label[i];
            }
            if (sem_label == 0) {
                pc_out[i] = Eigen::Vector3d(0,0,0);
                label_out[i] = 0;
                continue;
            }
            pc_out[i] = Eigen::Vector3d(values_cloud[4 * i],values_cloud[4 * i + 1],values_cloud[4 * i + 2]);
            label_out[i] = (int) sem_label;
        }
        in_label.close();
        in_cloud.close();
        return std::make_pair(pc_out,label_out);
    }
    

    


      // perform loop closure detection
    double SemanticGraph::loop_pairs_similarity(std::string cloud_file1, std::string cloud_file2, std::string label_file1, std::string label_file2){
        
        double simility_score = 0;
        auto cloud1 = load_cloud(cloud_file1,label_file1);
        auto cloud2 = load_cloud(cloud_file2,label_file2);

        auto start_time = std::chrono::steady_clock::now();

        // clustering   
        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points1(new pcl::PointCloud<pcl::PointXYZL>());  
        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points2(new pcl::PointCloud<pcl::PointXYZL>());  
        auto cluster_box1 = SGLC::cluster(cloud1.first,cloud1.second,*background_points1,deltaA,deltaR,deltaP,cluster_view);
        auto cluster_box2 = SGLC::cluster(cloud2.first,cloud2.second,*background_points2,deltaA,deltaR,deltaP,cluster_view);

        // graph
        auto graph1 = build_graph(cluster_box1,edge_th,sub_interval);
        auto graph2 = build_graph(cluster_box2,edge_th,sub_interval);

        // graph des
        auto graph1_des = gen_graph_descriptors(graph1,edge_th,60);
        auto graph2_des = gen_graph_descriptors(graph2,edge_th,60);


        if(graph1.node_desc.size()==0 || graph2.node_desc.size()==0){
                auto desc1 = gen_background_descriptors(background_points1);
                auto desc2 = gen_background_descriptors(background_points2);
                double score_ssc = calculate_sim(desc1, desc2);
                if(score_ssc>back_sim_th_without_node) simility_score = score_ssc;
                else simility_score=0;
                return simility_score;
        }
        
        auto [match_node_inGraph1,match_node_inGraph2] = find_correspondences_withidx(graph1,graph2);
        auto [refine_match_node_inGraph1,refine_match_node_inGraph2] = outlier_pruning(graph1,graph2,match_node_inGraph1,match_node_inGraph2);

        if(refine_match_node_inGraph1.size()>=3){

            int best_inlier = 0;
            auto [trans,score] = ransac_alignment(refine_match_node_inGraph1,refine_match_node_inGraph2,match_node_inGraph1.first,match_node_inGraph2.first,30,best_inlier);

            simility_score = score;

            Eigen::Vector3d translation = trans.matrix().block<3, 1>(0, 3);
            double distance = translation.norm();
        
            pcl::PointCloud<pcl::PointXYZL>::Ptr background_points1_trans(new pcl::PointCloud<pcl::PointXYZL>);
            transformPointCloud(*background_points1, *background_points1_trans, trans.matrix());
            auto desc1 = gen_background_descriptors(background_points1_trans);  
            auto desc2 = gen_background_descriptors(background_points2);
            double score_ssc = calculate_sim(desc1, desc2);

            if(score<graph_sim_th || score_ssc<back_sim_th || distance>10) simility_score=0;   

        }
        else{

            simility_score = 0;
        }
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();   

        total_time = total_time + duration;
        frame_count = frame_count + 1;
        
        return simility_score;
    }

Eigen::Matrix4d SemanticGraph::loop_poses_estimation(std::string cloud_file1, std::string cloud_file2, std::string label_file1, std::string label_file2){

    Eigen::Matrix4d trans_pose = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d trans_icp = Eigen::Matrix4d::Identity();
    double simility_score = 0;
    auto cloud1 = load_cloud(cloud_file1,label_file1);
    auto cloud2 = load_cloud(cloud_file2,label_file2);

    auto start_time = std::chrono::steady_clock::now();

    // clustering   
    pcl::PointCloud<pcl::PointXYZL>::Ptr background_points1(new pcl::PointCloud<pcl::PointXYZL>());  
    pcl::PointCloud<pcl::PointXYZL>::Ptr background_points2(new pcl::PointCloud<pcl::PointXYZL>());  
    std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> clusterPointL_a;
    std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> clusterPointL_b;
    auto cluster_box1 = SGLC::cluster_point(cloud1.first,cloud1.second,*background_points1,clusterPointL_a,deltaA,deltaR,deltaP,cluster_view);
    auto cluster_box2 = SGLC::cluster_point(cloud2.first,cloud2.second,*background_points2,clusterPointL_b,deltaA,deltaR,deltaP,cluster_view);

    // graph
    auto graph1 = build_graph(cluster_box1,edge_th,sub_interval);
    auto graph2 = build_graph(cluster_box2,edge_th,sub_interval);
    
    // graph desp
    auto graph1_des = gen_graph_descriptors(graph1,edge_th,60);
    auto graph2_des = gen_graph_descriptors(graph2,edge_th,60);

    auto start_time_icp = std::chrono::steady_clock::now();

    // insufficient nodes for RANSAC
    if(graph1.node_desc.size()==0 || graph2.node_desc.size()==0){
            auto desc1 = gen_background_descriptors(background_points1);
            auto desc2 = gen_background_descriptors(background_points2);
            double score_ssc = calculate_sim(desc1, desc2);
            if(score_ssc>0.7) simility_score = score_ssc;
            else simility_score=0;
            FastIcp instanceIcp(0.25,0.25,1,5); // 18ms
            std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> back_pt1;
            std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> back_pt2;
            for(int i=0;i<background_points1->size();i++){
                back_pt1.first.emplace_back(Eigen::Vector3d((*background_points1)[i].x,(*background_points1)[i].y,(*background_points1)[i].z));
                back_pt1.second.emplace_back((*background_points1)[i].label);
            }
            for(int i=0;i<background_points2->size();i++){
                back_pt2.first.emplace_back(Eigen::Vector3d((*background_points2)[i].x,(*background_points2)[i].y,(*background_points2)[i].z));
                back_pt2.second.emplace_back((*background_points2)[i].label);
            }
            trans_icp = instanceIcp.get_trans(back_pt1,back_pt2,trans_pose);
            return trans_icp;
    }

    // build vertiex correspondence of graph
    auto [match_node_inGraph1,match_node_inGraph2] = find_correspondences_withidx(graph1,graph2);
    auto [refine_match_node_inGraph1,refine_match_node_inGraph2] = outlier_pruning(graph1,graph2,match_node_inGraph1,match_node_inGraph2);

    if(refine_match_node_inGraph1.size()>=3){ 

        // coarse-fine-refine poses estimation
        // ransac for graph nodes
        int best_inlier = 0;
        auto [trans,score] = ransac_alignment(refine_match_node_inGraph1,refine_match_node_inGraph2,match_node_inGraph1.first,match_node_inGraph2.first,100,best_inlier);
        simility_score = score;
        trans_pose = trans.matrix();
        // point-to-point icp for foreground points
        FastIcp instanceIcp(0.25,0.25,1,5); 
        trans_icp = instanceIcp.get_trans(clusterPointL_a,clusterPointL_b,trans_pose);

        // point-to-plane icp for background points
        auto back_cloud1 = pcl2eigen(background_points1);
        auto back_cloud2 = pcl2eigen(background_points2);
        PlaneIcp backgroundPlaceIcp(0.25,1); 
        auto trans_planeicp = backgroundPlaceIcp.getTransPlaneIcp(back_cloud1,back_cloud2,trans_icp);

        trans_icp = trans_planeicp;
    }
    else{
        simility_score = 0;
    }

    auto end_time_icp = std::chrono::steady_clock::now();
    auto duration_icp = std::chrono::duration<double, std::milli>(end_time_icp - start_time_icp).count();
    total_time = total_time + duration_icp;
    frame_count = frame_count+1;
    return trans_icp;
}


 std::vector<float> SemanticGraph::gen_scan_descriptors(std::string cloud_file,std::string label_file){
        
        auto cloud = load_cloud(cloud_file,label_file);

        auto start_time = std::chrono::steady_clock::now();
        // clustering   
        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points(new pcl::PointCloud<pcl::PointXYZL>());  
        auto cluster_box = SGLC::cluster(cloud.first,cloud.second,*background_points,deltaA,deltaR,deltaP,cluster_view);

        // graph
        auto graph = build_graph(cluster_box,edge_th,sub_interval);

        // graph desp
        auto graph_des = gen_graph_descriptors(graph,edge_th,sub_interval);

        auto back_ground = gen_back_descriptors_global(background_points);

        std::vector<float> global_graph_desp;
        global_graph_desp.insert(global_graph_desp.end(),graph_des.begin(),graph_des.end());
        global_graph_desp.insert(global_graph_desp.end(),back_ground.begin(),back_ground.end());

        float sum_of_squares_graph = std::inner_product(global_graph_desp.begin(), global_graph_desp.end(), global_graph_desp.begin(), 0.0f);
        float l2_norm_graph = std::sqrt(sum_of_squares_graph);
        if(l2_norm_graph==0) l2_norm_graph=0.000001f;
        for (auto it = global_graph_desp.begin(); it != global_graph_desp.end(); ++it) {
            *it /= l2_norm_graph; // 归一化
        }

        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_time = total_time + duration;
        frame_count = frame_count + 1;
        return global_graph_desp;
    }


    std::vector<float> SemanticGraph::gen_graph_descriptors(SGLC::Graph graph,float edge_dis_th,int subinterval){
        float sub_interval_value = edge_dis_th/subinterval;
        std::vector<float> graph_desc(subinterval*6+3,0);

        for(size_t i=0; i<graph.edges.size();i++){
            if(graph.node_labels[graph.edges[i].first]==1 && graph.node_labels[graph.edges[i].second]==1){
                graph_desc[int(graph.edge_value[i]/sub_interval_value)+3]++;
            }
            else if(graph.node_labels[graph.edges[i].first]==2 && graph.node_labels[graph.edges[i].second]==2){
                graph_desc[subinterval*1+int(graph.edge_value[i]/sub_interval_value)+3]++;
            }
            else if(graph.node_labels[graph.edges[i].first]==3 && graph.node_labels[graph.edges[i].second]==3)
            {
                graph_desc[subinterval*2+int(graph.edge_value[i]/sub_interval_value)+3]++;
            }
            else if((graph.node_labels[graph.edges[i].first]==3 && graph.node_labels[graph.edges[i].second]==2)||(graph.node_labels[graph.edges[i].first]==2 && graph.node_labels[graph.edges[i].second]==3)){
                graph_desc[subinterval*3+int(graph.edge_value[i]/sub_interval_value)+3]++;
            }
            else if((graph.node_labels[graph.edges[i].first]==1 && graph.node_labels[graph.edges[i].second]==2)||(graph.node_labels[graph.edges[i].first]==2 && graph.node_labels[graph.edges[i].second]==1)){
                graph_desc[subinterval*4+int(graph.edge_value[i]/sub_interval_value)+3]++;
            }
            else if((graph.node_labels[graph.edges[i].first]==1 && graph.node_labels[graph.edges[i].second]==3)||(graph.node_labels[graph.edges[i].first]==3 && graph.node_labels[graph.edges[i].second]==1)){
                graph_desc[subinterval*5+int(graph.edge_value[i]/sub_interval_value)+3]++;
            }
        }

        float sum_of_squares_edge = std::inner_product(graph_desc.begin(), graph_desc.end(), graph_desc.begin(), 0.0f);
        float l2_norm_edge = std::sqrt(sum_of_squares_edge);
        if(l2_norm_edge==0) l2_norm_edge=0.000001f;
        for (auto it = graph_desc.begin(); it != graph_desc.end(); ++it) {
            *it /= l2_norm_edge; 
        }

        float sum_norm_l2 = std::sqrt(graph.car_num*graph.car_num + graph.trunk_num*graph.trunk_num + graph.pole_like_num*graph.pole_like_num); 
        if(sum_norm_l2==0) sum_norm_l2=0.000001f;
        graph_desc[0] = ((float)graph.car_num)/sum_norm_l2;
        graph_desc[1] = ((float)graph.trunk_num)/sum_norm_l2;
        graph_desc[2] = ((float)graph.pole_like_num)/sum_norm_l2;

        float sum_of_squares_all= std::inner_product(graph_desc.begin(), graph_desc.end(), graph_desc.begin(), 0.0f);
        float l2_norm_all = std::sqrt(sum_of_squares_all);
        if(l2_norm_all==0) l2_norm_all=0.000001f;
        for (auto it = graph_desc.begin(); it != graph_desc.end(); ++it) {
            *it /= l2_norm_all; 
        }

        return graph_desc;
    }



    double SemanticGraph::get_cos_simscore(const std::vector<int> vec1, const std::vector<int> vec2){
        assert(vec1.size()==vec2.size());
        double tmp= 0.0;
        for(size_t i=0;i<vec1.size();i++){
            tmp += vec1[i]*vec2[i];
        }
        double simility = tmp / (getMold(vec1)*getMold(vec2));
        return simility;
    }

    // claculate cosine similiraty
    double SemanticGraph::get_cos_simscore(const std::vector<float> vec1, const std::vector<float> vec2){
        assert(vec1.size()==vec2.size());
        double tmp= 0.0;
        for(size_t i=0;i<vec1.size();i++){
            tmp += vec1[i]*vec2[i];
        }
        double simility = tmp / (getMold(vec1)*getMold(vec2));
        return simility;
    }

    double SemanticGraph::getMold(const std::vector<int> vec){
        double sum = 0.0;
        for(size_t i=0;i<vec.size();i++){
            sum += vec[i]*vec[i];
        }
        if(sum==0) sum = 0.00000000001; 
        return std::sqrt(sum);
    }

    double SemanticGraph::getMold(const std::vector<float> vec){
        double sum = 0.0;
        for(size_t i=0;i<vec.size();i++){
            sum += vec[i]*vec[i];
        }
        if(sum==0) sum = 0.00000000001; 
        return std::sqrt(sum);
    }


     std::tuple<V3d_i,V3d_i> SemanticGraph::find_correspondences_withidx(SGLC::Graph graph1,SGLC::Graph graph2){

        std::vector<std::vector<double>> associationCost_Matrix;
        associationCost_Matrix.resize(graph1.node_desc.size(),std::vector<double>(graph2.node_desc.size(),0));

        for(size_t i=0;i<graph1.node_desc.size();i++){
            for(size_t j=0;j<graph2.node_desc.size();j++){
                if(graph1.node_labels[i] == graph2.node_labels[j]){ // check node's label
                    double node_des_cost = 1 - get_cos_simscore(graph1.node_desc[i],graph2.node_desc[j]); 
                    if(node_des_cost > 0.5 || node_des_cost < 0) node_des_cost = 1e8;
                    if(std::abs(graph1.node_dimensions[i].x()-graph2.node_dimensions[j].x())>box_diff_th ||
                            std::abs(graph1.node_dimensions[i].y()-graph2.node_dimensions[j].y())>box_diff_th ||
                                std::abs(graph1.node_dimensions[i].z()-graph2.node_dimensions[j].z())>box_diff_th) node_des_cost = 1e8;
                    associationCost_Matrix[i][j] = node_des_cost;

                }
                else{
                    associationCost_Matrix[i][j] = 1e8;
                }
            }
        }
        // use Hungarian algorithm solve optimal correnpondences
        std::vector<int> assignment;
        HungarianAlgorithm HungAlgo;
        assignment.clear();
        HungAlgo.Solve(associationCost_Matrix,assignment);

       // match results
        std::vector<Eigen::Vector3d> query_nodes_center;
        std::vector<Eigen::Vector3d> match_nodes_center;
        std::vector<int> query_nodes_idx;
        std::vector<int> match_nodes_idx;
        for(size_t i=0;i<graph1.node_desc.size();i++){
            if(assignment[i]!=-1 && associationCost_Matrix[i][assignment[i]]<1e8){
                query_nodes_center.emplace_back(graph1.node_centers[i]);
                query_nodes_idx.emplace_back(i);
                match_nodes_center.emplace_back(graph2.node_centers[assignment[i]]);
                match_nodes_idx.emplace_back(assignment[i]);
            }
        }
        return {std::make_pair(query_nodes_center,query_nodes_idx),std::make_pair(match_nodes_center,match_nodes_idx)};
    }

    std::tuple<std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>> SemanticGraph::outlier_pruning(SGLC::Graph graph1,SGLC::Graph graph2,V3d_i match_node1, V3d_i match_node2){
        
        std::vector<Eigen::Vector3d> inlier_match_node1;
        std::vector<Eigen::Vector3d> inlier_match_node2;

        if(match_node1.first.size()<15) {
            inlier_match_node1 = match_node1.first;
            inlier_match_node2 = match_node2.first;
            return {inlier_match_node1,inlier_match_node2};
        }
        
        assert(match_node1.first.size()==match_node2.first.size());
        for(int i=0;i<match_node1.first.size();i++){
            
            int matche_sub_Triangle_th=0;
            std::vector<Eigen::Vector3d> nodeSubgraphTriangle1;
            std::vector<Eigen::Vector3d> nodeSubgraphTriangle2;
            
            // node in graph1
            auto edge_list1 = graph1.edge_matrix.row(match_node1.second[i]);
            std::vector<int> indices1;
            for(int m=0;m<edge_list1.size();m++){
                if(edge_list1[m]<edge_local_triangle && edge_list1[m]!=0) indices1.push_back(m);
            }

            if(indices1.size()<2) { 
                    continue;
            }
            for(int m=0; m<indices1.size();m++){
                if(m==(indices1.size()-1)) break;
                for(int n=m+1; n<indices1.size();n++){
                    std::vector<float> sub_triangle(3);
                    sub_triangle[0] = (match_node1.first[i]-graph1.node_centers[indices1[m]]).norm();
                    sub_triangle[1] = (match_node1.first[i]-graph1.node_centers[indices1[n]]).norm();
                    sub_triangle[1] = (graph1.node_centers[indices1[m]]-graph1.node_centers[indices1[n]]).norm();
                    std::sort(sub_triangle.begin(), sub_triangle.end()); 
                    nodeSubgraphTriangle1.emplace_back(Eigen::Vector3d(sub_triangle[0],sub_triangle[1],sub_triangle[2]));
                }
            }
            // node in graph2
            auto edge_list2 = graph2.edge_matrix.row(match_node2.second[i]);
            std::vector<int> indices2;
            for(int m=0;m<edge_list2.size();m++){
                if(edge_list2[m]<edge_local_triangle && edge_list2[m]!=0) indices2.push_back(m);
            }

            if(indices2.size()<2) { 
                    continue;
            }
            for(int m=0; m<indices2.size();m++){
                if(m==(indices2.size()-1)) break;
                for(int n=m+1; n<indices2.size();n++){
                    std::vector<float> sub_triangle(3);
                    sub_triangle[0] = (match_node2.first[i]-graph2.node_centers[indices2[m]]).norm();
                    sub_triangle[1] = (match_node2.first[i]-graph2.node_centers[indices2[n]]).norm();
                    sub_triangle[1] = (graph2.node_centers[indices2[m]]-graph2.node_centers[indices2[n]]).norm();
                    std::sort(sub_triangle.begin(), sub_triangle.end()); 
                    nodeSubgraphTriangle2.emplace_back(Eigen::Vector3d(sub_triangle[0],sub_triangle[1],sub_triangle[2]));
                }
            }

            for(int m=0;m<nodeSubgraphTriangle1.size();m++){
                for(int n=0;n<nodeSubgraphTriangle2.size();n++){
                    if(abs(nodeSubgraphTriangle1[m].x() - nodeSubgraphTriangle2[n].x())<0.25&&
                        abs(nodeSubgraphTriangle1[m].y() - nodeSubgraphTriangle2[n].y())<0.25&&
                        abs(nodeSubgraphTriangle1[m].z() - nodeSubgraphTriangle2[n].z())<0.25){
                            matche_sub_Triangle_th++;
                    }
                }
            }

            if(indices1.size()>3 && indices2.size()>3){
                if(matche_sub_Triangle_th>=true_node_corres_th) {
                    inlier_match_node1.emplace_back(match_node1.first[i]);
                    inlier_match_node2.emplace_back(match_node2.first[i]);
                }
            }
            else{
                if(matche_sub_Triangle_th>=1) {
                    inlier_match_node1.emplace_back(match_node1.first[i]);
                    inlier_match_node2.emplace_back(match_node2.first[i]);
                }
            }
            

        }
        return {inlier_match_node1,inlier_match_node2};
    }


    std::tuple<Eigen::Isometry3d,double> SemanticGraph::ransac_alignment(std::vector<Eigen::Vector3d> match_node1,
                                                                        std::vector<Eigen::Vector3d> match_node2,
                                                                        std::vector<Eigen::Vector3d> node1,
                                                                        std::vector<Eigen::Vector3d> node2,
                                                                                int max_inter, 
                                                                                int& best_inlier_num){
        assert(match_node1.size()==match_node2.size());
        std::random_device rd;
        std::mt19937 gen(rd());
                
        std::uniform_int_distribution<> dis(0, match_node1.size()-1);
        
        // int best_inlier_num = 0;
        double final_loss = 1000000.0;
        Eigen::Isometry3d final_tans = Eigen::Isometry3d::Identity();


        for(int i=0;i<max_inter;i++){
            int inlier_num =0;
            double loss = 0.0;
            int select_index1 = dis(gen);
            int select_index2 = dis(gen);
            int select_index3 = dis(gen);

            std::vector<Eigen::Vector3d> select_node1(3);
            std::vector<Eigen::Vector3d> select_node2(3);

            select_node1[0] = match_node1[select_index1];
            select_node1[1] = match_node1[select_index2];
            select_node1[2] = match_node1[select_index3];

            select_node2[0] = match_node2[select_index1];
            select_node2[1] = match_node2[select_index2];
            select_node2[2] = match_node2[select_index3];

            Eigen::Isometry3d trans_matrix = solveSVD(select_node1,select_node2);

            // std::cout<<"trans_matrix (rotation):"<<trans_matrix.linear()<<std::endl;
            // std::cout<<"trans_matrix (translation):"<<trans_matrix.translation()<<std::endl;

            //================graph match========================
            float voxel_size = 0.2;
            tsl::robin_map<Eigen::Vector3i, VoxelBlock, VoxelHash> node2_voxel_map;
            
            std::for_each(node2.cbegin(), node2.cend(), [&](const Eigen::Vector3d &point) {
                auto voxel = Eigen::Vector3i((point/voxel_size).template cast<int>());
                auto search = node2_voxel_map.find(voxel);
                if (search != node2_voxel_map.end()) {
                    auto &voxel_block = search.value();
                    voxel_block.AddPoint(point);
                } else {
                    node2_voxel_map.insert({voxel, VoxelBlock{{point}}});
                }
            });


            for( size_t i=0;i< node1.size();i++){

                Eigen::Vector3d trans_match_node1 = trans_matrix*node1[i];
                
                auto kx = static_cast<int>(trans_match_node1[0] / voxel_size);
                auto ky = static_cast<int>(trans_match_node1[1] / voxel_size);
                auto kz = static_cast<int>(trans_match_node1[2] / voxel_size);
                std::vector<Eigen::Vector3i> voxels;

                voxels.reserve(9);
                for (int i = kx - 1; i < kx + 1 + 1; ++i) {
                    for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                        for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                             voxels.emplace_back(i, j, k);
                        }
                    }
                }
                
                std::vector<Eigen::Vector3d> neighboors;
                std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
                    auto search = node2_voxel_map.find(voxel);
                    if (search != node2_voxel_map.end()) {
                        const auto &points = search->second.points;
                        if (!points.empty()) {
                                neighboors.emplace_back(points[0]);
                            }
                        }
                });
                if(neighboors.size()>0){
                    double trans_node_dis = (neighboors[0] - trans_match_node1).norm();
                    inlier_num++;
                    loss = loss + trans_node_dis;
                }

            }
            //TODO: 这里final_loss，计算考虑整个包括整个graph1的点
            if(inlier_num>best_inlier_num){
                best_inlier_num = inlier_num;
                final_tans = trans_matrix;
                final_loss = loss/inlier_num;
                if((((float)inlier_num/node1.size())>0.5 && inlier_num>2)||inlier_num>8){  
                    break;
                }
                
            }
        }
        if(best_inlier_num<3) final_loss=1000000;

        double score = exp(-final_loss);
        return {final_tans,score};
    }
    
     std::tuple<Eigen::Isometry3d,double> SemanticGraph::ransac_alignment(std::vector<Eigen::Vector3d> match_node1,std::vector<Eigen::Vector3d> match_node2,int max_inter, int& best_inlier_num){
        assert(match_node1.size()==match_node2.size());
        std::random_device rd;
        std::mt19937 gen(rd());
                
        std::uniform_int_distribution<> dis(0, match_node1.size()-1);
        
        // int best_inlier_num = 0;
        double final_loss = 1000000.0;
        Eigen::Isometry3d final_tans = Eigen::Isometry3d::Identity();

        for(int i=0;i<max_inter;i++){
            int inlier_num =0;
            double loss = 0.0;

            int select_index1 = dis(gen);
            int select_index2 = dis(gen);
            int select_index3 = dis(gen);

            std::vector<Eigen::Vector3d> select_node1(3);
            std::vector<Eigen::Vector3d> select_node2(3);

            select_node1[0] = match_node1[select_index1];
            select_node1[1] = match_node1[select_index2];
            select_node1[2] = match_node1[select_index3];

            select_node2[0] = match_node2[select_index1];
            select_node2[1] = match_node2[select_index2];
            select_node2[2] = match_node2[select_index3];


            Eigen::Isometry3d trans_matrix = solveSVD(select_node1,select_node2);

            // std::cout<<"trans_matrix (rotation):"<<trans_matrix.linear()<<std::endl;
            // std::cout<<"trans_matrix (translation):"<<trans_matrix.translation()<<std::endl;

            for( size_t i=0;i< match_node1.size();i++){
                double trans_node_dis = (trans_matrix*match_node1[i] - match_node2[i]).norm();
                if(trans_node_dis<inlier_th){  
                    inlier_num++;
                    loss = loss + trans_node_dis;
                }
            }
            if(inlier_num>best_inlier_num){
                best_inlier_num = inlier_num;
                final_tans = trans_matrix;
                final_loss = loss/inlier_num;
            }
        }
        if(best_inlier_num<3) final_loss=1000000;

        double score = exp(-final_loss);
        return {final_tans,score};
    }

    Eigen::Isometry3d SemanticGraph::solveSVD(std::vector<Eigen::Vector3d> match_node1,std::vector<Eigen::Vector3d> match_node2){

        if(match_node1.empty() || match_node2.empty() || match_node1.size()!=match_node2.size()){
            std::cout<<"Error! solve SVD: input pointcloud size is not same or empty"<<std::endl;
        }

        int N=match_node1.size();
        Eigen::Vector3d node1_sum{0,0,0}, node2_sum{0,0,0};
        for(int i=0;i<N;i++){
            node1_sum+=match_node1[i];
            node2_sum+=match_node2[i];
        }
        Eigen::Vector3d node1_mean = node1_sum/N;
        Eigen::Vector3d node2_mean = node2_sum/N;


        std::vector<Eigen::Vector3d> node1_list,node2_list;
        for(int i=0;i<N;i++){
            node1_list.emplace_back(match_node1[i]-node1_mean);
            node2_list.emplace_back(match_node2[i]-node2_mean);
        }


        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for(int i=0;i<N;i++){
            W += node1_list[i] * node2_list[i].transpose();
        }
        for (size_t i = 0; i < W.size(); i++) {
            if (std::isnan(W(i))) {
                std::cout << "error: the input points are wrong, can't solve with SVD." << std::endl;
            }
        }

        // svd 
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d E = Eigen::Matrix3d::Identity();

        Eigen::Matrix3d R_temp = V*(U.transpose());


        if (R_temp.determinant() < 0) {
            E(2,2) = -1;
        }
        Eigen::Matrix3d R = V * E * (U.transpose());
        Eigen::Vector3d T = node2_mean - R * node1_mean;

        // 构建位姿矩阵
        Eigen::Isometry3d trans_matrix = Eigen::Isometry3d::Identity();
        trans_matrix.linear() = R;
        trans_matrix.translation() = T;   

        return  trans_matrix;
    }

    double SemanticGraph::get_alignment_score(SGLC::Graph graph1, SGLC::Graph graph2, Eigen::Isometry3d trans){
        
      pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kd_tree(new pcl::KdTreeFLANN<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      for(size_t i=0; i<graph2.node_centers.size();i++){
            pcl::PointXYZ pi(graph2.node_centers[i].x(),graph2.node_centers[i].y(),graph2.node_centers[i].z());
            target_cloud->push_back(pi);
      }
      double total_distance= 0.0;
      int inlier = 0;
      kd_tree->setInputCloud(target_cloud);

      for(size_t i=0; i<graph1.node_centers.size();i++){
            Eigen::Vector3d query_node = graph1.node_centers[i];
            query_node = trans*query_node;
            pcl::PointXYZ query_point(query_node.x(),query_node.y(),query_node.z());
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            if (kd_tree->nearestKSearch(query_point, 1, pointIdxNKNSearch,
                                pointNKNSquaredDistance) > 0){     
                Eigen::Vector3d nearestNode = graph2.node_centers[pointIdxNKNSearch[0]];
                int nearestLables = graph2.node_labels[pointIdxNKNSearch[0]];
                
                if(graph1.node_labels[i]==nearestLables && pointNKNSquaredDistance[0]<0.2){ 
                    inlier++;
                    total_distance +=  pointNKNSquaredDistance[0];
                }
            }
      }


    double inlier_ratio = (double)inlier/graph1.node_centers.size();
    double fitScore = 10000;
    Eigen::Vector3d translation = trans.matrix().block<3, 1>(0, 3);
    double distance = translation.norm();
    fitScore = total_distance/graph1.node_centers.size();
    return fitScore;
    }

    

    Eigen::MatrixXf SemanticGraph::gen_background_descriptors(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud){

        auto ring_step = (max_dis - min_dis) / rings;
        auto sector_step = 360. / sectors;
        Eigen::MatrixXf Ssc_background = Eigen::MatrixXf::Zero(rings,sectors);
        
        for(int i=0;i < (int)filtered_pointcloud->points.size(); i++){
            auto label = filtered_pointcloud->points[i].label;

            if (order_vec[label] > 0){
                
                double distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);

                if (distance >= max_dis || distance < min_dis)
                    {continue;}
                int sector_id = cv::fastAtan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x)/ sector_step;
                int ring_id = (distance - min_dis) / ring_step;

                if (ring_id >= rings || ring_id < 0)
                    {continue;}
                if (sector_id >= sectors || sector_id < 0)
                    {continue;}
                if (order_vec[label] > order_vec[Ssc_background(ring_id, sector_id)])
                {   
                    Ssc_background(ring_id, sector_id) = label;
                }
            }
        }
        return Ssc_background;
    }

    std::vector<float> SemanticGraph::gen_back_descriptors_global(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud){

        auto ring_step = (max_dis - min_dis) / rings;
        auto sector_step = 360. / sectors;
        Eigen::MatrixXf Ssc_background_build = Eigen::MatrixXf::Zero(rings,sectors);
        Eigen::MatrixXf Ssc_background_road = Eigen::MatrixXf::Zero(rings,sectors);
        std::vector<float> background_dep;
        for(int i=0;i < (int)filtered_pointcloud->points.size(); i++){
            auto label = filtered_pointcloud->points[i].label;
   
            if (order_vec[label] > 0){

                double distance = std::sqrt(filtered_pointcloud->points[i].x * filtered_pointcloud->points[i].x + filtered_pointcloud->points[i].y * filtered_pointcloud->points[i].y);

                if (distance >= max_dis || distance < min_dis)
                    {continue;}
                int sector_id = cv::fastAtan2(filtered_pointcloud->points[i].y, filtered_pointcloud->points[i].x)/ sector_step;
                int ring_id = (distance - min_dis) / ring_step;
                
                if (ring_id >= rings || ring_id < 0)
                    {continue;}
                if (sector_id >= sectors || sector_id < 0)
                    {continue;}
                if (label == 13) //build
                {
                    Ssc_background_build(ring_id, sector_id) = 1;
                }
                else if (label == 9) //road
                {
                    Ssc_background_road(ring_id, sector_id) = 1;
                }
            }
        }
        Eigen::VectorXf build_rowsum = Ssc_background_build.rowwise().sum();
        Eigen::VectorXf road_rowsum = Ssc_background_road.rowwise().sum();
        background_dep.insert(background_dep.end(),build_rowsum.data(),build_rowsum.data()+build_rowsum.size());
        background_dep.insert(background_dep.end(),road_rowsum.data(),road_rowsum.data()+road_rowsum.size());

        float sum_of_squares_node = std::inner_product(background_dep.begin(), background_dep.end(), background_dep.begin(), 0.0f);
        float l2_norm_node = std::sqrt(sum_of_squares_node);
        for (auto it = background_dep.begin(); it != background_dep.end(); ++it) {
            *it /= l2_norm_node; 
        }
        return background_dep;
    }

    double SemanticGraph::calculate_sim(Eigen::MatrixXf &desc1, Eigen::MatrixXf &desc2){
        double similarity = 0;
        int sectors = desc1.cols();
        int rings = desc1.rows();
        int valid_num = 0;
        for (int p = 0; p < sectors; p++)
        {
            for (int q = 0; q < rings; q++)
            {
                if (desc1(q, p) == 0 && desc2(q, p) == 0)
                {
                    continue;
                }
                valid_num++;

                if (desc1(q, p) == desc2(q, p))
                {
                    similarity++;
                }
            }
        }

        return similarity / valid_num;
    }

    std::pair<std::vector<Eigen::Vector3d>, std::vector<int>>SemanticGraph::pcl2eigen(pcl::PointCloud<pcl::PointXYZL>::Ptr pointcloud){
        std::vector<Eigen::Vector3d> eigen_pc(pointcloud->size());
        std::vector<int> eigen_pc_label(pointcloud->size());

        tbb::parallel_for(size_t(0),pointcloud->size(), [&](size_t i){
            eigen_pc[i] =  Eigen::Vector3d((*pointcloud)[i].x, (*pointcloud)[i].y, (*pointcloud)[i].z);
            eigen_pc_label[i] = (*pointcloud)[i].label;
        });
        return std::make_pair(eigen_pc,eigen_pc_label);
    }

}

