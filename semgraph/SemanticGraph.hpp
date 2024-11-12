// This file is covered by the LICENSE file in the root of this project.
// developed by Neng Wang, <neng.wang@hotmail.com>
// reference: CVC_Cluster : https://github.com/wangx1996/Lidar-Segementation

#pragma once

#include <vector>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include <tbb/parallel_for.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <chrono>

#include "Coreutils.hpp" 
#include "SemanticCluster.hpp"
#include "Hungarian.hpp"
#include "FastIcp.hpp"
#include "PlaneIcp.hpp"
namespace SGLC
{
    class SemanticGraph
    {
    private:
        struct VoxelHash{
            size_t operator()(const Eigen::Vector3i &voxel) const {
                const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
                return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
                }
            }; 
        struct VoxelBlock {
            std::vector<Eigen::Vector3d> points;
            inline void AddPoint(const Eigen::Vector3d &point) {
                points.push_back(point);
            }
        };
        bool show=false;
        bool remap=true;
        bool cluster_view = false;

 
        double deltaA = 2;
        double deltaR = 0.35; 
        double deltaP = 1.2; 

        // graph
        double edge_th = 5;
        double sub_interval = 30;
        int graph_node_dimension = 60;
        double box_diff_th =  2;
        double edge_local_triangle = 20;
        int true_node_corres_th = 2;
        double graph_sim_th = 0.5;
        double back_sim_th = 0.5 ;
        double back_sim_th_without_node = 0.7;
        // ransac 
        double inlier_th = 0.2;
        // scan context - like background descriptors
        double max_dis=50;
        double min_dis=5;
        int rings=24;
        int sectors=360;
        int sectors_range=360;
        std::vector<int> order_vec = {0, 22, 0, 0, 21, 20, 0, 0, 0, 10, 11, 12, 13, 15, 16, 14, 17, 9, 18, 19, 0, 0, 0, 0, 0, 0};

        // Ransac and ICP
        int max_icp_iteration = 500;
        double estimation_threshold = 0.0001;
        
        YAML::Node learning_map;
        std::vector<int> label_map;
        typedef std::tuple<u_char, u_char, u_char> Color;
        std::map<uint32_t, Color> _color_map, _argmax_to_rgb;
        std::shared_ptr<pcl::visualization::CloudViewer> viewer;
    public:
        int frame_count = 0;
        double total_time = 0;
        SemanticGraph(std::string conf_file);
        ~SemanticGraph();

        // load LiDAR scan
        std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> load_cloud(std::string file_cloud, 
                                                                            std::string file_label);
        
        // graph
        SGLC::Graph build_graph(std::vector<SGLC::Bbox> cluster_boxes, 
                                                        double edge_dis_th,
                                                        double subinterval);
        
        Eigen::MatrixXf matrix_decomposing(Eigen::MatrixXf MatrixInput,
                                                    int Dimension);

        std::vector<float> gen_graph_descriptors(SGLC::Graph graph,
                                                    float edge_th,
                                                    int subinterval);

        std::vector<float> gen_scan_descriptors(std::string cloud_file,
                                                std::string label_file);

        // background 
        Eigen::MatrixXf gen_background_descriptors(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud);

        std::vector<float> gen_back_descriptors_global(pcl::PointCloud<pcl::PointXYZL>::Ptr filtered_pointcloud);


        
        // mathcing node and outlier pruning
        std::tuple<V3d_i,V3d_i>  find_correspondences_withidx(SGLC::Graph graph1,
                                                                SGLC::Graph graph2);

        std::tuple<std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>> outlier_pruning(SGLC::Graph graph1,SGLC::Graph graph2,
                                                                                                V3d_i match_node1, V3d_i match_node2);

        // ransac
        std::tuple<Eigen::Isometry3d,double>ransac_alignment(std::vector<Eigen::Vector3d> match_node1,
                                                            std::vector<Eigen::Vector3d> match_node2,
                                                            std::vector<Eigen::Vector3d> node1,
                                                            std::vector<Eigen::Vector3d> node2,
                                                            int max_inter,int& best_inlier_num);
                                                            
        std::tuple<Eigen::Isometry3d,double> ransac_alignment(std::vector<Eigen::Vector3d> match_node1,
                                                            std::vector<Eigen::Vector3d> match_node2,
                                                                int max_inter, int& best_inlier_num);

        Eigen::Isometry3d solveSVD(std::vector<Eigen::Vector3d> match_node1,
                                    std::vector<Eigen::Vector3d> match_node2);

        // similarity 
        double loop_pairs_similarity(std::string cloud_file1, std::string cloud_file2, 
                                        std::string label_file1, std::string label_file2);
        double get_cos_simscore(const std::vector<float> vec1, const std::vector<float> vec2);
        double get_cos_simscore(const std::vector<int> vec1, const std::vector<int> vec2);

        double getMold(const std::vector<float> vec);
        double getMold(const std::vector<int> vec);

        double get_alignment_score(SGLC::Graph graph1, SGLC::Graph graph2, Eigen::Isometry3d trans);

        double calculate_sim(Eigen::MatrixXf &desc1, Eigen::MatrixXf &desc2);

        // poses estimation
        Eigen::Matrix4d loop_poses_estimation(std::string cloud_file1, std::string cloud_file2,
                                             std::string label_file1, std::string label_file2);

        std::pair<std::vector<Eigen::Vector3d>, std::vector<int>> pcl2eigen(pcl::PointCloud<pcl::PointXYZL>::Ptr pointcloud);
    };
    

    

    
    

}