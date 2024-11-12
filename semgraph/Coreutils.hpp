

#pragma once

#include <Eigen/Core>
#include <vector>

namespace SGLC{

struct Bbox
{
    Eigen::Vector3d center;
    Eigen::Vector3d dimension;
    double theta = 0.0;
    int label = -1;
    double score = 0.0;
};

struct Graph
{
    std::vector<int> node_labels;
    std::vector<int> node_stable;
    std::vector<Eigen::Vector3d> node_centers;
    std::vector<Eigen::Vector3d> node_dimensions;
    std::vector<std::vector<float>> node_desc; 
    std::vector<std::pair<int,int>> edges;
    std::vector<double> edge_value;
    std::vector<double> edge_weights;
    Eigen::MatrixXf graph_matrix;
    Eigen::MatrixXf edge_matrix;
    int car_num=0;
    int trunk_num=0;
    int pole_like_num=0;
};


}


