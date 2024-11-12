/*
    This is simplified and modified from kiss-icp(https://github.com/PRBonn/kiss-icp.git)
*/


#pragma once

#define FMT_HEADER_ONLY
#include "fmt/format.h"

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tsl/robin_map.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

using Vector4dVector = std::vector<Eigen::Vector4d>;
using Vector4dVectorTuple = std::tuple<Vector4dVector, Vector4dVector>;
typedef std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> V3d_i;
namespace Eigen {
    using Matrix6d = Eigen::Matrix<double, 6, 6>;
    using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

constexpr int MAX_NUM_ITERATIONS_ = 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;
class FastIcp
{
private:
    /* data */
    struct VoxelHash {
    size_t operator()(const Eigen::Vector3i &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
        return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
        }
    }; 
    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector4d> points;
        int num_points_;
        // int num_pole_like_points_;
        inline void AddPoint(const Eigen::Vector4d &point) {
            // if(int(point(3))!=71 && int(point(3))!=80){
                if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
            // }
            // else{
            //     if (points.size() < static_cast<size_t>(num_pole_like_points_)) points.push_back(point);
            // }

        }
    };

    struct ResultTuple {
        ResultTuple(std::size_t n) {
            source.reserve(n);
            target.reserve(n);
        }
        std::vector<Eigen::Vector4d> source;
        std::vector<Eigen::Vector4d> target;
    };
    struct ResultTupleRe {
        ResultTupleRe() {
            JTJ.setZero();
            JTr.setZero();
        }

        ResultTupleRe operator+(const ResultTupleRe &other) {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            return *this;
        }

        Eigen::Matrix6d JTJ;
        Eigen::Vector6d JTr;
    };
    float down_voel_size_ = 0.5;
    float down_voel_size_pole_ = 0.2;


    std::vector<Eigen::Vector3d> CorrectKITTIScan(const std::vector<Eigen::Vector3d> &frame);
    V3d_i VoxelDownsampleSemantic(const V3d_i &frame,  double voxel_size, double voxel_size_pole);
    std::vector<Eigen::Vector4d>  FusePointsAndLabels(const std::pair<std::vector<Eigen::Vector3d>,
                                                                    std::vector<int>> &frame);
    V3d_i SeparatePointsAndLabels(const std::vector<Eigen::Vector4d> &frame);
    Sophus::SE3d AlignClouds(const std::vector<Eigen::Vector4d> &source4d,
                         const std::vector<Eigen::Vector4d> &target4d);
    void TransformPoints4D(const Sophus::SE3d &T, std::vector<Eigen::Vector4d> &points);
    // voxel map
    float map_voel_size_ = 0.1;
    int max_points_per_voxel_ = 5;
    tsl::robin_map<Eigen::Vector3i, VoxelBlock, VoxelHash> map_;
    void AddPoints(const std::vector<Eigen::Vector4d> &points);
    Vector4dVectorTuple GetCorrespondences(
                    const Vector4dVector &points, double max_correspondance_distance);
    Sophus::SE3d RegisterFrameSemantic(const std::vector<Eigen::Vector4d> &frame,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance);
public:
    FastIcp(float voxel_size_downsample,float voxel_size_downsample_pole,float voxel_size_map,int max_points_per_voxel);
    ~FastIcp();
    Eigen::Matrix4d get_trans(V3d_i cloudA, V3d_i cloudB, Eigen::Matrix4d initTrans);




    
};


