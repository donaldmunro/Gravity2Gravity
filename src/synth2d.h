#ifndef GRAVITYPOSE_TESTING2D_H
#define GRAVITYPOSE_TESTING2D_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

namespace synth2d
{
   void create_test_data(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                         std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations);

   void synthesize(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                   std::vector<Eigen::Matrix3d>& rotation_matrices,
                   std::vector<Eigen::Vector3d>& translations,
                   double maxdepth);

   void synthesize_noisy(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                         std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations,
                         const double train_depth, const double start_deviation = 0.2, const double end_deviation = 0.8,
                         const double inc_deviation = 0.1);

}

#endif //GRAVITYPOSE_TESTING2D_H
