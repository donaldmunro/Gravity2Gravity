#ifndef TESTING3D_3AEF7D1E_A860_4B68_B0B8_4967AA30F28C_H
#define TESTING3D_3AEF7D1E_A860_4B68_B0B8_4967AA30F28C_H
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

namespace synth3d
{
   void create_test_data(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                         std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations);

   void synthesize(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                   std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations);

   void synthesize_noisy(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                         std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations,
                         const double start_deviation = 0.2, const double end_deviation = 0.8,
                         const double inc_deviation = 0.1);
};

#endif //TESTING3D_3AEF7D1E_A860_4B68_B0B8_4967AA30F28C_H
