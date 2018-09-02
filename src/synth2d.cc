#include "synth2d.h"
#include "pose2d.h"
#include "poseocv.h"
#include "pose-other.h"
#include "common.h"
#include "Ransac.hh"
#include "math.hh"

#ifdef USE_THEIA_RANSAC
static theia::RansacParameters RANSAC_parameter;
#else
static templransac::RANSACParams RANSAC_parameter(10);
#endif
inline void* RANSAC_params(double error_threshold = 9, double sample_inlier_probability = 0.99)
{
#ifdef USE_THEIA_RANSAC
    RANSAC_parameter.error_thresh = error_threshold;
    RANSAC_parameter.failure_probability = 1.0 - sample_inlier_probability;
    RANSAC_parameter.max_iterations = 10000;
#else
   RANSAC_parameter.error_threshold = error_threshold;
   RANSAC_parameter.sample_inlier_probability = sample_inlier_probability;
#endif
   return static_cast<void*>(&RANSAC_parameter);
}
namespace synth2d
{
   void create_test_data(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                         std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations)
//---------------------------------------------------------------------------------------------
   {
      const double g = 9.80665;
      std::random_device rand_device;
      std::mt19937 rand_generator;
      std::uniform_real_distribution<double> distribution(-5, 5);

      Eigen::Vector3d train_g = Eigen::Vector3d(0, g, 0);
      Eigen::Vector3d query_g = Eigen::Vector3d(g, 0, 0);
      gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
      Eigen::Quaterniond Q = mut::euler2Quaternion(0, 0, mut::degreesToRadians(90.0));
      Eigen::Matrix3d R = Q.toRotationMatrix();
      rotation_matrices.push_back(R);
      translations.emplace_back(10, 5, 3);

      train_g = Eigen::Vector3d(0, g, 0);
      query_g = Eigen::Vector3d(g, 0, 0);
      gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
      Q = mut::euler2Quaternion(0, 0, mut::degreesToRadians(90.0));
      R = Q.toRotationMatrix();
      rotation_matrices.push_back(R);
      translations.emplace_back(10, 5, -3);

      train_g = Eigen::Vector3d(0, g, 0);
      query_g = Eigen::Vector3d(g / 2, g / 2, 0);
      gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
      Q = mut::euler2Quaternion(0, 0, mut::degreesToRadians(45.0));
      R = Q.toRotationMatrix();
      rotation_matrices.push_back(R);
      translations.emplace_back(distribution(rand_generator), distribution(rand_generator),
                                distribution(rand_generator));

      train_g = Eigen::Vector3d(0, g, 0);
      query_g = Eigen::Vector3d(0, -g, 0);
      gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
      Q = mut::euler2Quaternion(0, 0, mut::degreesToRadians(-180.0));
      R = Q.toRotationMatrix();
      rotation_matrices.push_back(R);
      translations.emplace_back(distribution(rand_generator), distribution(rand_generator),
                                distribution(rand_generator));

      int n = 100;
      for (int i=0; i<n; i++)
      {
         if ( (mut::random_vector3d_fixed_norm_kludge(g, train_g, rand_generator)) &&
              (mut::random_vector3d_fixed_norm_kludge(g, query_g, rand_generator)) )
         {
            gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
            Q.setFromTwoVectors(query_g, train_g);
            R = Q.toRotationMatrix();
            rotation_matrices.push_back(R);
            translations.emplace_back(distribution(rand_generator), distribution(rand_generator),
                                      distribution(rand_generator));
         }
         else
            n++;
      }

      train_g = Eigen::Vector3d(-0.001095810, 7.897157669, 5.814230919);
      query_g = Eigen::Vector3d(-0.440319300, 9.011748314, 3.842511892);
      gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
      Q = mut::euler2Quaternion(mut::degreesToRadians(13.28760549), mut::degreesToRadians(1.28268859), mut::degreesToRadians(-2.24394751));
      R = Q.toRotationMatrix();
      rotation_matrices.push_back(R);
      translations.emplace_back(distribution(rand_generator), distribution(rand_generator),
                                distribution(rand_generator));

      train_g = Eigen::Vector3d(-0.001095810, 7.897157669, 5.814230919);
      query_g = Eigen::Vector3d(-1.700023532, 6.482434750, 7.159492493);
      gravity_vectors.emplace_back(std::make_pair(train_g, query_g));
      Q = mut::euler2Quaternion(mut::degreesToRadians(169.04700715), mut::degreesToRadians(173.30330220), mut::degreesToRadians(172.52050993));
      R = Q.toRotationMatrix();
      rotation_matrices.push_back(R);
      translations.emplace_back(distribution(rand_generator), distribution(rand_generator),
                                distribution(rand_generator));
   }

   inline void add_noise(Eigen::Vector3d& tp, std::normal_distribution<double>* noise_distribution,
                         std::mt19937* noise_generator, std::uniform_real_distribution<double>* outlier_distribution,
                         const double ransac_outlier_probability)
//---------------------------------------------------------------------------------------------------------------
   {
      double xnoise = (*noise_distribution)(*noise_generator);
      double ynoise = (*noise_distribution)(*noise_generator);
      if ((ransac_outlier_probability > 0) &&
          ((*outlier_distribution)(*noise_generator) <= ransac_outlier_probability))
      {
         tp[0] += xnoise;
         tp[1] += ynoise;
      }
      else if (ransac_outlier_probability <= 0)
      {
         tp[0] += xnoise;
         tp[1] += ynoise;
      }
   }

   void generate_points(std::vector<std::pair<cv::Point3d, cv::Point3d>>& image_pts,
                        std::vector<std::pair<cv::Point3d, cv::Point3d>>& camera_pts,
                        const size_t no, const double minx, const double maxx, const double miny, const double maxy,
                        double depth, const Eigen::Matrix3d& K, const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
                        const double noise_deviation = 0, const double ransac_outlier_probability = 0)
//----------------------------------------------------------------------------------------------------------------
   {
      image_pts.clear();
      camera_pts.clear();
      std::unique_ptr<std::random_device> rand_device;
      std::unique_ptr<std::mt19937> rand_generator;
      rand_device.reset(new std::random_device());
      rand_generator.reset(new std::mt19937((*rand_device)()));
      std::unique_ptr<std::normal_distribution<double>> noise_distribution;
      std::unique_ptr<std::uniform_real_distribution<double>> outlier_distribution;
      std::unique_ptr<std::uniform_real_distribution<double>> x_distribution;
      std::unique_ptr<std::uniform_real_distribution<double>> y_distribution;
      if (noise_deviation != 0)
      {
         noise_distribution.reset(new std::normal_distribution<double>(0, noise_deviation));
         if (ransac_outlier_probability > 0)
            outlier_distribution.reset(new std::uniform_real_distribution<double>(0, 1));
      }
      x_distribution.reset(new std::uniform_real_distribution<double>(minx, maxx));
      y_distribution.reset(new std::uniform_real_distribution<double>(miny, maxy));
      double xnoise = 0, ynoise = 0, transformed_depth = 0;
      Eigen::Matrix3d EKI = K.inverse();
      size_t n= no;
      if (depth < 0)
         depth = 1;
      for (size_t i = 0; i < no; i++)
      {
         double x = ((*x_distribution)(*rand_generator)), y = ((*y_distribution)(*rand_generator));
         Eigen::Vector3d tp(x, y, depth);
         Eigen::Vector3d tip = K * tp;
         tip /= tip[2];
         Eigen::Vector3d qp = R * tp + t;
         Eigen::Vector3d qip = K * qp;
         if (mut::near_zero(qip[2], 0.0000001))
         {
            n++;
            continue;
         }
         else
            qip /= qip[2];
         if (noise_deviation != 0)
            add_noise(qip, noise_distribution.get(), rand_generator.get(), outlier_distribution.get(),
                      ransac_outlier_probability);
         image_pts.emplace_back(
               std::make_pair(cv::Point3d(tip[0], tip[1], tip[2]), cv::Point3d(qip[0], qip[1], qip[2])));
         camera_pts.emplace_back(std::make_pair(cv::Point3d(tp[0], tp[1], tp[2]), cv::Point3d(qp[0], qp[1], qp[2])));
      }
   }

   void synthesize(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                   std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations,
                   double maxdepth)
//----------------------------------------------------------------------------------------------------------
   {
      std::vector<std::pair<cv::Point3d, cv::Point3d>> image_pts, camera_pts;
//      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eK((double*) K.data);
      Eigen::Matrix3d K = Eigen::Matrix3d::Identity(), KI = K.inverse();
      Eigen::Quaterniond Q;
      Eigen::Vector3d T;
      std::random_device rand_device;
      std::mt19937 rand_generator;
      std::uniform_real_distribution<double> distribution(0, std::abs(maxdepth));
      for (size_t i = 0; i < gravity_vectors.size(); i++)
      {
         const Eigen::Vector3d& train_g = gravity_vectors[i].first;
         const Eigen::Vector3d& query_g = gravity_vectors[i].second;
         const Eigen::Matrix3d& R = rotation_matrices[i];
         const Eigen::Vector3d& t = translations[i];
         double depth = (maxdepth > 0) ? distribution(rand_generator) : -1;
         double minx, maxx, miny, maxy;
         if (depth < 0)
         {
            minx = miny = -10;
            maxx = maxy = 10;
         }
         else
         {
            minx = miny = -maxdepth;
            maxx = maxy = maxdepth;
         }
         generate_points(image_pts, camera_pts, 3, minx, maxx, miny, maxy, depth,
                         K, R, t);
         std::cout << train_g.transpose() << " to " << query_g.transpose() << " depth " << depth << std::endl
                   << "-------------------------------------------------------------------" << std::endl;

         std::cout << "Gravity Pose with Depth " << train_g.transpose() << " to " << query_g.transpose() << " depth "
                   << depth << std::endl  << "---------------------------------------------------" << std::endl;
         if (depth <= 0)
            pose2d::pose(image_pts, train_g, query_g, K, Q, T);
         else
            pose2d::pose(image_pts, train_g, query_g, depth, K, Q, T);
         display_rotation(Q);
         std::cout << "Translation: " << T.transpose() << " " << t.transpose() << std::endl;
         double max_error, mean_error;
         if (depth > 0)
         {
            project2d(K, Q.toRotationMatrix(), T, image_pts, depth, max_error, mean_error);
            std::cout << "Max Error " << max_error << ", Mean Error " << mean_error << std::endl;

            std::vector<cv::Point3d> tpts, qpts;
            for (const std::pair<cv::Point3d, cv::Point3d>& pp : image_pts)
            {
               tpts.emplace_back(pp.first);
               qpts.emplace_back(pp.second);
            }
            pose2d::pose_translation(KI, tpts, qpts, depth, Q.toRotationMatrix(), T);
            std::cout << "**Translation only " << T.transpose() << std::endl;
            pose2d::pose_translation(KI, image_pts, depth, Q.toRotationMatrix(), T);
            std::cout << "**Translation only 2 " << T.transpose() << std::endl;
         }
         generate_points(image_pts, camera_pts, 100, minx, maxx, miny, maxy, depth, K, R, t);
         if (depth <= 0)
            pose2d::pose_ransac(image_pts, train_g, query_g, K, Q, T, RANSAC_params(0.5), 3);
         else
            pose2d::pose_ransac(image_pts, train_g, query_g, depth, K, Q, T, RANSAC_params(0.5), 3);
         std::cout << "**RANSAC: " << T.transpose() << " " << t.transpose() << std::endl;
         if (depth > 0)
         {
            project2d(K, Q.toRotationMatrix(), T, image_pts, depth, max_error, mean_error);
            std::cout << "Max Error " << max_error << ", Mean Error " << mean_error << std::endl;
         }
      }
   }

   void synthesize_noisy(std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& gravity_vectors,
                         std::vector<Eigen::Matrix3d>& rotation_matrices, std::vector<Eigen::Vector3d>& translations,
                         const double maxdepth, const double start_deviation, const double end_deviation,
                         const double inc_deviation)
//-----------------------------------------------------------------------------------------------------------------
   {
      std::vector<std::pair<cv::Point3d, cv::Point3d>> image_pts, camera_pts;
//      cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
//      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) K.data);
      Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
      Eigen::Quaterniond Q, QQ;
      Eigen::Vector3d T;
      std::random_device rand_device;
      std::mt19937 rand_generator;
      std::uniform_real_distribution<double> distribution(0, std::abs(maxdepth));
      for (size_t i = 0; i < gravity_vectors.size(); i++)
      {
         const Eigen::Vector3d& train_g = gravity_vectors[i].first;
         const Eigen::Vector3d& query_g = gravity_vectors[i].second;
         const Eigen::Matrix3d& R = rotation_matrices[i];
         const Eigen::Vector3d& t = translations[i];
         double depth = (maxdepth > 0) ? distribution(rand_generator) : -1;
         double minx, maxx, miny, maxy;
         if (depth <= 0)
         {
            minx = miny = -10;
            maxx = maxy = 10;
         }
         else
         {
            minx = miny = -maxdepth;
            maxx = maxy = maxdepth;
         }
         for (double deviation = start_deviation; deviation < end_deviation; deviation += inc_deviation)
         {
            generate_points(image_pts, camera_pts, 3, minx, maxx, miny, maxy, depth, K, R, t, deviation);
            std::cout << "Noise (pixels) " << deviation << std::endl;
            if (depth <= 0)
               pose2d::pose(image_pts, train_g, query_g, K, Q, T);
            else
               pose2d::pose(image_pts, train_g, query_g, depth, K, Q, T);
            display_rotation(Q, true, true);

            std::cout << " Translation: " << T.transpose() << " (" << t.transpose() << ")" << std::endl;
            double max_error, mean_error;
            if (depth > 0)
            {
               project2d(K, Q.toRotationMatrix(), T, image_pts, depth, max_error, mean_error);
               std::cout << "Max Error " << max_error << ", Mean Error " << mean_error << std::endl;
            }
            generate_points(image_pts, camera_pts, 100, minx, maxx, miny, maxy, depth, K, R, t, deviation);
            if (depth <= 0)
               pose2d::pose_ransac(image_pts, train_g, query_g, K, Q, T, RANSAC_params(), 3);
            else
               pose2d::pose_ransac(image_pts, train_g, query_g, depth, K, Q, T, RANSAC_params(1.5), 3);
            std::cout << "**RANSAC (100 points): " << T.transpose() << " | " << t.transpose() << std::endl;
            if (depth > 0)
            {
               project2d(K, Q.toRotationMatrix(), T, image_pts, depth, max_error, mean_error);
               std::cout << "Max Error " << max_error << ", Mean Error " << mean_error << std::endl;
            }
            std::cout << "###############################################################################" << std::endl;
         }
      }
   }
}