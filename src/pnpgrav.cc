#include <iostream>
#include <iomanip>
#include <cstdarg>
#include <math.h>
#include <fstream>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <memory>
#include <algorithm>

#include <yaml-cpp/yaml.h>

#include <Eigen/Core>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv/cv.hpp>

#include "optionparser.h"
#include "math.hh"
#include "pose3d.h"
#include "synth3d.h"
#include "SelectMatchedPoints.h"
#include "common.h"
#include "poseocv.h"
#include "Ransac.hh"
#include "Optimization.h"

#ifdef FILESYSTEM_EXPERIMENTAL
#include <experimental/filesystem>
namespace filesystem = std::experimental::filesystem;
#elif defined(STD_FILESYSTEM)
#include <filesystem>
namespace filesystem = std::filesystem;
#else
#include <boost/filesystem.hpp>
namespace filesystem = boost::filesystem;
#endif

void help();

struct Arg : public option::Arg
//=============================
{
   static void printError(const char *msg1, const option::Option &opt, const char *msg2)
   //-----------------------------------------------------------------------------------
   {
      fprintf(stderr, "%s", msg1);
      fwrite(opt.name, (size_t) opt.namelen, 1, stderr);
      fprintf(stderr, "%s", msg2);
      help();
   }

   static option::ArgStatus Required(const option::Option &option, bool msg)
   //-----------------------------------------------------------------------
   {
      if (option.arg != 0)
         return option::ARG_OK;

      if (msg) printError("Option '", option, "' requires an argument\n");
      return option::ARG_ILLEGAL;
   }

   static option::ArgStatus Unknown(const option::Option &option, bool msg)
   //----------------------------------------------------------------------
   {
      if (msg) printError("Unknown option '", option, "'\n");
      return option::ARG_ILLEGAL;
   }
};

const char *USAGE_DESC =
      "USAGE: pnpgrav [options] train-image train-gravity-vector query-image query-gravity-vector"
      "       where gravity-vector is 3 numbers separated by commas without spaces (or if there are spaces then surrounded by quotes)"
      "       eg 9.6019897,0.8189485,1.8170115\n"
      "OR   : pnpgrav [options] train-yaml-file query-yaml-file\n"
      "       where the yaml files are from TangoCamera (https://github.com/donaldmunro/TangoCamera)"
      "       and the image files are assumed to have the same base name as the yaml file.\n"
      "OR   : pnpgrav -s (--synth) when using synthesized points for testing\n"
      "In the non-synthesized cases 3 or more 3D points need to be specified using -p\n";
const char *CALIBRATION_DESC =
      "-c <arg>, \t--calibration=<arg>\tSpecify camera calibration details. <arg> format fx,fy,ox,oy where:\n"
            "fx = Focal length (X), fy = Focal length (Y), ox = X offset, oy = Y offset.\n"
            "Example -c 776.728236,776.0521739,370.32,300.88\n"
            "Alternately a path to a file containing the calibration parameters in the same comma delimited format\n"
            "(without the -c) may be specified\n"
            "If yaml files are specified then should not be necessary as the calibration should be in the "
            "yaml file. If specified in addition to the yaml file then it overrides the yaml file calibration.";
const char *REMAP_GRAVITY_DESC =
"-m <arg>, \t--remap <arg>\tPerform Android remap coordinate system on gravity vector (when the vector was recorded while in "
"one of the landscape modes or reverse portrait mode). <arg> is the device rotation (int) where:"
" 0 =   device default (portrait for phones and many but not all tablets.\n"
" 90 =  90 degree rotation (landscape for devices with default of portrait)\n"
" 180 = 180 degree rotation (reversed portrait for devices with default of portrait)\n"
" 270 = 270 degree rotation (reversed landscape for devices with default of portrait)\n"
"The device orientation should already be defined in the yaml files if they are used.";
const char *POINT_DESC =
      "-p <arg>, \t--3dpt <arg>\tSpecify a 3D model point or a file containing 3d points. Multiple -p  "
      "arguments may be specified, one for each point. Alternately specify a file containing a 3D points "
      "on each line. Command line points must be 3 numbers separated by commas without spaces (or if there "
      "are spaces then surrounded by quotes). Points in a file must be 3 numbers separated by commas followed"
      "by a carriage return\n";
const char *TANGOCAM_DESC =
      "-T \t --tango \tSpecifies that 3D points entered using -p (--3dpt) were captured from a TangoCamera "
      "(https://github.com/donaldmunro/TangoCamera) point cloud (ply file) so the 2D projections can be "
      "calculated from intrinsics in the TangoCamera .yaml file.\n";
const char *TRAIN_IMG_DESC =
      "Train image to display to user instead of the actual training image. This image can contain locations "
      "for the world points. For use where TangoCamera (-T) option is not specified so there is no way to "
      "project the world points to the training image.";
const char *CREATE_MATCHES_DESC =
      "-M <arg>, \t--creatematches <arg>\tCreate a file containing pixel correspondences after selecting "
            "correspondences in the alternating image selection GUI when using -p (--3dpt). <arg> is the "
            "name of the created file.";
const char *BENCHMARK_NOISE_DESC =
      "\t --noisebench <arg>     \t  Run Noise benchmark for non-synthesised data read from match files "
            "<arg> = start-deviation,increment-deviation,end-deviation";
const char *MATCH_FILE_DESC =
      "-m <arg>, \t --matchfile <arg>     \t <arg> = 3D to 2D match file each line mapping 3d world points to 2d image points"
      " in the query image ie x,y,z -> u,v\n";

enum OptionIndex { UNKNOWN, HELP, CALIBRATION, REMAP_GRAVITY, SYNTHESIZED, SYNTHESIZED_NOISE, POINT, TANGOCAM, CREATE_MATCHES,
                   MATCH_FILE, TRAIN_IMG, BENCHMARK_NOISE, SILENT };
const option::Descriptor usage[] =
      {
            { UNKNOWN,              0, "", "",             Arg::Unknown, USAGE_DESC},
            { HELP,                 0, "h", "help",        Arg::None,    "-h or --help  \tPrint usage and exit." },
            { CALIBRATION,          0, "c","calibration",  Arg::Required, CALIBRATION_DESC },
            { REMAP_GRAVITY,        0, "r","remap",        Arg::Required, REMAP_GRAVITY_DESC },
            { SYNTHESIZED,          0, "s","synth",        Arg::None, "-s or --synth    \tRun test on synthesized data." },
            { SYNTHESIZED_NOISE,    0, "y","synth-noise",  Arg::None, "-y or --synth-noise \tRun test on noisy synthesized data." },
            { POINT,                0, "p","3dpt",         Arg::Required, POINT_DESC },
            { TANGOCAM,             0, "T","tango",        Arg::None, TANGOCAM_DESC },
            { CREATE_MATCHES,       0, "M","creatematches",Arg::Required, CREATE_MATCHES_DESC },
            { MATCH_FILE,           0, "m","matchfile",    Arg::Required, MATCH_FILE_DESC },
            { TRAIN_IMG,            0, "t","train-img",    Arg::Required, TRAIN_IMG_DESC },
            { BENCHMARK_NOISE,      0, "", "noisebench", Arg::Required, BENCHMARK_NOISE_DESC },
            { SILENT,               0, "S","silent",       Arg::None,    "-S or --silent  \tDon't display image matches" },
            { 0, 0, 0, 0, 0, 0 },
      };

void help() { option::printUsage(std::cout, usage); }

bool parse_file(const char *filepath, std::vector<std::string>& tokens)
//----------------------------------------------------------------
{
   std::ifstream f(filepath);
   if (! f.good())
      return false;
   std::string content( (std::istreambuf_iterator<char>(f)), (std::istreambuf_iterator<char>()));
   auto p = content.find("gravity:");
   if (p == std::string::npos)
   {
      content.erase(std::remove (content.begin(), content.end(), ' '), content.end()); // remove spaces
      return (split(trim(content, "\t\n "), tokens, ",") == 3); // assumed txt file
   }
   p = content.find("data:", p);
   if (p == std::string::npos)
      return false;
   p +=5;
   auto pp = content.find("[", p);
   if (pp == std::string::npos)
      pp = p;
   else
      pp++;
   auto ppp = content.find("]", pp);
   if (ppp == std::string::npos)
      ppp = content.find("\n", pp);
   std::string s;
   if (ppp == std::string::npos)
      s = content.substr(pp);
   else
      s = content.substr(pp, ppp-pp);
   s.erase(std::remove (s.begin(), s.end(), ' '), s.end());
   return (split(trim(s, "\t\n "), tokens, ",") == 3);
}

bool read_input_image(const char *imagename, const char *gravity, cv::Mat& img, Eigen::Vector3d& gravity_vec,
                      std::string& name, std::stringstream& errs)
//-------------------------------------------------------------------------------------------
{
   img = cv::imread(imagename);
   if (img.empty())
   {
      errs << "ERROR: Reading image " << imagename;
      return false;
   }

   filesystem::path f(imagename);
   std::string ext = f.extension().string();
   name = f.filename().string();
   auto p = name.find_last_of(".");
   if (p != std::string::npos)
      name = std::string(name, 0, p);

   std::vector<std::string> tokens;
   if (split(gravity, tokens, ",") != 3)
   {
      if (! parse_file(gravity, tokens))
      {
         errs << "Gravity vector (2nd non switch option) must be in the form x,y,z where x, y and z are numbers"
               " or a text file containing x,y,z or a .yaml file with a gravity: data: x,y,z. "
               "eg 9.6019897,0.8189485,1.8170115";
         return false;
      }
   }
   gravity_vec = Eigen::Vector3d(std::stod(trim(tokens[0])), std::stod(trim(tokens[1])), std::stod(trim(tokens[2])));
   return true;
}

inline Eigen::Quaterniond yaml_quaternion(const YAML::Node node)
//--------------------------------------------------------------
{
   double w = node[0].as<double>();
   double x = node[1].as<double>();
   double y = node[2].as<double>();
   double z = node[3].as<double>();
   return Eigen::Quaterniond(w, x, y, z);
}

inline Eigen::Vector3d yaml_vector(const YAML::Node node)
//-------------------------------------------------------
{
   Eigen::Vector3d v;
   v[0] = node[0].as<double>();
   v[1] = node[1].as<double>();
   v[2] = node[2].as<double>();
   return v;
}

bool read_yaml(const char *filename, cv::Mat& img, Eigen::Vector3d& gravity_vec, cv::Mat* K,
               int& device_rotation, Eigen::Quaterniond& rotation, Eigen::Vector3d& translation,
               Eigen::Quaterniond& camera_IMU_rotation, Eigen::Vector3d& camera_IMU_translation,
               Eigen::Quaterniond& depth_IMU_rotation, Eigen::Vector3d& depth_IMU_translation,
               std::string& name, std::stringstream& errs)
//-------------------------------------------------------------------------------------------
{
   filesystem::path f(filename);
   std::string ext = f.extension();
   std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
   ext = trim(ext);
   if (ext != ".yaml")
   {
      errs << filename << " not a yaml file";
      return false;
   }
   name = f.filename();
   std::string dir(filesystem::absolute(f).parent_path().string());
   auto p = name.find_last_of(".");
   if (p != std::string::npos)
      name = std::string(name, 0, p);
   filesystem::path image(dir + "/" + name + ".jpg");
   if (filesystem::exists(image))
      img = cv::imread(image.string().c_str());
   else
   {
      image = filesystem::path(dir + "/" + name + ".png");
      if (filesystem::exists(image))
         img = cv::imread(image.string().c_str());
      else
      {
         errs << name << ".jpg, .png" << " not found in " << dir;
         return false;
      }
   }
   if (img.empty())
   {
      errs << "ERROR: Reading image " << image;
      return false;
   }
   filesystem::path yaml(dir + "/" + name + ".yaml");
   if (! filesystem::exists(yaml))
   {
      errs << yaml.filename() << " not found in " << dir;
      return false;
   }
   YAML::Node config = YAML::LoadFile(yaml.string());
   if (K != nullptr)
   {
      *K = cv::Mat::eye(3, 3, CV_64FC1);
      double fx = config["fx"].as<double>();
      double fy = config["fy"].as<double>();
      double cx = config["cx"].as<double>();
      double cy = config["cy"].as<double>();
      K->at<double>(0, 0) = fx;
      K->at<double>(1, 1) = fy;
      K->at<double>(0, 2) = cx;
      K->at<double>(1, 2) = cy;
   }
   device_rotation = config["deviceRotation"].as<int>();

   rotation = yaml_quaternion(config["rotation"]);
   translation = yaml_vector(config["translation"]);
   gravity_vec = yaml_vector(config["gravity"]);
   camera_IMU_rotation = yaml_quaternion(config["imuRotation"]);
   camera_IMU_translation = yaml_vector(config["imuTranslation"]);
   depth_IMU_rotation = yaml_quaternion(config["d_imuRotation"]);
   depth_IMU_translation  = yaml_vector(config["d_imuTranslation"]);
   return true;
}

bool process_calibration_arg(std::string calibration_values, cv::Mat& intrinsics)
//---------------------------------------------------------------------------------------
{
   filesystem::path f(calibration_values);
   if (filesystem::exists(f))
   {
      std::ifstream file;
      file.open(calibration_values);
      if (file)
      {
         std::string content((std::istreambuf_iterator<char>(file)),
                             (std::istreambuf_iterator<char>()));
         content.erase(std::remove(content.begin(), content.end(), ' '), content.end());
         content.erase(std::remove(content.begin(), content.end(), '\t'), content.end());
         content.erase(std::remove(content.begin(), content.end(), '\n'), content.end());
         calibration_values = content;
      }
   }

   intrinsics = cv::Mat::eye(3, 3, CV_64FC1);
   std::vector<std::string> tokens;
   auto calib_opts = split(calibration_values, tokens, ",");
   try
   {
      if (calib_opts > 0)
      {
         double v = std::stod(trim(tokens[0]));
         intrinsics.at<double>(0, 0) = v;
         if (calib_opts > 1)
         {
            v = std::stod(trim(tokens[1]));
            intrinsics.at<double>(1, 1) = v;
         }
         if (calib_opts > 2)
         {
            v = std::stod(trim(tokens[2]));
            intrinsics.at<double>(0, 2) = v;
         }
         if (calib_opts > 3)
         {
            v = std::stod(trim(tokens[3]));
            intrinsics.at<double>(1, 2) = v;
         }
         if (calib_opts != 4)
            std::cerr << "WARNING: Incomplete Calibration Matrix." << std::endl;
         std::cout << "Calibration Matrix" << std::endl << intrinsics << std::endl;
      }
      else
      {
         std::cerr << "ERROR: Invalid calibration parameter " << calibration_values << std::endl;
         exit(1);
      }
   }
   catch (...)
   {
      std::cerr << "Error parsing calibration values " << calibration_values << std::endl;
      exit(1);
   }
   return true;
}

inline void point_vals(std::vector<std::string>& tokens, double& x, double& y, double& z)
//-------------------------------------------------------------------------------------
{
   x = y = z = mut::NaN;
   try
   {
      x = std::stod(trim(tokens[0]));
      y = std::stod(trim(tokens[1]));
      z = std::stod(trim(tokens[2]));
   }
   catch (...)
   {
      x = y = z = mut::NaN;
   }
   if (isnan(x) || isnan(y) || isnan(z))
   {
      std::cerr << "Non-numeric coordinate ( " << tokens[0] << ", " << tokens[1]  << ", " << tokens[2]
                << ")" << std::endl;
      exit(1);
   }
}

bool process_point_arg(const char *arg, std::vector<cv::Point3d>& pts)
//-------------------------------------------------------------------
{
   std::vector<std::string> tokens;
   double x, y, z;
   pts.clear();
   auto no = split(arg, tokens, ",");
   if (no != 3)
   {
      std::ifstream f(arg);
      if (! f)
      {
         std::cerr << "Invalid argument " << arg << " to -p option." << std::endl;
         std::exit(1);
      }
      char buf[120];
      f.getline(buf, 120);
      while (! f.eof())
      {
         std::string line = trim(buf);
         if ( (line.empty()) || (line[0] == '#') )
         {
            f.getline(buf, 120);
            continue;
         }
         no = split(line, tokens, ",");
         if (no != 3)
         {
            std::cerr << "Invalid argument " << line << " to -p option." << std::endl;
            std::exit(1);
         }
         point_vals(tokens, x, y, z);
         pts.emplace_back(x, y, z);
         f.getline(buf, 120);
      }
   }
   else
   {
      point_vals(tokens, x, y, z);
      pts.emplace_back(x, y, z);
   }
   return true;
}

#ifdef USE_THEIA_RANSAC
static theia::RansacParameters RANSAC_parameter;
#else
static templransac::RANSACParams RANSAC_parameter(9);
#endif
inline void* RANSAC_params(double error_threshold, double sample_inlier_probability = 0.99)
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

inline size_t match_combinations(const std::vector<std::pair<cv::Point3d, cv::Point2d>>& v, int r,
                                 std::vector<std::vector<std::pair<cv::Point3d, cv::Point2d>>>& combinations)
//------------------------------------------------------------------------------------------------------------------------------
{
   int n = static_cast<int>(v.size());
   if (n == 0) return 0;
   std::vector<bool> b(n);
   std::fill(b.begin(), b.begin() + r, true);

   do
   {
      std::vector<std::pair<cv::Point3d, cv::Point2d>> combination;
      for (int i = 0; i < n; ++i)
      {
         if (b[i])
            combination.push_back(v[i]);
      }
      combinations.push_back(combination);
   } while (std::prev_permutation(b.begin(), b.end()));
   return combinations.size();
}

int main(int argc, char **argv)
//-----------------------------
{
   argc -= (argc > 0);
   argv += (argc > 0); // skip program name argv[0] if present
   option::Stats stats(usage, argc, argv);

   std::vector<option::Option> options(stats.options_max);
   std::vector<option::Option> buffer(stats.buffer_max);
   option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);

//   option::Parser parse(usage, argc, argv, options, buffer);
   if (parse.error())
      return 1;
   if ((options[HELP]) || (argc == 0) )
   {
      help();
      return 0;
   }

   cv::Mat train_img, query_img, intrinsics;
   Eigen::Quaterniond tango_train_rotation, tango_query_rotation, tango_camera_IMU_rotation, tango_depth_IMU_rotation;
   Eigen::Vector3d train_g, query_g, translation, tango_train_translation, tango_query_translation,
                   tango_camera_IMU_translation, tango_depth_IMU_translation;
   int device_rotation = 0;
   double angle;
   std::string basename;
   std::stringstream errs;
   if (parse.nonOptionsCount() == 4)
   {
      if (! read_input_image(parse.nonOption(0), parse.nonOption(1), train_img, train_g,
                             basename, errs))
      {
         std::cerr << "Error reading training image/gravity vector: " << errs.str() << ". Args: "
                   << parse.nonOption(0) <<  parse.nonOption(1) << std::endl;
         exit(1);
      }
      if (! read_input_image(parse.nonOption(2), parse.nonOption(3), query_img, query_g,
                             basename, errs))
      {
         std::cerr << "Error reading query image/gravity vector: " << errs.str()  << ". Args: "
                   << parse.nonOption(0) <<  parse.nonOption(1) << std::endl;
         exit(1);
      }
   }
   else if (parse.nonOptionsCount() == 2)
   {
      if (! read_yaml(parse.nonOption(0), train_img, train_g, &intrinsics, device_rotation,
                      tango_train_rotation, tango_train_translation,
                      tango_camera_IMU_rotation, tango_camera_IMU_translation,
                      tango_depth_IMU_rotation, tango_depth_IMU_translation,  basename, errs))
      {
         std::cerr << "Error reading training data: " << errs.str() << std::endl;
         exit(1);
      }
      int query_dev_rotation;
      if (! read_yaml(parse.nonOption(1), query_img, query_g, nullptr, query_dev_rotation, tango_query_rotation,
                      tango_query_translation, tango_camera_IMU_rotation, tango_camera_IMU_translation,
                      tango_depth_IMU_rotation, tango_depth_IMU_translation, basename, errs))
      {
         std::cerr << "Error reading query data: " << errs.str() << std::endl;
         exit(1);
      }
   }
   else if ( (! options[SYNTHESIZED]) && (! options[SYNTHESIZED_NOISE]) )
   {
      help();
      exit(0);
   }

   if (options[CALIBRATION])
   {
      if (! intrinsics.empty())
         std::cerr << "WARNING: Overriding calibration values from YAML file." << std::endl;
      if (! process_calibration_arg(options[CALIBRATION].arg, intrinsics))
         exit(1);
   }

   std::vector<cv::Point3d> world_pts, image_pts, transformed_image_pts;
   Eigen::Quaterniond Q;
   Eigen::Vector3d T;
   double roll, pitch, yaw;
   if ( (options[SYNTHESIZED]) || (options[SYNTHESIZED_NOISE]) )
   {
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> gravity_vectors;
      std::vector<Eigen::Matrix3d> rotation_matrices;
      std::vector<Eigen::Vector3d> translations;
      synth3d::create_test_data(gravity_vectors, rotation_matrices, translations);
      if (options[SYNTHESIZED])
         synth3d::synthesize(gravity_vectors, rotation_matrices, translations);
      else
         synth3d::synthesize_noisy(gravity_vectors, rotation_matrices, translations, 0.5, 5, 0.5);
      return 0;
   }
   else
   {
      cv::Mat show_train_image, show_query_image;
      if (options[MATCH_FILE])
      {
         query_img.copyTo(show_query_image);
         read_matches(options[MATCH_FILE].arg, world_pts, transformed_image_pts, nullptr,
                     &show_query_image, false, true);
      }
      else if (options[POINT].count() > 0)
      {
         auto t = tango_camera_IMU_translation - tango_depth_IMU_translation;
         int i = 1, ypos = -1;
         if (options[TANGOCAM])
            train_img.copyTo(show_train_image);
         for (option::Option *opt = options[POINT]; opt; opt = opt->next())
         {
            std::vector<cv::Point3d> xyzs;
            process_point_arg(opt->arg, xyzs);
            for (cv::Point3d pt : xyzs)
            {
               double x, y, z;
               x = pt.x; y = pt.y; z = pt.z;
               if (options[TANGOCAM])
               {
                  x += t[0];
                  y += t[1];
                  z += t[2];
               }
               world_pts.emplace_back(x, y, z);
               if (options[TANGOCAM])
               {
                  cv::Vec3d v = mut::ocvMultMatVec(intrinsics, x, y, z);
                  v /= v[2];
                  image_pts.emplace_back(v[0], v[1], v[2]);
                  int qx = cvRound(v[0]), qy = cvRound(v[1]);
                  int xp = qx - 1, yp = qy - 6;
                  std::stringstream ss;
                  ss << i++ << " (" << x << ", " << y << ", " << z << ")";
                  if ( (ypos >= 0) && (std::abs(ypos-yp) < 5) )
                  {
                     yp -= 15;
                     xp -= 5;
                  }
                  ypos = yp;
                  cvLabel(show_train_image, ss.str(), cv::Point2i(std::max(xp, 0), std::max(yp, 0)));
                  cv::circle(show_train_image, cv::Point2i(cvRound(qx), cvRound(qy)), 2, cv::Scalar(0, 0, 0), 1);
                  cv::circle(show_train_image, cv::Point2i(cvRound(qx), cvRound(qy)), 3, cv::Scalar(255, 255, 255), 1);
                  cv::circle(show_train_image, cv::Point2i(cvRound(qx), cvRound(qy)), 4, cv::Scalar(0, 0, 255), 1);
                  cv::circle(show_train_image, cv::Point2i(cvRound(qx), cvRound(qy)), 5, cv::Scalar(0, 255, 255), 1);
                  cv::circle(show_train_image, cv::Point2i(cvRound(qx), cvRound(qy)), 6, cv::Scalar(0, 0, 255), 1);
               }
            }
         }

         cv::Mat training_img;
         if (options[TRAIN_IMG])
         {
            training_img = cv::imread(options[TRAIN_IMG].arg);
            if (training_img.empty())
            {
               std::cerr << "Error reading " << options[TRAIN_IMG].arg << std::endl;
               std::exit(1);
            }
         }
         else
            training_img = train_img;
         SelectPoints selector("Select Corresponding Points", training_img, query_img, &image_pts);
         auto no = selector.select(options[TRAIN_IMG] != nullptr);
         if (no < 3)
         {
            std::cerr << "More than " << no << " points required." << std::endl;
            exit(1);
         }
         transformed_image_pts = selector.right_points3d(1);
         std::ofstream ofs;
         if (options[CREATE_MATCHES])
         {
            ofs.open(options[CREATE_MATCHES].arg);
            if (ofs.bad())
            {
               std::cerr << "Error creating point match output file " << options[CREATE_MATCHES].arg
                         << " (" << std::strerror(errno) << "). Sending output to stdout." << std::endl;

            }
            for (size_t j=0; j<std::min(world_pts.size(), transformed_image_pts.size()); j++)
            {
               auto wpt = world_pts[j];
               auto ipt = transformed_image_pts[j];
               if (ofs)
               {
                  ofs << wpt.x << "," << wpt.y << "," << wpt.z << " -> " << ipt.x << "," << ipt.y << std::endl;
                  ofs.flush();
               }
               else
                  std::cout << wpt.x << "," << wpt.y << "," << wpt.z << " -> " << ipt.x << "," << ipt.y << std::endl;
            }
         }
         int j = 1;
         query_img.copyTo(show_query_image);
         for (cv::Point3d pt : transformed_image_pts)
         {
            double x = pt.x, y = pt.y;
            std::stringstream ss;
            ss << j++;
            cvLabel(show_query_image, ss.str(), cv::Point2i(cvRound(x-1), cvRound(y-6)));
            plot_circles(show_query_image, x, y);
         }
      }
      if (! options[SILENT])
      {
         if ( (show_train_image.empty()) && (options[TRAIN_IMG]) )
            show_train_image = cv::imread(options[TRAIN_IMG].arg);
         int n = show_images("Training Image - Enter to continue, Esc to Abort", show_train_image, 0,
                             "Query Image - Enter to continue, Esc to Abort", show_query_image, 150);
         if ( (n > 0) && (cv::waitKey(0) == 27) )
            std::exit(1);
      }

      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double *) intrinsics.data);
      Eigen::Matrix3d KI = K.inverse();

      if (options[BENCHMARK_NOISE])
      {
         double start_deviation = mut::NaN, incr_deviation = mut::NaN, end_deviation = mut::NaN;
         std::vector<std::string> tokens;
         std::size_t deviation_cnt = split(options[BENCHMARK_NOISE].arg, tokens, ",");
         if (deviation_cnt != 3)
         {
            std::cerr << "--noisebench option must have 3 comma delimited parameters (start,increment,end) "
                      << options[BENCHMARK_NOISE].arg << std::endl;
            std::exit(1);
         }
         try
         {
            start_deviation = std::stod(trim(tokens[0]));
            incr_deviation = std::stod(trim(tokens[1]));
            end_deviation = std::stod(trim(tokens[2]));
         } catch (...)
         {
            start_deviation = incr_deviation = end_deviation = mut::NaN;
         }
         if (std::isnan(start_deviation))
         {
            std::cerr <<  "Error reading deviation for --noisebench (" << options[BENCHMARK_NOISE].arg << ")"
                      << std::endl;
            std::exit(1);
         }

         cv::Mat empty_img;
         int N = 100;
         std::ofstream gnuplot("gnuplot3d.data");
         for (double deviation=start_deviation; deviation<=end_deviation; deviation +=incr_deviation)
         {
            double mean_mean = 0, cv_mean_mean = 0, theia_mean_mean = 0;
            cv::Mat empty_img;
            for (int k=0; k<N; k++)
            {
               std::vector<cv::Point3d> noisy_image_pts;
               noise(deviation, transformed_image_pts, true, true, true, false, &noisy_image_pts);
               std::vector<cv::Point2d> image_points;
               std::transform(noisy_image_pts.cbegin(), noisy_image_pts.cend(), std::back_inserter(image_points),
                              [](const cv::Point3d& pt) -> cv::Point2d { return cv::Point2d(pt.x/pt.z, pt.y/pt.z); });
               std::vector<std::pair<cv::Point3d, cv::Point2d>> pts;
               for (size_t i=0; i<std::min(world_pts.size(), noisy_image_pts.size()); i++)
               {
                  const cv::Point3d& ipt = noisy_image_pts[i];
                  pts.emplace_back(std::make_pair(world_pts[i], cv::Point2d(ipt.x, ipt.y)));
               }

               pose3d::pose_ransac(pts, train_g, query_g, KI, Q, T, RANSAC_params(10), 3);
               Eigen::Matrix3d R = Q.toRotationMatrix();
               double max_error, mean_error, stddev_error;
               project3d(K, R, T, pts, max_error, mean_error);
               mean_mean += mean_error;

               cv::Mat OR, rotation_vec, translations3x1;
               long time;
               if (poseocv::pose_PnP(world_pts, noisy_image_pts, intrinsics, rotation_vec, translations3x1, &OR, time,
                                     false, cv::SOLVEPNP_EPNP))
               {
                  cv::cv2eigen(OR, R);
                  cv::cv2eigen(translations3x1, T);
                  project3d(K, R, T, world_pts, noisy_image_pts, max_error, mean_error);
               }
               else
               {
                  std::cerr << "OpenCV solvePnP failed" << std::endl;
                  std::exit(1);
               }
               cv_mean_mean += mean_error;
            }
            double m1 = mean_mean/N, m2 = cv_mean_mean/N, m3 = theia_mean_mean/N;
            std::cout << "Deviation " << deviation << ": GravPose = " << m1 << ", OpenCV Iterative = "
                      << m2 << ", Theia = " << m3 << std::endl;
            gnuplot << deviation << "\t" << m1 << "\t" << m2 << "\t" << m3 << std::endl;
         }
         std::exit(0);
      }

      for (int i=0; i<argc; i++)
         std::cout << argv[i] << " ";
      std::cout << std::endl << "============================================================================" << std::endl;
//      train_g << 0, 9.80665, 0;
      std::cout << train_g.transpose() << " to " << query_g.transpose() << std::endl;
//      display_pts3d(world_pts, image_pts, transformed_image_pts);
      std::vector<cv::Point2d> image_points;
      std::transform(transformed_image_pts.cbegin(), transformed_image_pts.cend(), std::back_inserter(image_points),
                     [](const cv::Point3d& pt) -> cv::Point2d { return cv::Point2d(pt.x/pt.z, pt.y/pt.z); });
      std::vector<std::pair<cv::Point3d, cv::Point2d>> point_matches, point_matches_3, point_matches_nonplanar;
      std::vector<double> z_coords;
      if (world_pts.size() > 3)
      {
         for (size_t i=0; i<std::min(world_pts.size(), image_points.size()); i++)
         {
            point_matches.emplace_back(std::make_pair(world_pts[i], image_points[i]));
            double z = world_pts[i].z;
            auto it = std::find_if(z_coords.begin(), z_coords.end(),
                                   [z](const double& zv){ return mut::near_zero(z - zv, 0.0001);});
            if (it == z_coords.end())
            {
               z_coords.emplace_back(z);
               point_matches_nonplanar.emplace_back(std::make_pair(world_pts[i], image_points[i]));
            }
         }
         std::vector<std::vector<std::pair<cv::Point3d, cv::Point2d>>> combinations;
         match_combinations(point_matches, 3, combinations);
         size_t index = combinations.size();
         double min_mean_error = 9999999;
         for (size_t j=0; j<combinations.size(); j++)
         {
            const std::vector<std::pair<cv::Point3d, cv::Point2d>>& cpts = combinations[j];
            pose3d::pose(cpts, train_g, query_g, KI, Q, T);
            double maxError, meanError;
            Eigen::Matrix3d R = Q.toRotationMatrix();
            project3d(K, R, T, point_matches, maxError, meanError);
            if (meanError < min_mean_error)
            {
               min_mean_error = meanError;
               index = j;
            }
         }
         if (index < combinations.size())
            point_matches_3 = std::move(combinations[index]);
         else
            point_matches_3 = std::vector<std::pair<cv::Point3d, cv::Point2d>>(&point_matches[0],&point_matches[3]);
      }
      else
      {
         std::transform(transformed_image_pts.cbegin(), transformed_image_pts.cend(), std::back_inserter(image_points),
                        [](const cv::Point3d &pt) -> cv::Point2d
                        { return cv::Point2d(pt.x / pt.z, pt.y / pt.z); });
         for (size_t i=0; i<std::min(world_pts.size(), image_points.size()); i++)
         {
            point_matches.emplace_back(std::make_pair(world_pts[i], image_points[i]));
            double z = world_pts[i].z;
            auto it = std::find_if(z_coords.begin(), z_coords.end(),
                                   [z](const double& zv){ return mut::near_zero(z - zv, 0.0001);});
            if (it == z_coords.end())
            {
               z_coords.emplace_back(z);
               point_matches_nonplanar.emplace_back(std::make_pair(world_pts[i], image_points[i]));
            }
         }
         point_matches_3 = std::vector<std::pair<cv::Point3d, cv::Point2d>>(&point_matches[0],&point_matches[3]);
      }

      pose3d::pose(point_matches_3, train_g, query_g, KI, Q, T);
      display_rotation(Q, false);
      std::cout << std::fixed << std::setprecision(4) << "Translation: [" << T[0]  << "," << T[1] << "," << T[2]
                << "]" << std::endl;
      double max_error, mean_error, stddev_error;
      Eigen::Matrix3d R = Q.toRotationMatrix();
      cv::Mat img;
      query_img.copyTo(img);
      project3d(K, R, T, point_matches, max_error, mean_error, &img);
      std::cout << "Max Error " << max_error << ", mean error " << mean_error
                << "\n---------------------------------------------------------\n";
      cv::imwrite("reproject.png", img);

      Eigen::Vector3d rT(T);
      int iterations;
      pose3d::refine(world_pts, image_points, K, Q, rT, iterations);
//      R = Q.toRotationMatrix();
      std::cout << "Refined: iterations " << iterations << " ";
      std::cout << std::fixed << std::setprecision(4) << "Translation: [" << rT.transpose() << "]" << std::endl;
      project3d(K, R, rT, point_matches, max_error, mean_error);
      std::cout << "Max Error " << max_error << ", mean error " << mean_error
                << "\n-----------------------------------------------------\n";

      std::cout << "OpenCV solvePnP: " << std::endl;
      poseocv::display_PnP(world_pts, transformed_image_pts, intrinsics, true, &query_img, false, true);
   }
}

