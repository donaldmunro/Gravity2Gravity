#include <cstdlib>
#include <iomanip>
#include <cerrno>
#include <cstring>

#include <iostream>
#include <cstdarg>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <memory>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/eigen.hpp>

#include <yaml-cpp/yaml.h>

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

#include "optionparser.h"
#include "math.hh"
#include "common.h"
#include "SelectMatchedPoints.h"
#define ANDROIDEM_OPENCV 1
#define ANDROIDEM_EIGEN 1
#include "androidem.hh"
#include "pose2d.h"
#include "poseocv.h"
#include "pose-other.h"
#include "Optimization.h"
#include "Ransac.hh"
#include "synth2d.h"

#define MAIN 1

void help();

struct Arg : public option::Arg
//=============================
{
   static void printError(const char *msg1, const option::Option &opt, const char *msg2)
   //-----------------------------------------------------------------------------------
   {
      fprintf(stderr, "%s", msg1);
      fwrite(opt.name, (std::size_t) opt.namelen, 1, stderr);
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
      "USAGE: planargrav [options] train-image train-gravity-vector query-image query-gravity-vector\n"
      "       where gravity-vector is 3 numbers separated by commas without spaces (unless surrounded by quotes)\n"
      "       eg 9.6019897,0.8189485,1.8170115\n"
      "OR     planargrav -s (--synth) runs synthesized data test\n"
      "OR     planargrav [options] train-yaml-file query-yaml-file uses images with the same basename as the "
      "       yaml files with yaml files from/similar to those created by TangoCamera (https://github.com/donaldmunro/TangoCamera)\n"
      "In the first two cases feature points to match are selected by displaying the two images side by side and "
      "selecting points alternately in each image unless -m (--matchfile) or -b (--blobs) is specified. If -m "
      "is specified then a file with matching pixel locations is used. If -b is specified then circular blobs "
      "of different sizes are used to specify feature points to match. If neither -m or -b is specified"
      "then the -M (--creatematches) option can be used to create a matchfile from the alternating selection "
      "GUI option for subsequent use. The match file can also be created manually from programs such as"
      "gimp. The format is multiple lines of x,y -> u,v (or x,y = u,v) where x and y are pixel locations in the training image "
      "and u,y are corresponding pixel locations in the query image\n";
const char *CALIBRATION_DESC =
      "-c <arg>, \t--calibration=<arg>\tSpecify camera calibration details. <arg> format fx,fy,ox,oy where:\n"
            "fx = Focal length (X), fy = Focal length (Y), ox = X offset, oy = Y offset.\n"
            "Example -c 776.728236,776.0521739,370.32,300.88\n"
            "Alternately a path to a file containing the calibration parameters in the same comma delimited format\n"
            "(without the -c) may be specified\n";
const char *REMAP_GRAVITY_DESC =
"-m <arg>, \t--remap <arg>\tPerform Android remap coordinate system on gravity vector (when the vector was recorded while in "
"one of the landscape modes or reverse portrait mode). <arg> is the device rotation (int) where:"
" 0 =   device default (portrait for phones and many but not all tablets.\n"
" 90 =  90 degree rotation (landscape for devices with default of portrait)"
" 180 = 180 degree rotation (reversed portrait for devices with default of portrait)"
" 270 = 270 degree rotation (reversed landscape for devices with default of portrait)";
const char *DEPTH_DESC =
      "-d <arg>, \t--depth <arg>\tThe depth (z component) of the objects plane for planar relative pose.";
const char *BLOB_DESC =
      "-b <arg>, \t--blobs <arg>\tThe test images contain circular blobs of different radii denoting"
      "the feature points that are matched. The circular blobs must be of distinctive radii with matches"
      "related by similar radii in the separate images. <arg> is the expected number of blobs.";
const char *MATCHFILE_DESC =
      "-m <arg>, \t--matchfile <arg>\tA file containing pixel correspondences in the form of multiple "
      "lines of x,y -> u,v where x and y are pixel locations in the training image and u,y are corresponding "
      "pixel locations in the query image\n";
const char *CREATE_MATCHES_DESC =
      "-M <arg>, \t--creatematches <arg>\tCreate a file containing pixel correspondences after selecting "
      "correspondences in the alternating image selection GUI. <arg> is the name of the created file. "
      "If <arg> is an existing file then the matches in it will be copied and new correspondences added "
      "before overwriting <arg>.";
const char *SEL_MATCHES_USING_DESC =
     "-T <arg>, \t --tmatches <arg>   \t Read left (training) training image correspondences from match file. Selection only "
     "need to be done on the right hand query image in the selection GUI";
const char *MATCH_3D_DESC =
      "-3 <arg>, \t --3d <arg>     \t <arg> = 3D to 2D match file each line mapping 3d world points to 2d image points "
      "in the query image ie x,y,z -> u,v\n";
const char *ADD_NOISE_DESC =
      "-N <arg>, \t --noise <arg>     \t Add noise for non-synthesised data read from match files <arg> = noise deviation";
const char *BENCHMARK_NOISE_DESC =
      "\t --noisebench <arg>     \t  Run Noise benchmark for non-synthesised data read from match files "
            "<arg> = start-deviation,increment-deviation,end-deviation";

enum OptionIndex { UNKNOWN, HELP, CALIBRATION, REMAP_GRAVITY, SYNTHESIZED, SYNTHESIZED_NOISE, DEPTH, BLOBS, MATCHFILE,
                   CREATE_MATCHES, SEL_MATCHES_USING, MATCH_3D, ADD_NOISE, BENCHMARK_NOISE, SILENT};
const option::Descriptor usage[] =
      {
            { UNKNOWN,              0, "", "",              Arg::Unknown, USAGE_DESC},
            { HELP,                 0, "h", "help",         Arg::None,    "-h or --help  \tPrint usage and exit." },
            { CALIBRATION,          0, "c","calibration",   Arg::Required, CALIBRATION_DESC },
            { REMAP_GRAVITY,        0, "r","remap",         Arg::Required, REMAP_GRAVITY_DESC },
            { SYNTHESIZED,          0, "s","synth",         Arg::None, "Synthesize and display" },
            { SYNTHESIZED_NOISE,    0, "S","noisy",         Arg::None, "Noisy synthesize and display" },
            { DEPTH,                0, "d","depth",         Arg::Required, DEPTH_DESC },
            { BLOBS,                0, "b","blobs",         Arg::Required, BLOB_DESC },
            { MATCHFILE,            0, "m","matchfile",     Arg::Required, MATCHFILE_DESC },
            { CREATE_MATCHES,       0, "M","creatematches", Arg::Required, CREATE_MATCHES_DESC },
            { SEL_MATCHES_USING,    0, "T","tmatches",      Arg::Required, SEL_MATCHES_USING_DESC },
            { MATCH_3D,             0, "3","3d",            Arg::Required, MATCH_3D_DESC },
            { ADD_NOISE,            0, "N","noise",         Arg::Required, ADD_NOISE_DESC },
            { BENCHMARK_NOISE,      0, "", "noisebench", Arg::Required, BENCHMARK_NOISE_DESC },
            { SILENT,               0, "q","quiet",         Arg::None,    "-q or --quiet  \tDon't display image matches" },
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
                      std::stringstream& errs)
//-------------------------------------------------------------------------------------------
{
   img = cv::imread(imagename);
   if (img.empty())
   {
      errs << "ERROR: Reading image " << imagename;
      return false;
   }
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

bool read_yaml(const char *filename, cv::Mat& img, Eigen::Vector3d& gravity_vec, cv::Mat* K,
               int& device_rotation, Eigen::Quaterniond& rotation, Eigen::Vector3d& translation,
               std::stringstream& errs)
//-------------------------------------------------------------------------------------------
{
   filesystem::path f(filename);
   if (! filesystem::exists(f))
   {
      errs << filesystem::absolute(f)  << " not found.";
      return false;
   }
   std::string ext = f.extension();
   std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
   ext = trim(ext);
   if (ext != ".yaml")
   {
      errs << filename << " not a yaml file";
      return false;
   }
   std::string name = f.filename().string(), dir = filesystem::absolute(f).parent_path().string();
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
   YAML::Node rotation_node = config["rotation"];
   double w = rotation_node[0].as<double>();
   double x = rotation_node[1].as<double>();
   double y = rotation_node[2].as<double>();
   double z = rotation_node[3].as<double>();
   rotation = Eigen::Quaterniond(w, x, y, z);
   YAML::Node translation_node = config["translation"];
   translation[0] = translation_node[0].as<double>();
   translation[1] = translation_node[1].as<double>();
   translation[2] = translation_node[2].as<double>();
   YAML::Node gravity_node = config["gravity"];
   gravity_vec[0] = gravity_node[0].as<double>();
   gravity_vec[1] = gravity_node[1].as<double>();
   gravity_vec[2] = gravity_node[2].as<double>();
   return true;
}

template <typename V> void colors(const cv::Mat& img, const cv::Point2f pt, int& r, int& g, int& b)
//-------------------------------------------------------------------------------------------------
{
   V tbgr = img.at<V>(cvRound(pt.x), cvRound(pt.y));
   b = tbgr[0];
   g = tbgr[1];
   r = tbgr[2];
}

bool equal_color(const cv::Mat& img1, const cv::Point2f& pt1, const cv::Mat& img2, const cv::Point2f& pt2,
                 int& r, int& g, int& b)
//---------------------------------------------------------------------------------------------------------
{
   assert(img1.channels() == img2.channels());
   int r2, g2, b2;
   switch (img1.channels())
   {
      case 3:
         colors<cv::Vec3f>(img1, pt1, r, g, b);
         colors<cv::Vec3f>(img2, pt2, r2, g2, b2);
         break;
      case 4:
         colors<cv::Vec4f>(img1, pt1, r, g, b);
         colors<cv::Vec4f>(img2, pt2, r2, g2, b2);
         break;
      default:
         g = b = r = static_cast<int>(img1.at<float>(cvRound(pt1.x), cvRound(pt1.y)));
         return (r == img2.at<float>(cvRound(pt2.x), cvRound(pt2.y)) );
   }
   return ( (r == r2) && (g == g2) && (b == b2) );
}

size_t match_blobs(const cv::Mat &train_image, const cv::Mat& query_image,
                   const cv::SimpleBlobDetector::Params& blob_params,
                   std::vector<cv::Point3d>& train_pts, std::vector<cv::Point3d>& query_pts,
                   cv::Mat& show_train_img, cv::Mat& show_query_img)
//-----------------------------------------------------------------------------------------------------
{
   cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(blob_params);
   std::vector<cv::KeyPoint> train_keypoints, query_keypoints;
   cv::Mat train_img, query_img;
   cv::cvtColor(train_image, train_img, CV_RGB2GRAY);
   cv::cvtColor(query_image, query_img, CV_RGB2GRAY);
   detector->detect(train_img, train_keypoints);
   detector->detect(query_img, query_keypoints);
   size_t ct = train_keypoints.size(), qt = query_keypoints.size();

   if (show_train_img.empty())
      train_image.copyTo(show_train_img);
   if (show_query_img.empty())
      query_image.copyTo(show_query_img);
   int j = 1;
   size_t matches = 0;
   for (cv::KeyPoint tkp : train_keypoints)
   {
      int r =0, g =0, b =0;
      cv::KeyPoint *pqkp = nullptr;
      std::size_t i=0;
      for (; i< query_keypoints.size(); i++)
      {
         cv::KeyPoint& qkp = query_keypoints[i];
         bool radius_match = (cvRound(std::abs(tkp.size - qkp.size)) <= 2);
         bool color_match = equal_color(train_image, tkp.pt, query_image, qkp.pt, r, g, b);
         if ( (radius_match) && (color_match) )
         {
            pqkp = &qkp;
            break;
         }

      }
      std::stringstream ss;
      if (pqkp != nullptr)
      {
         matches++;
         cv::Point2f pt = tkp.pt;
         train_pts.emplace_back(pt.x, pt.y, 1);
         ss << j;
         cvLabel(show_train_img, ss.str(), pt - cv::Point2f(1,4));
//         cv::putText(show_train_img, ss.str(), pt, CV_FONT_HERSHEY_DUPLEX, 0.8, color, 1, CV_AA);
         cv::Scalar color(255 - r, 255 - g, 255 - b);
         cv::circle(show_train_img, pt, cvRound(tkp.size), color, 3);
         cv::circle(show_train_img, pt, cvRound(tkp.size) + 1, cv::Scalar(255, 255, 255), 1);
         cv::circle(show_train_img, pt, cvRound(tkp.size) + 2, cv::Scalar(0, 0, 0), 1);
         cv::circle(show_train_img, pt, cvRound(tkp.size) + 3, cv::Scalar(0, 255, 255), 1);
         pt = pqkp->pt;
         query_pts.emplace_back(pt.x, pt.y, 1);
//         cv::putText(show_query_img, ss.str(), pt, CV_FONT_HERSHEY_DUPLEX, 0.8, cvScalar(255, 255, 255), 1, CV_AA);
         cvLabel(show_query_img, ss.str(), pt - cv::Point2f(1,4));
         cv::circle(show_query_img, pt, cvRound(pqkp->size), color, 1);
         cv::circle(show_query_img, pt, cvRound(tkp.size) + 1, cv::Scalar(255, 255, 255), 1);
         cv::circle(show_query_img, pt, cvRound(tkp.size) + 2, cv::Scalar(0, 0, 0), 1);
         cv::circle(show_query_img, pt, cvRound(tkp.size) + 3, cv::Scalar(0, 255, 255), 1);
         j++;
         query_keypoints.erase(query_keypoints.begin()+i);
      }
   }
   return matches;
}

void find_blobs(const char *cblobs, const cv::Mat& train_img, const cv::Mat& query_img,
                cv::Mat& show_train_img, cv::Mat& show_query_img,
                std::vector<cv::Point3d>& train_pts, std::vector<cv::Point3d>& query_pts)
//---------------------------------------------------------------------------------------
{
   std::size_t n;
   try { n = static_cast<std::size_t>(std::stol(trim(cblobs))); } catch (...) { n = 0; }
   cv::SimpleBlobDetector::Params blob_params;
   blob_params.minThreshold = 20;
   blob_params.maxThreshold = 260;
   blob_params.thresholdStep = 5;
   blob_params.filterByCircularity = true;
   blob_params.minCircularity = 0.8;
   blob_params.maxCircularity = std::numeric_limits<float>::max();
//         blob_params.filterByColor = true;
//         blob_params.blobColor = 0; // black blobs
   std::size_t no = match_blobs(train_img, query_img, blob_params, train_pts, query_pts,
                                show_train_img, show_query_img);
//         blob_params.blobColor = 255; // white blobs
//         n += match_blobs(train_img, query_img, blob_params, train_pts, query_pts,
//                          show_train_img, show_query_img);

   if (no < n)
   {
      std::cerr << "Number of matched keypoints in (" << no << ") != " << n << std::endl;
      cv::imwrite("train-error.jpg", show_train_img);
      cv::imwrite("query-error.jpg", show_query_img);
      exit(1);
   }
}

inline size_t match_combinations(const std::vector<std::pair<cv::Point3d, cv::Point3d>>& v, int r,
                                 std::vector<std::vector<std::pair<cv::Point3d, cv::Point3d>>>& combinations)
//------------------------------------------------------------------------------------------------------------------------------
{
   int n = static_cast<int>(v.size());
   if (n == 0) return 0;
   std::vector<bool> b(n);
   std::fill(b.begin(), b.begin() + r, true);

   do
   {
      std::vector<std::pair<cv::Point3d, cv::Point3d>> combination;
      for (int i = 0; i < n; ++i)
      {
         if (b[i])
            combination.push_back(v[i]);
      }
      combinations.push_back(combination);
   } while (std::prev_permutation(b.begin(), b.end()));
   return combinations.size();
}

inline void combine_matches(const std::vector<cv::Point3d>& train_pts, const std::vector<cv::Point3d>& query_pts,
                            double depth, const Eigen::Vector3d& train_g, const Eigen::Vector3d& query_g,
                            const Eigen::Matrix3d& K,
                            std::vector<std::pair<cv::Point3d, cv::Point3d>>& matched_pts,
                            std::vector<std::pair<cv::Point3d, cv::Point3d>>& matched_pts_3)
//---------------------------------------------------------------------------------------------------------------
{
   matched_pts.clear(); matched_pts_3.clear();
   for (size_t i=0; i<std::min(train_pts.size(), query_pts.size()); i++)
      matched_pts.emplace_back(std::make_pair(train_pts[i], query_pts[i]));
   if ( (! std::isnan(depth)) && (depth > 0) )
   {
      std::vector<std::vector<std::pair<cv::Point3d, cv::Point3d>>> combinations;
      match_combinations(matched_pts, 3, combinations);
      size_t index = combinations.size();
      double min_mean_error = 9999999;
      Eigen::Quaterniond Q;
      Eigen::Vector3d T;
      Eigen::Matrix3d KI = K.inverse();
      for (size_t j = 0; j < combinations.size(); j++)
      {
         const std::vector<std::pair<cv::Point3d, cv::Point3d>>& cpts = combinations[j];
         pose2d::pose(cpts, train_g, query_g, KI, Q, T);
         double maxError, meanError;
         Eigen::Matrix3d R = Q.toRotationMatrix();
         project2d(K, R, T, matched_pts, depth, maxError, meanError);
         if (meanError < min_mean_error)
         {
            min_mean_error = meanError;
            index = j;
         }
      }
      if (index < combinations.size())
         matched_pts_3 = std::move(combinations[index]);
      else
         matched_pts_3 = std::vector<std::pair<cv::Point3d, cv::Point3d>>(&matched_pts[0], &matched_pts[3]);
   }
   else
      matched_pts_3 = std::vector<std::pair<cv::Point3d, cv::Point3d>>(&matched_pts[0], &matched_pts[3]);
}

void show_reprojection(const cv::Mat& img, const cv::Mat& intrinsics, const std::vector<cv::Point3d>& train_pts,
                       const std::vector<cv::Point3d>& query_pts, const Eigen::Quaterniond& Q, const Eigen::Vector3d& T,
                       const double depth, const char *filename, double& max_error, double& mean_error,
                       bool draw_query =true)
//--------------------------------------------------------------------------------------------------------------------
{
   std::vector<std::pair<cv::Point3d, cv::Point3d>> matched_pts;
   for (size_t i=0; i<std::min(train_pts.size(), query_pts.size()); i++)
      matched_pts.emplace_back(std::make_pair(train_pts[i], query_pts[i]));
   cv::Mat project_image;
   img.copyTo(project_image);
   Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double *) intrinsics.data);
   Eigen::Matrix3d R = Q.toRotationMatrix();
   project2d(EK, R, T, matched_pts, depth, max_error, mean_error, &project_image, draw_query);
   cv::imwrite(filename, project_image);
}

#ifdef USE_THEIA_RANSAC
static theia::RansacParameters RANSAC_parameter;
#else
static templransac::RANSACParams RANSAC_parameter(9);
#endif
inline void* RANSAC_params(double error_threshold, double sample_inlier_probability = 0.99, int max_iter = std::numeric_limits<int>::max())
{
#ifdef USE_THEIA_RANSAC
    RANSAC_parameter.error_thresh = error_threshold;
    RANSAC_parameter.failure_probability = 1.0 - sample_inlier_probability;
   RANSAC_parameter.max_iterations = max_iter;
#else
   RANSAC_parameter.error_threshold = error_threshold;
   RANSAC_parameter.sample_inlier_probability = sample_inlier_probability;
#endif
   return static_cast<void*>(&RANSAC_parameter);
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
   std::vector<std::string> tokens;
   Eigen::Quaterniond tango_train_rotation, tango_query_rotation;
   Eigen::Vector3d train_g, query_g, translation, tango_train_translation, tango_query_translation;
   int device_rotation = 0;
   double depth = mut::NaN;
   double angle;

   std::stringstream errs;
   if (parse.nonOptionsCount() == 4)
   {
      if (! read_input_image(parse.nonOption(0), parse.nonOption(1), train_img, train_g,
                             errs))
      {
         std::cerr << "Error reading training image/gravity vector: " << errs.str() << std::endl;
         exit(1);
      }
      if (! read_input_image(parse.nonOption(2), parse.nonOption(3), query_img, query_g,
                             errs))
      {
         std::cerr << "Error reading query image/gravity vector: " << errs.str() << std::endl;
         exit(1);
      }
   }
   else if (parse.nonOptionsCount() == 2)
   {
      if (! read_yaml(parse.nonOption(0), train_img, train_g, &intrinsics, device_rotation,
                      tango_train_rotation, tango_train_translation, errs))
      {
         std::cerr << "Error reading training data: " << errs.str() << std::endl;
         exit(1);
      }
      int query_dev_rotation;
      if (! read_yaml(parse.nonOption(1), query_img, query_g, nullptr, query_dev_rotation, tango_query_rotation,
                      tango_query_translation, errs))
      {
         std::cerr << "Error reading query data: " << errs.str() << std::endl;
         exit(1);
      }
   }
   else if ( (! (options[SYNTHESIZED])) && (! (options[SYNTHESIZED_NOISE])) )
   {
      std::cerr << "Requires training and query images/names" << std::endl;
      help();
      exit(1);
   }

   if ( (options[CALIBRATION]) && (intrinsics.empty()) )
   {
      std::string calibration_values = options[CALIBRATION].arg;
      std::ifstream file;
      intrinsics = cv::Mat::eye(3, 3, CV_64FC1);
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
   }
   else
   {
      if (intrinsics.empty())
      {
         intrinsics = cv::Mat::eye(3, 3, CV_64FC1);
         std::cerr << "WARNING: Calibration matrix is Identity." << std::endl;
      }
   }
   Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> K((double *) intrinsics.data);
   std::vector<cv::Point3d> train_pts, query_pts, world_pts, image_pts;
   Eigen::Quaterniond Q;
   Eigen::Vector3d T;
   if (options[DEPTH])
   {
      try
      {
         depth = std::stod(trim(options[DEPTH].arg));
      }
      catch (...)
      {
         std::cerr << "Error parsing depth (not a number ?)" << options[DEPTH].arg << std::endl;
         exit(1);
      }
   }
   if ( (options[SYNTHESIZED]) || (options[SYNTHESIZED_NOISE]) )
   {
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> gravity_vectors;
      std::vector<Eigen::Matrix3d> rotation_matrices;
      std::vector<Eigen::Vector3d> translations;
      if ( (std::isnan(depth)) || (depth < 0) )
         depth = 10; //depth is maxdepth
      synth2d::create_test_data(gravity_vectors, rotation_matrices, translations);
      if (options[SYNTHESIZED])
         synth2d::synthesize(gravity_vectors, rotation_matrices, translations, depth);
      else
         synth2d::synthesize_noisy(gravity_vectors, rotation_matrices, translations, depth, 0.5, 5, 0.5);
      return 0;
   }
   else
   {
      cv::Mat show_train_img, show_query_img, show_image_img;
      if (options[MATCH_3D])
      {
         query_img.copyTo(show_image_img);
         read_matches(options[MATCH_3D].arg, world_pts, image_pts, nullptr, &show_image_img, true, true);
      }
      if (options[MATCHFILE])
      {
         train_img.copyTo(show_train_img);
         query_img.copyTo(show_query_img);
         read_matches(options[MATCHFILE].arg, train_pts, query_pts, &show_train_img, &show_query_img);
      }
      else if (options[BLOBS])
         find_blobs(options[BLOBS].arg, train_img, query_img, show_train_img, show_query_img, train_pts, query_pts);
      else if (! options[MATCH_3D])
      {
         std::vector<cv::Point3d> preselected_left_pts, preselected_right_pts;
         if (options[SEL_MATCHES_USING])
         {
            read_matches(options[SEL_MATCHES_USING].arg, preselected_left_pts, query_pts, &show_train_img, &show_query_img);
            query_pts.clear();
         }
         else if (options[CREATE_MATCHES])
         {
            std::ifstream ifs(options[CREATE_MATCHES].arg);
            if (ifs)
               read_matches(options[CREATE_MATCHES].arg, preselected_left_pts, preselected_right_pts,
                            &show_train_img, &show_query_img);
         }
         SelectPoints selector("Select Corresponding Points", train_img, query_img,
                               &preselected_left_pts, &preselected_right_pts);
         if ( (selector.select(false) < 3) || (selector.last_key() == 27) )
         {
            std::cerr << "Not enough points" << std::endl;
            exit(1);
         }
         train_pts = selector.left_points3d();
         query_pts = selector.right_points3d();
         if (options[CREATE_MATCHES])
         {
            std::string filename = options[CREATE_MATCHES].arg;
            std::ofstream ofs(filename);
            if (ofs)
            {
               for (size_t i=0; i<train_pts.size(); i++)
               {
                  const cv::Point3d& tpt = train_pts[i];
                  const cv::Point3d& qpt = query_pts[i];
                  ofs << std::fixed << std::setprecision(4) << tpt.x << "," << tpt.y << " -> "
                      << qpt.x << "," << qpt.y << std::endl;
               }
            }
            else
               std::cerr << "Could not create match file " << filename << " (" << std::strerror(errno)
                         << ")" << std::endl;
         }
      }
      if (! options[SILENT])
      {
         int n = show_images("Training Image - Enter to continue, Esc to Abort", show_train_img, 0,
                             "Query Image - Enter to continue, Esc to Abort", show_query_img, 150,
                             "PnP Image - Enter to continue, Esc to Abort", show_image_img, 300);
         if ( (n > 0) && (cv::waitKey(0) == 27) )
            std::exit(1);
      }
   }
   if (options[REMAP_GRAVITY])
   {
      try { device_rotation = std::stoi(trim(options[REMAP_GRAVITY].arg)); } catch (...) { device_rotation = -1; }
      if (device_rotation < 0)
      {
         std::cerr << "Could not parse device rotation (" << options[REMAP_GRAVITY].arg << ")" << std::endl;
         exit(1);
      }
      switch (device_rotation)
      {
         case 0: case 90: case 180: case 270:
            androidem::remapCoordinateSystemVector(train_g, device_rotation);
            androidem::remapCoordinateSystemVector(query_g, device_rotation);
            break;
         default: std::cerr << "Invalid device rotation (" << options[REMAP_GRAVITY].arg <<
                  ") Must be one of 0, 90, 180, 270" << std::endl;
            exit(1);
      }
   }
   for (int i=0; i<argc; i++)
      std::cout << argv[i] << " ";
   std::cout << std::endl << "============================================================================" << std::endl;
   std::cout << train_g.transpose() << " to " << query_g.transpose() << " depth " << depth << " ";
   std::vector<std::pair<cv::Point3d, cv::Point3d>> matched_pts, matched_points_3;
   if ( (!train_pts.empty()) && (! query_pts.empty()) )
   {
      if (options[ADD_NOISE])
      {
         double deviation = mut::NaN;
         try { deviation = std::stod(trim(options[ADD_NOISE].arg)); } catch (...) { deviation = mut::NaN; }
         if (std::isnan(deviation))
         {
            std::cerr <<  "Error reading deviation as argument for -N --noise (" << options[ADD_NOISE].arg << ")"
                      << std::endl;
            std::exit(1);
         }
         noise(deviation, query_pts, true);
         noise(deviation, image_pts, true);
      }
      if (options[BENCHMARK_NOISE])
      {
         double start_deviation = mut::NaN, incr_deviation = mut::NaN, end_deviation = mut::NaN;
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

         Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double *) intrinsics.data);
         cv::Mat empty_img;
         int N = 100;
         std::ofstream gnuplot("gnuplot.data");
         for (double deviation=start_deviation; deviation<=end_deviation; deviation +=incr_deviation)
         {
            double mean_mean = 0, cv_mean_mean = 0, theia_mean_mean = 0;
            for (int k=0; k<N; k++)
            {
               std::vector<cv::Point3d> noisy_query_pts, noisy_image_pts;
               noise(deviation, query_pts, true, true, true, false, &noisy_query_pts);
               noise(deviation, image_pts, true, true, true, false, &noisy_image_pts);
               combine_matches(train_pts, noisy_query_pts, depth, train_g, query_g, K, matched_pts, matched_points_3);
               if (std::isnan(depth))
                  pose2d::pose(matched_points_3, train_g, query_g, intrinsics, Q, T);
               else
                  pose2d::pose(matched_points_3, train_g, query_g, depth, intrinsics, Q, T);
               Eigen::Matrix3d R = Q.toRotationMatrix();
               double max_error, mean_error;
               long time;
               project2d(EK, R, T, matched_pts, depth, max_error, mean_error);
               mean_mean += mean_error;

               cv::Mat OR, rotation_vec, translations3x1;
               if (poseocv::pose_PnP(world_pts, noisy_image_pts, intrinsics, rotation_vec, translations3x1, &OR, time,
                                     false, cv::SOLVEPNP_EPNP))
               {
                  cv::cv2eigen(OR, R);
                  cv::cv2eigen(translations3x1, T);
                  double stddev_error;
                  project3d(EK, R, T, world_pts, noisy_image_pts, max_error, mean_error);
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
      combine_matches(train_pts, query_pts, depth, train_g, query_g, K, matched_pts, matched_points_3);
      if ( (std::isnan(depth)) || (depth == 0) )
         pose2d::pose(matched_points_3, train_g, query_g, intrinsics, Q, T);
      else
         pose2d::pose(matched_points_3, train_g, query_g, depth, intrinsics, Q, T);

      display_rotation(Q, false);
      std::cout << "Translation: " << T.transpose() << std::endl;
      double max_error, mean_error;
      show_reprojection(query_img, intrinsics, train_pts, query_pts, Q, T, depth, "reproject.jpg",
                        max_error, mean_error, ! std::isnan(depth));
      std::cout << "Max Error " << max_error << ", mean error " << mean_error << std::endl;
      if (std::isnan(depth))
         std::cout << (Q.toRotationMatrix()*Eigen::Vector3d(train_pts[0].x, train_pts[0].y, train_pts[0].z)).transpose()
                   << " + l" <<  T.transpose() << " = " << query_pts[0] << std::endl;

      max_error = mean_error = mut::NaN;
      int iterations;
      if (! std::isnan(depth))
      {
         Eigen::Vector3d oT = T;
         if (translation_levenberg_marquardt2d_depth(train_pts, query_pts, K, Q, oT, depth, iterations))
         {
            std::cout << "Refined Translation: " << oT.transpose() << " (After " << iterations << " iterations) ";
            show_reprojection(query_img, intrinsics, train_pts, query_pts, Q, oT, depth, "reproject-refined.jpg",
                              max_error, mean_error);
            std::cout << " Max Error " << max_error << ", mean error " << mean_error << std::endl;
         }
      }

      if ( (std::isnan(depth)) || (depth <= 0) )
      {
         if (train_pts.size() >= 5)
         {
            std::vector<Eigen::Quaterniond> Qs;
            std::vector<Eigen::Vector3d> Ts;
            poseother::FivePointRelativePose(intrinsics, train_pts, query_pts, Qs, Ts);
            if (Qs.size() > 0)
            {
               Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> EK((double*) intrinsics.data);
               std::cout << "****************** FivePointRelativePose" << std::endl;
               int ino = 1;
               for (size_t k = 0; k < Qs.size(); k++)
               {
                  Eigen::Matrix3d R = Qs[k].toRotationMatrix();
                  Eigen::Vector3d T = Ts[k];
                  const Eigen::Vector3d euler = R.eulerAngles(0, 1, 2);
                  std::cout << "Rotation: " << mut::radiansToDegrees(euler[0]) << "\u00B0 (" << euler[0] << "), "
                            << mut::radiansToDegrees(euler[1]) << "\u00B0 (" << euler[1] << ") "
                            << mut::radiansToDegrees(euler[2])
                            << "\u00B0 (" << euler[2] << ") " << std::endl;
                  std::cout << "Translation " << T.transpose();
                  double max_error, mean_error;
                  cv::Mat project_image;
                  query_img.copyTo(project_image);
                  project2d(EK, R, T, matched_pts, mut::NaN, max_error, mean_error, &project_image, true);
                  std::cout << " Max Error " << max_error << ", Mean Error " << mean_error;
                  std::stringstream ss;
                  ss << "reproject-5pt-" << ino++ << ".jpg";
                  cv::imwrite(ss.str().c_str(), project_image);
                  std::cout << std::endl << "---------------------------" << std::endl;
               }
            }
         }
         else
         {
            std::vector<cv::Mat> rotations, translations, normals;
            poseocv::homography_pose(intrinsics, train_pts, query_pts, rotations, translations, normals, true);
            std::cout << "decomposeHomography" << std::endl
                      << "-------------------------------------------------------------" << std::endl;
            for (size_t solution_no = 0; solution_no < rotations.size(); solution_no++)
            {
               cv::Mat& rotation = rotations[solution_no];
               Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R((double*) rotation.data);
               cv::Mat& translation = translations[solution_no];
               Eigen::Vector3d T(translation.at<double>(0, 0), translation.at<double>(1, 0), translation.at<double>(2, 0));
               cv::Mat normal = normals[solution_no].t();
               Eigen::Vector3d euler = R.eulerAngles(0, 1, 2);;
               std::cout << solution_no << ": Rotation: Roll " << mut::radiansToDegrees(euler[0]) << " (" << euler[0]
                         << "), Pitch "
                         << mut::radiansToDegrees(euler[1]) << " (" << euler[1] << ") Yaw " << mut::radiansToDegrees(euler[2])
                         << " (" << euler[2] << ") Translation: " << T.transpose() << std::endl;
            }
            std::cout << "=======================================================================" << std::endl;
         }
      }
      std::cout << "=======================================================================" << std::endl;
   }

   if ( (! world_pts.empty()) && (! image_pts.empty()) )
   {
      cv::Mat rotation_vec, translations3x1, R;
      long time;
      double max_error, mean_error;
      poseocv::display_PnP(world_pts, image_pts, intrinsics, false, &query_img, false, true);
      std::cout << "\n=========================================" << std::endl;
   }
}
