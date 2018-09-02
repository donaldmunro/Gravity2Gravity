#ifndef GRAVITYPOSE_UT_H
#define GRAVITYPOSE_UT_H

#include <ostream>
#include <string>
#include <vector>
#include <random>

#include <Eigen/Core>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "math.hh"

std::string trim(const std::string &str, std::string chars  = " \t");

std::size_t split(std::string s, std::vector<std::string>& tokens, std::string delim ="\t\n ");

void cvLabel(const cv::Mat& img, const std::string label, const cv::Point &pt, const cv::Scalar background =cv::Scalar(0, 0, 0),
             const cv::Scalar foreground =cv::Scalar(255, 255, 255), const int font = cv::FONT_HERSHEY_DUPLEX);

inline double double_from_string(std::string s) { try { return std::stod(trim(s)); } catch (...) { return mut::NaN; }}
inline double float_from_string(std::string s) { try { return std::stof(trim(s)); } catch (...) { return mut::NaN_F; }}

void display_rotation(const Eigen::Quaterniond& Q, bool isMinimal =false, bool is_eol =true);

void display_pts(const std::vector<cv::Point3d>& train_img_pts, const std::vector<cv::Point3d>& query_img_pts,
                 const std::vector<cv::Point3d>* original_pts =nullptr);

void display_pts3d(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point3d>& image_pts,
                    const std::vector<cv::Point3d>& transformed_world_pts);

void read_matches(const char *matchfile, std::vector<cv::Point3d>& train_pts, std::vector<cv::Point3d>& query_pts,
                  const cv::Mat* show_train_img =nullptr, const cv::Mat* show_query_img =nullptr,
                  bool is_draw_circles =true, bool isLabelQueryCoords =false);

void project3d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts,double& max_error, double& mean_error,
               cv::Mat* project_image = nullptr, bool draw_query =true,
               cv::Scalar train_color = cv::Scalar(0, 0, 0), cv::Scalar query_color = cv::Scalar(0, 255, 255));

void project3d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               const std::vector<cv::Point3d>& world_points, const std::vector<cv::Point2d>& image_pts,
               double& max_error, double& mean_error,
               cv::Mat* project_image = nullptr, bool draw_query =true,
               cv::Scalar train_color = cv::Scalar(0, 0, 0), cv::Scalar query_color = cv::Scalar(0, 255, 255));

void project3d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               const std::vector<cv::Point3d>& world_points, const std::vector<cv::Point3d>& image_pts,
               double& max_error, double& mean_error,
               cv::Mat* project_image = nullptr, bool draw_query =true,
               cv::Scalar train_color = cv::Scalar(0, 0, 0), cv::Scalar query_color = cv::Scalar(0, 255, 255));


static int show_images() { return 0; }

void project2d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
               const double depth, double& max_error, double& mean_error,
               cv::Mat* project_image = nullptr, bool draw_query =true,
               cv::Scalar train_color = cv::Scalar(0, 0, 0), cv::Scalar query_color = cv::Scalar(0, 255, 255));

void noise(const double deviation, std::vector<cv::Point3d>& query_pts, bool isRoundInt, bool isX =true, bool isY =true,
           bool isZ =false, std::vector<cv::Point3d>* destination = nullptr);

template <typename ...Args>
int show_images(const char* title, const cv::Mat& img, int offset, Args... args)
//-------------------------------------------------------------------------------
{
   int n = show_images(args...);
   if (! img.empty())
   {
      cv::namedWindow(title);
      cv::imshow(title, img);
      cv::moveWindow(title, offset, 0);
      return n + 1;
   }
   else
      return n;
}

inline void plot_circles(const cv::Mat& img, double x, double y,
                         const std::vector<cv::Scalar>& colors  ={ cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255),
                                                                   cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 255),
                                                                   cv::Scalar(0, 0, 255)})
//----------------------------------------------------------------------------------------------------------------
{
   int i = 2;
   for (cv::Scalar color : colors)
   {
      cv::circle(img, cv::Point2i(cvRound(x), cvRound(y)), i, color, 1);
      i++;
   }
}

inline void plot_rectangles(const cv::Mat& img, double x, double y,
                            const std::vector<cv::Scalar>& colors  ={ cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255),
                                                                      cv::Scalar(255, 0, 0),cv::Scalar(0, 255, 255),
                                                                      cv::Scalar(0, 0, 255)})
//--------------------------------------------------------------------------------------------------------
{
   int i = 1;
   for (cv::Scalar color : colors)
   {
      cv::rectangle(img, cv::Point2i(cvRound(x) - i, cvRound(y) - i), cv::Point2i(cvRound(x) + i, cvRound(y) + i),
                    color, 1);
      i++;
   }
}


std::ostream& operator<<(std::ostream& out, const Eigen::Quaterniond& Q);
#endif
