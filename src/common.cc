#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


#include "common.h"
#include "poseocv.h"

inline std::string trim(const std::string &str, std::string chars)
//----------------------------------------------------------------
{
   if (str.length() == 0)
      return str;
   auto b = str.find_first_not_of(chars);
   auto e = str.find_last_not_of(chars);
   if (b == std::string::npos) return "";
   return std::string(str, b, e - b + 1);
}


std::size_t split(std::string s, std::vector<std::string>& tokens, std::string delim)
//-----------------------------------------------------------------------------------
{
   tokens.clear();
   std::size_t pos = s.find_first_not_of(delim);
   while (pos != std::string::npos)
   {
      std::size_t next = s.find_first_of(delim, pos);
      if (next == std::string::npos)
      {
         tokens.emplace_back(trim(s.substr(pos)));
         break;
      }
      else
      {
         tokens.emplace_back(trim(s.substr(pos, next-pos)));
         pos = s.find_first_not_of(delim, next);
      }
   }
   return tokens.size();
}

void cvLabel(const cv::Mat& img, const std::string label, const cv::Point &pt, const cv::Scalar background,
             const cv::Scalar foreground, const int font)
//--------------------------------------------------------------------------------------------------------
{
   double scale = 0.4;
   int thickness = 1;
   int baseline = 0;
   cv::Size text = cv::getTextSize(label, font, scale, thickness, &baseline);
   cv::rectangle(img, pt + cv::Point(0, baseline), pt + cv::Point(text.width, -text.height), background, CV_FILLED);
   cv::putText(img, label, pt, font, scale, foreground, thickness, 8);
}

void display_rotation(const Eigen::Quaterniond& Q, bool isMinimal, bool is_eol)
//------------------------------------------------
{
   if (! isMinimal)
      std::cout << "Q = [" << Q.w() << " ("  << Q.x() << ", " << Q.y() << ", " << Q.z() << ") ]" << std::endl;
//   Eigen::Vector3d euler = mut::rotation2Euler(Q.toRotationMatrix());
//   std::cout << "Rotation: Roll " << mut::radiansToDegrees(euler[0]) << "\u00B0 ("  << euler[0] << "), Pitch "
//             << mut::radiansToDegrees(euler[1]) << "\u00B0 (" << euler[1] << ") Yaw " << mut::radiansToDegrees(euler[2])
//             << "\u00B0 (" << euler[2] << ")" << std::endl;
   Eigen::Vector3d euler = Q.toRotationMatrix().eulerAngles(0, 1, 2);
   std::cout << std::fixed << std::setprecision(4) << "Rotation Roll,Pitch,Yaw: [" << mut::radiansToDegrees(euler[0]) << ","
             << mut::radiansToDegrees(euler[1]) << "," << mut::radiansToDegrees(euler[2]) << " ]\u00B0 ("
             << euler[0] << "," << euler[1] << "," << euler[2] << ") radians ("
             << mut::radiansToDegrees(euler[0]) << "\\textdegree,"
             << mut::radiansToDegrees(euler[1]) << "\\textdegree," << mut::radiansToDegrees(euler[2]) << "\\textdegree)";
   if (isMinimal)
   {
      if (is_eol)
         std::cout << std::endl;
   }
   else
   {
      double angle;
      Eigen::AngleAxisd aa(Q);
      std::cout << "Axis: " << aa.axis().transpose() << " Angle: " << mut::radiansToDegrees(aa.angle()) << "\u00B0"
                << std::endl;
   }
}

void display_pts(const std::vector<cv::Point3d>& train_img_pts, const std::vector<cv::Point3d>& query_img_pts,
                 const std::vector<cv::Point3d>* original_pts)
//------------------------------------------------------------------------------------------------------------------
{
   for (auto i=0; i<train_img_pts.size(); i++)
   {
      Eigen::Vector3d tp(train_img_pts[i].x, train_img_pts[i].y, train_img_pts[i].z);
      Eigen::Vector3d qp(query_img_pts[i].x, query_img_pts[i].y, query_img_pts[i].z);
      std::cout << tp.transpose() << " -> " << qp.transpose();
      if (original_pts)
      {
         Eigen::Vector3d op((*original_pts)[i].x, (*original_pts)[i].y, (*original_pts)[i].z);
         std::cout << " (" << op.transpose() << ")" << std::endl;
      }
      else
         std::cout << std::endl;
   }
}

void display_pts3d(const std::vector<cv::Point3d>& world_pts, const std::vector<cv::Point3d>& image_pts,
                    const std::vector<cv::Point3d>& transformed_world_pts)
//--------------------------------------------------------------------------------------------------------
{
   for (auto i = 0; i < world_pts.size(); i++)
   {
      Eigen::Vector3d wp(world_pts[i].x, world_pts[i].y, world_pts[i].z);
      Eigen::Vector3d ip;
      if (image_pts.size() > i)
         ip = Eigen::Vector3d(image_pts[i].x, image_pts[i].y, image_pts[i].z);
      else
         ip = Eigen::Vector3d(-1, -1, -1);
      Eigen::Vector3d tp(transformed_world_pts[i].x, transformed_world_pts[i].y, transformed_world_pts[i].z);
//      Eigen::Vector3d tip(transformed_image_pts[i].x, transformed_image_pts[i].y, transformed_image_pts[i].z);
      std::cout << wp.transpose() << " | " << ip.transpose() << " -> " << tp.transpose() // << " | " << tip.transpose()
                << std::endl;
   }
}

void read_matches(const char *matchfile, std::vector<cv::Point3d>& train_pts, std::vector<cv::Point3d>& query_pts,
                  const cv::Mat* show_train_img, const cv::Mat* show_query_img, bool is_draw_circles,
                  bool isLabelQueryCoords)
//----------------------------------------------------------------------------------------------------------------
{
   std::ifstream ifs(matchfile);
   if (! ifs)
   {
      std::cerr << "Could not open match file " << matchfile << " (" << std::strerror(errno)
                << ")" << std::endl;
      std::exit(1);
   }
   char buf[120];
   ifs.getline(buf, 120);
   int c = 1, ypos = -1;
   while (!ifs.eof())
   {
      std::string line = trim(buf);
      if ( (line.empty()) || (line[0] == '#') )
      {
         ifs.getline(buf, 120);
         continue;
      }
      auto pos = line.find('#');
      if (pos != std::string::npos)
         line = line.substr(0, pos);
      std::string target;
      pos = line.find("->");
      if (pos == std::string::npos)
      {
         pos = line.find('=');
         if (pos == std::string::npos)
         {
            std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
            std::exit(1);
         }
         else
            try { target = trim(line.substr(pos + 1)); } catch (...) { target = ""; }
      }
      else
         try { target = trim(line.substr(pos + 2)); } catch (...) { target = ""; }
      std::string source = trim(line.substr(0, pos));
      std::vector<std::string> tokens;
      size_t n = split(source, tokens, ",");
      if ( (n < 2) || (n > 3) )
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      double tx = double_from_string(tokens[0]);
      double ty = double_from_string(tokens[1]);
      double tz;
      if (n > 2)
         tz = double_from_string(tokens[2]);
      else
         tz = 1;
      if ( (std::isnan(tx)) || (std::isnan(ty)) || (std::isnan(tz)) )
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      train_pts.emplace_back(tx, ty, tz);
      std::stringstream ss;
      ss << c++;
      if ( (show_train_img != nullptr) && (! show_train_img->empty()) )
      {
         cvLabel(*show_train_img, ss.str(), cv::Point2i(cvRound(tx-1), cvRound(ty-6)));
         if (is_draw_circles)
            plot_circles(*show_train_img, tx, ty);
         else
            plot_rectangles(*show_train_img, tx, ty);
      }

      n = split(target, tokens, ",");
      if (n != 2)
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      double qx = double_from_string(tokens[0]);
      double qy = double_from_string(tokens[1]);
      if ( (std::isnan(qx)) || (std::isnan(qy)) )
      {
         std::cerr << "Invalid line " << line << " in match file " << matchfile << std::endl;
         std::exit(1);
      }
      query_pts.emplace_back(qx, qy, 1);
      if ( (show_query_img != nullptr) && (! show_query_img->empty()) )
      {
         int xp = cvRound(qx - 1);
         int yp = cvRound(qy - 6);
         if (isLabelQueryCoords)
         {
            ss << " (" << tx << ", " << ty << ", " << tz << ")";
            if ( (ypos >= 0) && (std::abs(ypos-yp) < 5) )
            {
               yp -= 15;
               xp -= 5;
            }
            ypos = yp;
         }

         cvLabel(*show_query_img, ss.str(), cv::Point2i(std::max(xp, 0), std::max(yp, 0)));
         if (is_draw_circles)
            plot_circles(*show_query_img, qx, qy);
         else
            plot_rectangles(*show_query_img, qx, qy);
      }
      ifs.getline(buf, 120);
   }
}

void project3d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d t,
               const std::vector<std::pair<cv::Point3d, cv::Point2d>>& pts, double& max_error, double& mean_error,
               cv::Mat* project_image, bool draw_query, cv::Scalar train_color, cv::Scalar query_color)
//--------------------------------------------------------------------------------------------------------
{
   max_error = std::numeric_limits<double >::min(); mean_error = 0;
   std::vector<cv::Scalar> black(7), cyan(7);
   std::fill(black.begin(), black.begin()+7, train_color);
   std::fill(cyan.begin(), cyan.begin()+7, query_color);
   std::vector<double> errors;
   for (std::pair<cv::Point3d, cv::Point2d> pt : pts)
   {
      cv::Point2d qpt = pt.second;
      if ( (project_image != nullptr) && (! project_image->empty()) && (draw_query) )
         plot_rectangles(*project_image, qpt.x, qpt.y, cyan);
      cv::Point3d tpt = pt.first;
      Eigen::Vector3d ipt = K*(R*Eigen::Vector3d(tpt.x, tpt.y, tpt.z) + Eigen::Vector3d(t[0], t[1], t[2]));
      if (! mut::near_zero(ipt[2], 0.0000001))
         ipt /= ipt[2];
      if ( (project_image != nullptr) && (! project_image->empty()) )
         plot_circles(*project_image, ipt[0], ipt[1], black);
      double error = (ipt - Eigen::Vector3d(qpt.x, qpt.y, 1)).norm();
      if (error > max_error)
         max_error = error;
      mean_error += error;
   }
   mean_error /= pts.size();
}

void project3d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               const std::vector<cv::Point3d>& world_points, const std::vector<cv::Point2d>& image_pts,
               double& max_error, double& mean_error, cv::Mat* project_image, bool draw_query,
               cv::Scalar train_color, cv::Scalar query_color)
//----------------------------------------------------------------------------------------------------
{
   std::vector<std::pair<cv::Point3d, cv::Point2d>> pts;
   for (size_t i=0; i<std::min(world_points.size(), image_pts.size()); i++)
      pts.emplace_back(std::make_pair(world_points[i], image_pts[i]));
   project3d(K, R, T, pts, max_error, mean_error, project_image, draw_query, train_color, query_color);
}

void project3d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               const std::vector<cv::Point3d>& world_points, const std::vector<cv::Point3d>& image_pts,
               double& max_error, double& mean_error, cv::Mat* project_image, bool draw_query,
               cv::Scalar train_color, cv::Scalar query_color)
//----------------------------------------------------------------------------------------------------
{
   std::vector<std::pair<cv::Point3d, cv::Point2d>> pts;
   for (size_t i=0; i<std::min(world_points.size(), image_pts.size()); i++)
   {
      const cv::Point3d& ipt = image_pts[i];
      pts.emplace_back(std::make_pair(world_points[i], cv::Point2d(ipt.x, ipt.y)));
   }
   project3d(K, R, T, pts, max_error, mean_error, project_image, draw_query, train_color, query_color);
}

void project2d(const Eigen::Matrix3d K, const Eigen::Matrix3d R, const Eigen::Vector3d T,
               std::vector<std::pair<cv::Point3d, cv::Point3d>>& pts,
               const double depth, double& max_error, double& mean_error,
               cv::Mat* project_image, bool draw_query, cv::Scalar train_color, cv::Scalar query_color)
//----------------------------------------------------------------------------------------------------------------
{
   max_error = std::numeric_limits<double >::min(); mean_error = 0;
   std::vector<cv::Scalar> black(7), cyan(7);
   std::fill(black.begin(), black.begin()+7, train_color);
   std::fill(cyan.begin(), cyan.begin()+7, query_color);
   Eigen::Matrix3d KI = K.inverse();
   for (std::pair<cv::Point3d, cv::Point3d> pt : pts)
   {
      cv::Point3d qpt = pt.second;
      if ( (project_image != nullptr) && (! project_image->empty()) && (draw_query) )
         plot_rectangles(*project_image, qpt.x, qpt.y, cyan);
      cv::Point3d tpt = pt.first;
      Eigen::Vector3d ray = KI*Eigen::Vector3d(tpt.x, tpt.y, 1);
      ray /= ray[2];
      if (! std::isnan(depth))
         ray *= depth;
      Eigen::Vector3d ipt = R*ray + T;
//      std::cout << (R*ray).transpose() << " + " << T.transpose() << " = " << ipt.transpose() << std::endl;
      ipt = K*ipt;
      ipt /= ipt[2];
      if ( (project_image != nullptr) && (! project_image->empty()) )
         plot_circles(*project_image, ipt[0], ipt[1], black);

      double error = (ipt - Eigen::Vector3d(qpt.x, qpt.y, qpt.z)).norm();
      if (error > max_error)
         max_error = error;
      mean_error += error;
//     std::cout << j << ": " << qpt << " " << ipt.transpose() << " = " << error << std::endl;
   }
   mean_error /= pts.size();
}

void noise(const double deviation, std::vector<cv::Point3d>& query_pts, bool isRoundInt, bool isX, bool isY, bool isZ,
           std::vector<cv::Point3d>* destination)
//-------------------------------------------------------------------------------------------------------
{
   std::random_device rd{};
   std::mt19937 gen{rd()};
   std::normal_distribution<double> N{0,deviation};
   if (destination != nullptr)
      destination->clear();
   for (cv::Point3d& pt : query_pts)
   {
      double& x = pt.x;
      double& y = pt.y;
      double& z = pt.z;
      double xnoise =0, ynoise =0, znoise =0;
      if (isX)
         xnoise = N(gen);
      if (isY)
         ynoise = N(gen);
      if (isZ)
         znoise = N(gen);
      if (destination != nullptr)
      {
         if (isRoundInt)
            destination->emplace_back(std::round(x + xnoise), std::round(y + ynoise), std::round(z + znoise));
         else
            destination->emplace_back(x + xnoise, y + ynoise, z + znoise);
      }
      else
      {
         x += xnoise;
         y += ynoise;
         z += znoise;
         if (isRoundInt)
         {
            x = std::round(x);
            y = std::round(y);
            z = std::round(z);
         }
      }
   }

}

std::ostream& operator<<(std::ostream& out, const Eigen::Quaterniond& Q)
{
   out << "[ " << Q.w() << ", (" << Q.x() << ", " << Q.y() << ", " << Q.z() << " ) ]";
   return out;
}
