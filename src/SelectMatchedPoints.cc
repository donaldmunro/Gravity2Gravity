#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <iomanip>
#include <limits>
#include <memory>

#include <opencv2/cvconfig.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d.hpp>

#include "SelectMatchedPoints.h"

static inline void drawPoint(cv::InputOutputArray img, const cv::Point2i& pt, const cv::Scalar& color)
//---------------------------------------------------------------------------------------------------
{
//   const int draw_shift_bits = 4;
//   const int draw_multiplier = 1 << draw_shift_bits;
//   cv::Point center( cvRound(pt.x * draw_multiplier), cvRound(pt.y * draw_multiplier) );
//   int radius = 5 * draw_multiplier;
   cv::circle(img, pt, 2, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
   cv::circle(img, pt, 3, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
   cv::circle(img, pt, 4, color, 1, cv::LINE_AA);
   cv::circle(img, pt, 5,  cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
   cv::circle(img, pt, 6,  cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
}

SelectPoints::SelectPoints(const std::string& window_name, const cv::Mat& leftimg,
                           const cv::Mat& rightimg, const std::vector<cv::Point3d>* preselected_left_pts,
                           const std::vector<cv::Point3d>* preselected_right_pts) :
                           left_img(leftimg), right_img(rightimg), left_preselected_image_pts(preselected_left_pts),
                           right_preselected_image_pts(preselected_right_pts), window_name(window_name)
//--------------------------------------------------------------------------------------------------------------------
{
   left_size = left_img.size();
   right_size = right_img.size();
   right_start = left_size.width;
   if ( (left_preselected_image_pts != nullptr) && (! left_preselected_image_pts->empty()) &&
        (right_preselected_image_pts != nullptr) && (! right_preselected_image_pts->empty()) )
   {
      const size_t no = std::min(left_preselected_image_pts->size(), right_preselected_image_pts->size());
      for (; current_point < no; current_point++)
      {
         cv::Point3d pt = (*left_preselected_image_pts)[current_point];
         int x = static_cast<int>(round(pt.x));
         int y = static_cast<int>(round(pt.y));
         left_select_points.emplace_back(cv::Point2i(x, y));
         pt = (*right_preselected_image_pts)[current_point];
         x = static_cast<int>(round(pt.x));
         y = static_cast<int>(round(pt.y));
         right_select_points.emplace_back(cv::Point2i(x + right_start, y));
      }
      right_preselected_image_pts = nullptr;
      if (no >= left_preselected_image_pts->size())
         left_preselected_image_pts = nullptr;
   }
}

std::size_t SelectPoints::select(bool isRightOnly)
//------------------------------------------------
{
   cv::namedWindow(window_name, cv::WINDOW_GUI_EXPANDED | cv::WINDOW_AUTOSIZE);
#ifndef HAVE_QT
   if (is_trackbar)
   {
      cv::createTrackbar(TRACK_X, window_name, nullptr, left_size.width);
      cv::createTrackbar(TRACK_Y, window_name, nullptr, left_size.height);
      is_trackbar = true;
   }
#endif
   is_right_only = isRightOnly;
   cv::setMouseCallback(window_name, SelectPoints::on_mouse_callback, this);
   if ( (left_preselected_image_pts != nullptr) && (! left_preselected_image_pts->empty()) )
   {
      cv::Point3d pt = (*left_preselected_image_pts)[current_point++];
      auto x = static_cast<int>(std::round(pt.x));
      auto y = static_cast<int>(std::round(pt.y));
      cv::Point2i ipt(x, y);
      left_select_points.emplace_back(ipt);
      is_select_left = false;
   }
   else
      is_select_left = ! isRightOnly;
   draw();
   while (is_selecting)
   {
      if ( (is_dragging) && (region.x >= 0) && (region.y >= 0) && (region.width > 0) && (region.height > 0) &&
           (region.area() > 5) )
         draw();
      try
      {
         lastKey = cv::waitKey(10);
      }
      catch (...)
      {
         lastKey = -1;
      }
      if (lastKey > 0)
      {
         switch (lastKey)
         {
            case 8:
               if (! undo_stack.empty())
               {
                  UndoMemento& memento = undo_stack.top();
                  is_select_left = memento.is_select_left;
                  current_point = memento.current_point;
                  if (memento.last_point_left)
                     left_select_points.pop_back();
                  else
                     right_select_points.pop_back();
                  undo_stack.pop();
               }
               draw();
               break;
            case 27:
            case 13:
            case 10:
               is_selecting = false;
               break;
         }
      }
   }

   cv::setMouseCallback(window_name, nullptr);
   cv::destroyWindow(window_name);
   return std::max(left_select_points.size(), right_select_points.size());
}

std::vector<cv::Point2d> SelectPoints::left_points()
//--------------------------------------------------
{
   std::vector<cv::Point2d> pts;
   for (cv::Point2i pt : left_select_points)
      pts.emplace_back(pt.x, pt.y);
   return pts;
}

std::vector<cv::Point2d> SelectPoints::right_points()
//---------------------------------------------------
{
   std::vector<cv::Point2d> pts;
   for (cv::Point2i pt : right_select_points)
      pts.emplace_back(pt.x - right_start, pt.y);
   return pts;
}

std::vector<cv::Point3d> SelectPoints::left_points3d(double z)
//--------------------------------------------------
{
   std::vector<cv::Point3d> pts;
   for (cv::Point2i pt : left_select_points)
      pts.emplace_back(pt.x, pt.y, z);
   return pts;
}

std::vector<cv::Point3d> SelectPoints::right_points3d(double z)
//---------------------------------------------------------------
{
   std::vector<cv::Point3d> pts;
   for (cv::Point2i pt : right_select_points)
      pts.emplace_back(pt.x - right_start, pt.y, z);
   return pts;
}


void SelectPoints::draw()
//-----------------------
{
   current_display = cv::Mat(left_size.height, left_size.width + right_size.width, left_img.type());
   cv::Mat left(current_display, cv::Rect(0, 0, left_size.width, left_size.height));
   left_img.copyTo(left);
   cv::Mat right(current_display, cv::Rect(left_size.width, 0, right_size.width, right_size.height));
   right_img.copyTo(right);

   if ( (is_dragging) && (region.x >= 0) && (region.y >= 0) && (region.width > 0) && (region.height > 0) &&
        (region.area() > 5) )
   {
      if ((region.x + region.width) >= current_display.cols)
         region.x = current_display.cols - region.width;
      if ((region.y + region.height) >= current_display.rows)
         region.y = current_display.rows - region.height;
      rectangle(current_display, region.tl(), region.br(), cv::Scalar(0, 0, 255));
   }

   for (cv::Point2i pt : left_select_points)
      drawPoint(current_display, pt, red);
   for (cv::Point2i pt : right_select_points)
      drawPoint(current_display, pt, red);
//   current_display = display;
   cv::imshow(window_name, current_display);
}

void SelectPoints::on_left_click(int x, int y)
//-----------------------------------------------
{
   if ( (is_select_left) && (x < right_start) )
   {
      left_select_points.emplace_back(x, y);
      undo_stack.emplace(true, is_select_left, current_point);
      last_pt = left_select_points.back();
      is_select_left = false;
      draw();
   }
   else if ( (! is_select_left) && (x >= right_start) )
   {
      right_select_points.emplace_back(x, y);
      undo_stack.emplace(false, is_select_left, current_point);
      last_pt = right_select_points.back();
      if ( (left_preselected_image_pts != nullptr) && (! left_preselected_image_pts->empty()) )
      {
         if (current_point < left_preselected_image_pts->size())
         {
            cv::Point3d pt = (*left_preselected_image_pts)[current_point++];
            auto xp = static_cast<int>(round(pt.x));
            auto yp = static_cast<int>(round(pt.y));
            cv::Point2i ipt(xp, yp);
            left_select_points.emplace_back(ipt);
            undo_stack.emplace(true, false, current_point);
            is_select_left = false;
            //cv::circle(left_img, ipt, 2, cv::Scalar(0, 0, 255), 2);
         }
         else
            is_select_left = (! is_right_only);
      }
      else
         is_select_left = (! is_right_only);
      draw();
   }
   else
   {
#ifdef HAVE_QT
      std::stringstream ss;
      ss << "Select point in " << ((is_select_left) ? "left " : "right ") << " image";
      cv::displayStatusBar(window_name, ss.str(), 1000);
#endif
   }
}

void SelectPoints::on_mouse_moved(int x, int y, const bool is_shift, const bool is_ctrl, const bool is_alt)
//---------------------------------------------------------------------------------------------------------
{
   std::stringstream ss;
   if (x >= right_start)
      x -= right_start;
#ifdef HAVE_QT
   //cv::displayStatusBar(window_name, ss.str());
   cv::displayOverlay(window_name, ss.str());
#else
   if (is_trackbar)
   {
      cv::setTrackbarPos(TRACK_X, window_name, x);
      cv::setTrackbarPos(TRACK_Y, window_name, y);
   }
#endif
}

/*
void SelectPoints::on_mouse_dragged(cv::Rect region, const bool is_shift, const bool is_ctrl, const bool is_alt)
//--------------------------------------------------------------------------------------------------------------------
{
   if ( (region.x >= 0) && (region.y >= 0) && (region.width > 0) && (region.height > 0) && (region.area() > 5) )
   {
      if ( (region.x + region.width) >= current_display.cols)
         region.x = current_display.cols - region.width;
      if ( (region.y + region.height) >= current_display.rows)
         region.y = current_display.rows - region.height;
      int c = 0;
      for (auto i=0; i<left_select_points.size();)
      {
         cv::Point2i pt = left_select_points[i];
         if (region.contains(pt))
         {
            left_select_points.erase(left_select_points.begin()+i);
            if (right_select_points.size() >= i)
               right_select_points.erase(right_select_points.begin()+i);
            if ( (pt == last_pt) && (! is_select_left) )
               is_select_left = true;
            c++;
         }
         else
            ++i;
      }
      for (auto i=0; i<right_select_points.size();)
      {
         cv::Point2i pt = right_select_points[i];
         if (region.contains(pt))
         {
            right_select_points.erase(right_select_points.begin()+i);
            if (left_select_points.size() >= i)
               left_select_points.erase(left_select_points.begin()+i);
            c++;
         }
         else
            ++i;
      }
      if (c > 0)
         draw();
   }
}
*/

void SelectPoints::on_mouse(int ev, int x, int y, int flags)
//----------------------------------------------------------
{
   const bool is_shift = (flags & cv::EVENT_FLAG_SHIFTKEY) == cv::EVENT_FLAG_SHIFTKEY;
   const bool is_ctrl = (flags & cv::EVENT_FLAG_CTRLKEY) == cv::EVENT_FLAG_CTRLKEY;
   const bool is_alt = (flags & cv::EVENT_FLAG_ALTKEY) == cv::EVENT_FLAG_ALTKEY;
   switch (ev)
   {
      case cv::EVENT_MOUSEMOVE:
         last_x = cx;
         last_y = cy;
         cx = x;
         cy = y;
         if (is_dragging)
         {
            region.width = x - region.x;
            region.height = y - region.y;
         }
         else if ( (last_x != x) || (last_y != y) )
         {
            region.width = region.height = 0;
            on_mouse_moved(x, y, is_shift, is_ctrl, is_alt);
         }
         break;

      case cv::EVENT_LBUTTONDOWN:
         is_dragging = true;
         region = cv::Rect(x, y, 0, 0);
         break;

      case cv::EVENT_LBUTTONUP:
         is_dragging = false;
         if (region.width < 0)
         {
            region.x += region.width;
            region.width *= -1;
         }
         if (region.height < 0)
         {
            region.y += region.height;
            region.height *= -1;
         }
         if (region.area() <= 8)
         {
            region = cv::Rect(x, y, 0, 0);
            on_left_click(x, y);
         }
//         else
//            on_mouse_dragged(region, is_shift, is_ctrl, is_alt);
         break;
   }
}

void SelectPoints::on_mouse_callback(int ev, int x, int y, int flags, void *param)
//--------------------------------------------------------------------------------
{
   SelectPoints *me = reinterpret_cast<SelectPoints *>(param);
   me->on_mouse(ev, x, y, flags);
}