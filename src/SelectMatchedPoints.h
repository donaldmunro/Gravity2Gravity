#ifndef TRAINER_SELECTPLANAR_H
#define TRAINER_SELECTPLANAR_H

/**
 * Allow users to select corresponding points in two images. If preselected_left_pts in the
 * constructor is not null then it is assumed to contain points in the source (left) image
 * which are displayed one by one allowing the user to select only right hand (destination)
 * points corresponding to the predefined points in the left image. If preselected_left_pts
 * is null then the user selects a left image point followed by a corresponding right hand
 * point. Press Enter to quit.
 */

#include <math.h>

#include <map>
#include <unordered_map>
#include <stack>

#include <opencv2/core/base.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>


typedef struct
{
   std::size_t operator()(const cv::KeyPoint& v) const
   //---------------------------------------------
   {
      std::size_t h = 2166136261U;
      const size_t scale = 16777619U;
      Cv32suf u;
      float x = roundf(v.pt.x);
      float y = roundf(v.pt.y);
      u.f = x; h = (scale * h) ^ u.u;
      u.f = y; h = (scale * h) ^ u.u;
      return h;
   }
} KeypointPtHash;

typedef struct
{
   bool operator()(const cv::KeyPoint& k1, const cv::KeyPoint& k2) const
   //------------------------------------------------------------
   {
      KeypointPtHash hasher;
      return (hasher(k1) == hasher(k2));
   }
} KeypointPtEq;

typedef union
{
   uint32_t class_id;
   struct
   {
      unsigned  no            : 16; // Keypoint number
      unsigned  imgIdx        : 8; // Training image id
      bool      is_selected   : 1; // 1 selected in trainer
      bool      is_hull       : 1; // 1 if part of convex hull else 0
   };
} KeyPointExtra_t;

struct UndoMemento
{
   bool last_point_left, is_select_left;
   size_t current_point;

   UndoMemento(bool was_left, bool select_left, size_t current_pt) : last_point_left(was_left),
      is_select_left(select_left), current_point(current_pt) {}
};

class SelectPoints
//===================
{
public:
   SelectPoints(const std::string& window_name, const cv::Mat& leftimg, const cv::Mat& rightimg,
                const std::vector<cv::Point3d>* preselected_left_pts = nullptr,
                const std::vector<cv::Point3d>* preselected_right_pts = nullptr);
   virtual ~SelectPoints()
   {
      current_display.release();
      left_img.release();
      right_img.release();
   }

   size_t select(bool isRightOnly =false);

   std::vector<cv::Point2d> left_points();
   std::vector<cv::Point2d> right_points();
   std::vector<cv::Point3d> left_points3d(double z=1);
   std::vector<cv::Point3d> right_points3d(double z=1);

   inline int last_key() { return lastKey; };

protected:
   virtual void on_left_click(int x, int y);
   virtual void on_mouse_moved(int x, int y, const bool is_shift, const bool is_ctrl, const bool is_alt);
//   virtual void on_mouse_dragged(cv::Rect region, const bool is_shift, const bool is_ctrl, const bool is_alt);

   int lastKey =-1;
   cv::Mat left_img, right_img;
   cv::Size left_size, right_size;
   const std::vector<cv::Point3d> *left_preselected_image_pts, *right_preselected_image_pts;
   size_t current_point =0;
   int right_start;
   int last_x =-1, last_y =-1;
   cv::Point2i last_pt;
   bool is_select_left = true, is_right_only = false;
   std::vector<cv::Point2i> left_select_points, right_select_points;
#ifndef HAVE_QT
   const std::string TRACK_X = "X", TRACK_Y = "Y";
#endif

private:
   void draw();
   static void on_mouse_callback(int ev, int x, int y, int flags, void *param);
   void on_mouse(int ev, int x, int y, int flags);

   std::string window_name;
   int cx =-1, cy =-1;
   bool is_dragging = false;
   bool is_selecting = true;
   cv::Rect region;
   const cv::Scalar red = cv::Scalar(0, 0, 255), green = cv::Scalar(0, 255, 0), yellow = cv::Scalar(0, 255, 255);
   cv::Mat current_display;
   std::stack<UndoMemento> undo_stack;
#ifndef HAVE_QT
   bool is_trackbar = false;
#endif
};
#endif //TRAINER_SELECTPLANAR_H
