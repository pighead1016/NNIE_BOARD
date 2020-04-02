#pragma once
#include <opencv2/opencv.hpp>
/**
* line for ax + by + c = 0
*/
class Line {
public:
	inline Line() = default;
	inline Line(float a, float b, float c)
		: a(a), b(b), c(c) {    this->angle=atan(	this->a / 	this->b) * 180 / CV_PI;}
  inline Line(const cv::Point2f &p, const float & angle) {
    float angle_rad=angle/180* CV_PI;
    this->angle=angle;
    this->p1 = p;
    x = p1.x;
    y = p1.y;
    this->p2 = cv::Point2f( line_x(160.0f),160.0f );
    a=sin(angle_rad);
    b=-cos(angle_rad);
    c=y*cos(angle_rad)-x*sin(angle_rad);
  }

	inline Line(const cv::Point2f &a, const cv::Point2f &b) {
		auto x1 = a.x;
		auto y1 = a.y;
		auto x2 = b.x;
		auto y2 = b.y;
    this->p1 = a;
    this->p2 = b;

		this->a = y2 - y1;
		this->b = x1 - x2;
		this->c = y1 * (x2 - x1) - x1 * (y2 - y1);
    this->x=(x1+x2)/2;
    this->y=(y1+y2)/2;
    this->angle=atan(	this->a / 	this->b) * 180 / CV_PI;
	}

	inline float distance(const cv::Point2f &p)const  {
		return /*std::fabs*/(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
	}

	static bool near_zero(float f) {
		return f <= DBL_EPSILON && -f <= DBL_EPSILON;
	}

	cv::Point2f projection(const cv::Point2f &p) const {
		if (near_zero(a)) {
			cv::Point2f result;
			result.x = p.x;
			result.y = -c / b;
			return  result;
		}
		if (near_zero(b)) {
			cv::Point2f result;
			result.x = -c / a;
			result.y = p.y;
			return result;
		}
		// y = kx + b  <==>  ax + by + c = 0
		auto k = -a / b;
		cv::Point2f o = { 0, float(-c / b) };
		cv::Point2f project = { 0 };
		project.x = (float)((p.x / k + p.y - o.y) / (1 / k + k));
		project.y = (float)(-1 / k * (project.x - p.x) + p.y);
		return project;
	}
  inline float line_x( int y_row)const{
      return -(y_row*b+c)/a;
  }
  inline float line_x( float y_row)const{
      return -(y_row*b+c)/a;
  }
  float distance(const Line& l,int row_y=80.0f) {
    return l.line_x(row_y)-this->line_x(row_y);
  }
  float similar(const Line& lane,float row_y=160.0f) {
    float lane_up_x = lane.line_x(0);
    float lane_down_x =lane. line_x(row_y);
    return fabs(distance(cv::Point2f(lane_up_x,0))) + fabs(distance(cv::Point2f(lane_down_x,row_y)));
  }
  float x,y;
  //cv::Point2f center(x,y);
  cv::Point2f p1,p2;
	float a = 0;
	float b = 0;
	float c = 0;
  float angle=0;
  bool state=false;
};
