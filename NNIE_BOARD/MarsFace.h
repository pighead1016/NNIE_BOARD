#pragma once

#include "MT_O_NET.h"
#include "MT_P_NET.h"
#include "LandMark_Net.h"
enum LandmarkType
{
	Point81,
	Point5
};
class MarsFace
{
public:
	MarsFace();
	~MarsFace();
	HI_S32 Init_Detection(HI_CHAR* mtcnnFile);
	HI_S32 Init_LandMark(HI_CHAR* mtcnnFile);

	std::vector<cv::Rect> Detection_Face(cv::Mat& cubeimg);
	int Tracing(cv::Mat &cubeing, std::vector<cv::Rect> &face, std::vector<int> &face_num);
	int LandPoint(cv::Mat  &img, cv::Rect &bbox, std::vector<cv::Point2f> &pts, LandmarkType nettype = Point81);
	int FacePose(const std::vector<cv::Point2f> &pts, float &roll, float &yaw, float &pitch);
	int get_cut_face(const cv::Mat &img,const std::vector<cv::Point2f> &pts, float pitch, float yaw,
		cv::Mat& smoke_img,cv::Mat* phone_img);
	int crop_face(const cv::Mat &img, const std::vector<cv::Point2f> &pts, cv::Mat &crop_img);
private:
	std::vector<MT_P_NET> p_nets;
	std::vector<MT_O_NET> o_nets;
	std::vector<LandMark_Net> landmark_net;
	float thres[3] = { 0.7f,0.6f,0.85f };
	float nmsthres[3] = { 0.8f,0.8f,0.3f };

private:
	float min_score = 0.3;
	float max_score = 0.5;
	int max_save_num = 0;
	int now_save_num = 0;
	HI_U64 max_PID = 0;
	std::vector<cv::Rect> trace_faces;
	std::vector<int> trace_face_num;
};

