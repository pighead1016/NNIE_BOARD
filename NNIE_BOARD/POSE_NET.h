#pragma once
#include "NNIE_Net.h"

class POSE_NET :
	protected NNIE_Net
{
public:
	POSE_NET();
	~POSE_NET();
	HI_S32 POSE_INIT(HI_CHAR * pszModelFile);
	std::vector<float> keypoints;
	void show_keypoint(cv::Mat& image);
	HI_S32 KetPoint(const cv::Mat img, float scale = 8.0f);
	int POSE_MAX_PEOPLE = 5;
private:
	std::vector<int> shape;
	cv::Size BaseSize;
	int numberBodyParts;
	void connectBodyPartsCpu(const float* const heatMapPtr, const float* const peaksPtr,
		const cv::Size& heatMapSize, const int maxPeaks, const int interMinAboveThreshold,
		const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor);
};

