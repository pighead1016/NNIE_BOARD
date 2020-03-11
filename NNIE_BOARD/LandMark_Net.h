#pragma once
#include "MT_O_NET.h"
class LandMark_Net :
	public NNIE_Net
{
	//friend MT_O_NET;
public:
	LandMark_Net();
	~LandMark_Net();
	HI_S32 INIT(HI_CHAR * pszModelFile, HI_CHAR * matFile);
	HI_S32 INIT(HI_CHAR *buffer, HI_U64 size);
	HI_S32 stand_LandMark_point(cv::Mat &img, std::vector<cv::Point2f> &Pt, bool fast = false);
	HI_S32 Point_nums;
	std::vector<cv::Point2f> stand_pts;

};

