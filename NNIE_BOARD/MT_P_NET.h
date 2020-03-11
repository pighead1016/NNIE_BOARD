#pragma once
#include "NNIE_Net.h"
class MT_P_NET :
	public NNIE_Net
{
public:
	MT_P_NET();
	~MT_P_NET();
	HI_S32 INIT(HI_CHAR * pszModelFile);
	HI_S32 INIT(HI_CHAR *buffer, HI_U64 size);
	HI_S32 BBoxes(cv::Mat srcimg, std::vector<cv::Rect> &bbox, std::vector<float> &socers, const float thres = 0.7, HI_U8 Seg = 0);

protected:

};

