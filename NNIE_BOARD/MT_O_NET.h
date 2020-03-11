#pragma once
#include "NNIE_Net.h"
class MT_O_NET :
	public NNIE_Net
{
public:
	MT_O_NET();
	~MT_O_NET();
public:
	HI_S32 INIT(HI_CHAR * pszModelFile);
	HI_S32 INIT(HI_CHAR * pszModelFile, HI_CHAR * matFile);
	HI_S32 INIT(HI_CHAR *buffer, HI_U64 size);
public:
	HI_S32 Confirm_bboxes(const cv::Mat cure_img, const cv::Rect in_bbox, cv::Rect &out_bbox, float &socer, const float thres=0.7, float pad = 0.2, HI_U8 Seg = 0);

};

