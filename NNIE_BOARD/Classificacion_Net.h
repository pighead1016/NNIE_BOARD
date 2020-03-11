#pragma once
#include "NNIE_Net.h"
class Classificacion_Net :
	public NNIE_Net
{
public:
	Classificacion_Net();
	~Classificacion_Net();
	HI_S32 CLASSIF_INIT(HI_CHAR * pszModelFile);
	HI_S32 Classificacion_num;
	HI_S32 Classificacion(cv::Mat img, std::vector<float>& score);
};

