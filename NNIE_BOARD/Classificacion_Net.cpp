#include "Classificacion_Net.h"



Classificacion_Net::Classificacion_Net()
{
}


Classificacion_Net::~Classificacion_Net()
{
}

HI_S32 Classificacion_Net::CLASSIF_INIT(HI_CHAR * pszModelFile)
{
	HI_S32 s32Ret = NNIE_NET_INIT(pszModelFile);
	/*check*/
	SAMPLE_SVP_CHECK_EXPR_RET(stModel.u32NetSegNum != 1, HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,seg is only 1 failed!\n");
	SAMPLE_SVP_CHECK_EXPR_RET(stModel.astSeg[0].u16DstNum != 1, HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,seg 0 dstnode is only 1 failed!\n");
	Classificacion_num = astSegData[0].astDst[0].unShape.stWhc.u32Width;
	return s32Ret;
}

HI_S32 Classificacion_Net::Classificacion(cv::Mat img, std::vector<float>& score)
{
	score.resize(Classificacion_num);
	SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S input_index = { 0,0 };
	SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S process_index = { 0,0 };
	SVP_FillSrcData_Mat(&input_index, img);
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	HI_S32* prob = (HI_S32*)astSegData[0].astDst[0].u64VirAddr;
	for (int i = 0; i < Classificacion_num; i++)
	{
		score[i] = prob[i] / 4096.0;
	}
	return HI_S32();
}
