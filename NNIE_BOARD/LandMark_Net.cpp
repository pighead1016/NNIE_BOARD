#include "LandMark_Net.h"



LandMark_Net::LandMark_Net()
{
}


LandMark_Net::~LandMark_Net()
{
}

HI_S32 LandMark_Net::INIT(HI_CHAR * pszModelFile, HI_CHAR * matFile)
{
	HI_S32 s32Ret = NNIE_NET_INIT(pszModelFile);
	this->Read_weight_bias(matFile);
	this->Point_nums = this->astSegData[1].astDst[0].unShape.stWhc.u32Width / 2;
	stand_pts.resize(Point_nums);
	return s32Ret;
}

HI_S32 LandMark_Net::INIT(HI_CHAR * buffer, HI_U64 size)
{
	HI_S32 s32Ret = NNIE_NET_INIT(buffer, size);
	this->Point_nums = this->astSegData[1].astDst[0].unShape.stWhc.u32Width / 2;
	stand_pts.resize(Point_nums);
	return s32Ret;
}

HI_S32 LandMark_Net::stand_LandMark_point(cv::Mat& img, std::vector<cv::Point2f>& Pt,bool fase)
{
	cv::Mat gray_img;
	if (img.channels() > 1)
		cv::cvtColor(img, gray_img, CV_BGR2GRAY);
	else
		gray_img = img;
	//first seg
	SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S input_index = { 0,0 };
	SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S process_index = { 0,1 };
	SVP_FillSrcData_Mat(&input_index, gray_img);
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	auto first_dst0 = astSegData[process_index.u32SegIdx].astDst[0];
	auto first_dst1 = astSegData[process_index.u32SegIdx].astDst[process_index.u32NodeIdx];
	HI_S32* data = (HI_S32 *)first_dst1.u64VirAddr;
	//second seg
	input_index = { 1,0 };
	process_index = { 1,0 };
	//fill input of seg second
	Inner(data, (void*)astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr, true);

	SAMPLE_SVP_CHECK_EXPR_TRACE(astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u32Stride !=
		4 * astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].unShape.stWhc.u32Width,
		SAMPLE_SVP_ERR_LEVEL_FATAL, "Error,seg input width failed!\n");
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	HI_MPI_SYS_MmzFlushCache(astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64PhyAddr,
	SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr),astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u32Stride);
#endif
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	auto second_dst = astSegData[process_index.u32SegIdx].astDst[process_index.u32NodeIdx];
	data = (HI_S32 *)second_dst.u64VirAddr;
	int nn = 0;
	for (auto& stand_pt : stand_pts) {
		stand_pt.x = data[nn] / 4096.0f;
		stand_pt.y = data[nn + 1] / 4096.0f;
		nn += 2;
	}
	if (fase)
	{
		Pt = stand_pts;
		return 2;
	}
	//third seg
	input_index = { 2,0 };
	process_index = { 2,0 };

	//HI_S32 input_c = this->astSrcs[2][0].unShape.stWhc.u32Chn;
	HI_S32 input_h = astSegData[input_index.u32SegIdx].astDst[input_index.u32NodeIdx].unShape.stWhc.u32Height;
	HI_S32 input_w = astSegData[input_index.u32SegIdx].astDst[input_index.u32NodeIdx].unShape.stWhc.u32Width;
	HI_S32 input_c = first_dst0.unShape.stWhc.u32Chn;//8
	HI_S32 outh = first_dst0.unShape.stWhc.u32Height;//27
	HI_S32 outw = first_dst0.unShape.stWhc.u32Width;//27

	//HI_S32 np = this->astDsts[0][1].unShape.stWhc.u32Width / 2;
	int *x = new int[Point_nums];
	int *y = new int[Point_nums];
	for (int p = 0; p < Point_nums; p++)
	{
		x[p] = int(this->stand_pts[p].x * (outh - 1)) - 1;
		y[p] = int(this->stand_pts[p].y * (outh - 1)) - 1;
	}
	HI_S32 *output_data = (HI_S32 *) first_dst0.u64VirAddr;
	HI_S32 *input_data = (HI_S32*)astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr;
	HI_U64 numm = 0;
	for (int c = 0; c < input_c; c++)
		for (int h = 0; h < 4; h++)
			for (int w = 0; w < Point_nums; w++)
				for (int p = 0; p < 4; p++)
				{
					int xx = x[w] + p;
					int yy = y[w] + h;
					if (xx >= 0 && xx < outh && yy >= 0 && yy < outh) {
						auto srcid = outh* outh* c + outw* yy + xx;
						input_data[numm++] = output_data[srcid];
					}
					else {
						input_data[numm++] = 0.0;
					}
				}
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);


	HI_S32 *remark = (HI_S32 *)astSegData[process_index.u32SegIdx].astDst[process_index.u32NodeIdx].u64VirAddr;
	//Pt.resize(np);
	for (int i = 0; i < Point_nums; i++)
	{
		this->stand_pts[i].x += remark[2 * i] / 4096.0;
		this->stand_pts[i].y += remark[2 * i + 1] / 4096.0;
	}
	Pt = stand_pts;
	delete[] x;
	delete[] y;
	return HI_S32();
}
