#include "MT_O_NET.h"



MT_O_NET::MT_O_NET()
{
}


MT_O_NET::~MT_O_NET()
{
}

HI_S32 MT_O_NET::INIT(HI_CHAR * pszModelFile)
{
	HI_S32 s32Ret = NNIE_NET_INIT(pszModelFile);
	return s32Ret;
}

HI_S32 MT_O_NET::INIT(HI_CHAR * pszModelFile, HI_CHAR * matFile)
{
	HI_S32 s32Ret = NNIE_NET_INIT(pszModelFile);
	this->Read_weight_bias(matFile);
	return s32Ret;
}

HI_S32 MT_O_NET::INIT(HI_CHAR * buffer, HI_U64 size)
{
	HI_S32 s32Ret = NNIE_NET_INIT(buffer,size);
	return s32Ret;
}

HI_S32 MT_O_NET::Confirm_bboxes(const cv::Mat core_img, const cv::Rect in_bbox, cv::Rect & out_bbox, float & socer, const float thres, float pad, HI_U8 Seg)
{
	int padding_width = pad * core_img.cols;
	int padding_height = pad * core_img.rows;
	cv::Mat pading_img(padding_height * 2 + core_img.rows, padding_width * 2 + core_img.cols, core_img.type());

	core_img.copyTo(pading_img(cv::Rect(padding_width, padding_height, core_img.cols, core_img.rows)));
	cv::Rect pad_rect(padding_width + in_bbox.x, padding_height + in_bbox.y, in_bbox.width, in_bbox.height);
	if (pad_rect.x < 0 || pad_rect.x + pad_rect.width >= pading_img.cols ||
		pad_rect.y < 0 || pad_rect.y + pad_rect.height >= pading_img.rows) {
		socer = -1;
		return HI_FAILURE;
	}
	HI_S32 *data;
	cv::Mat temp = pading_img(pad_rect);
	SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S input_index = { 0,0 };
	SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S process_index = { 0,0 };
	SVP_FillSrcData_Mat(&input_index, temp);
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	data = (HI_S32 *)(astSegData[process_index.u32SegIdx].astDst[process_index.u32NodeIdx].u64VirAddr);
	if (!this->weight_matrix.empty() || !bias_matrix.empty()) {

		//cv::Mat seg1_in = Inner(data, true);
		input_index = { 1,0 };
		process_index = { 1,0 };
		Inner(data, (void*)astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr, true);
		//memcpy((void*)astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr,(void*)seg1_in.data,astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u32Stride);
		SAMPLE_SVP_CHECK_EXPR_TRACE(astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u32Stride != 
			4* astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].unShape.stWhc.u32Width, 
			SAMPLE_SVP_ERR_LEVEL_FATAL,"Error,seg input width failed!\n");
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
		HI_MPI_SYS_MmzFlushCache(astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64PhyAddr,
		SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr),astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u32Stride);
#endif
		SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
		data = (HI_S32 *)(astSegData[process_index.u32SegIdx].astDst[process_index.u32NodeIdx].u64VirAddr);
	}
	if (data[1] > thres * 4096 && data[1] <= 4096)
	{
		float sn = data[2] / 4096.f;
		float xn = data[3] / 4096.f;
		float yn = data[4] / 4096.f;
		int crop_x = pad_rect.x;
		int crop_y = pad_rect.y;
		int crop_w = pad_rect.width;
		int rw = int(sn * crop_w);
		int rx = int(crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w);
		int ry = int(crop_y - 0.5 * sn * crop_w + crop_w * sn * yn + 0.5 * crop_w);

		if (rx >= 0 && rx + rw < pading_img.cols && ry >= 0 && ry + rw < pading_img.rows)
		{
			out_bbox.x = rx - padding_width;
			out_bbox.y = ry - padding_height;
			out_bbox.width = rw;
			out_bbox.height = rw;
			socer = data[1] / 4096.f;
		}
		else
		{
			socer = -1;
		}
	}
	else
	{
		socer = 0;
	}
	return HI_SUCCESS;
}
