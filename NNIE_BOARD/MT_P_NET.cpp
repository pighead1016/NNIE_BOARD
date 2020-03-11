#include "MT_P_NET.h"
MT_P_NET::MT_P_NET()
{
}

MT_P_NET::~MT_P_NET()
{
}

HI_S32 MT_P_NET::INIT(HI_CHAR * pszModelFile)
{
	HI_S32 s32Ret = NNIE_NET_INIT(pszModelFile);
	return s32Ret;
}

HI_S32 MT_P_NET::INIT(HI_CHAR * buffer, HI_U64 size)
{
	HI_S32 s32Ret = NNIE_NET_INIT(buffer, size);
	return s32Ret;
}

HI_S32 MT_P_NET::BBoxes(cv::Mat img, std::vector<cv::Rect>& bbox, std::vector<float>& scores, const float thres, HI_U8 Seg)
{
	SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S input_index = { 0,0 };
	SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S process_index = { 0,0 };
	SVP_FillSrcData_Mat(&input_index, img);
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	//printf("%d %d pnet this add = %llu\n",input_index.u32SegIdx,input_index.u32NodeIdx,astSegData[input_index.u32SegIdx].astSrc[input_index.u32NodeIdx].u64VirAddr);
	auto output=astSegData[process_index.u32SegIdx].astDst[process_index.u32NodeIdx];
	float cur_scale = (float)img.cols / this->stModel.astSeg[input_index.u32SegIdx].astSrcNode[input_index.u32NodeIdx].unShape.stWhc.u32Width;
	float ww = 12 * cur_scale;
	float w_h = (float)img.rows/img.cols ;
	int stride_ = 4;
	for (int h = 0; h <output.unShape.stWhc.u32Chn; h++)
		for (int w = 0; w < output.unShape.stWhc.u32Height; w++) {
			int nu = w + h*output.unShape.stWhc.u32Height;
			HI_S32 *data = (HI_S32 *)(output.u64VirAddr + nu*output.u32Stride);
			if (data[1] > thres * 4096)
			{
				float sn = data[2] / 4096.0;
				float xn = data[3] / 4096.0;
				float yn = data[4] / 4096.0;

				int crop_x = int(w * cur_scale * stride_);
				int crop_y = int(h * cur_scale * stride_*w_h);
				int crop_w = int(ww);
				int crop_h = int(ww*w_h);
				int rx = int(crop_x - 0.5 * sn * crop_w + crop_w * sn * xn + 0.5 * crop_w) + 0;
				int ry = int(crop_y - 0.5 * sn * crop_h + crop_h * sn * yn + 0.5 * crop_h) + 0;
				int rw = int(sn * crop_w);
				int rh= int(sn * crop_h);
				if (rx >= 0 && rx + rw < img.cols && ry >= 0 && ry + rh < img.rows)
				{
					bbox.push_back(cv::Rect(rx, ry, rw, rh));
					scores.push_back(data[1] / 4096.0);
				}
			}
		}
	return HI_SUCCESS;

}
