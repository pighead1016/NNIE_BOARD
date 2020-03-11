#include "FCW_NET.h"
#include "nms.h"
FCW_NET::FCW_NET()
{
}
FCW_NET::~FCW_NET()
{
}

HI_S32 FCW_NET::FCW_INIT(HI_CHAR * pszModelFile)
{
	set_arsize();
	init_priod_box();
	return NNIE_NET_INIT(pszModelFile);
}

std::vector<SSD_out> FCW_NET::Detection(const cv::Mat img, float max_socer, int max_num)
{
	SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S input_index = { 0,0 };
	SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S process_index = { 0,0 };
	SVP_FillSrcData_Mat(&input_index, img);
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	std::vector<SSD_out> result;
	detection_output_layer(result, max_socer, max_num);
	return result;
}

HI_S32 FCW_NET::detection_output_layer(std::vector<SSD_out>& result, float max_socer, int max_num)
{
	if (this->stModel.astSeg[0].u16DstNum != 2)
		return HI_FAILURE;
	for (int i = 0; i < class_num; i++)//clean the bbox and sorcer
	{
		sorcer[i].clear();
		bboxes[i].clear();
	}
	result.clear();
	auto prob = this->astSegData[0].astDst[1];
	auto box_gt = this->astSegData[0].astDst[0];
	HI_S32* data = (HI_S32*)(prob.u64VirAddr);
	HI_S32* roi = (HI_S32*)(box_gt.u64VirAddr);
	HI_S32 c = prob.unShape.stWhc.u32Chn;
	HI_S32 h = prob.unShape.stWhc.u32Height;
	HI_S32 w = prob.unShape.stWhc.u32Width;
	int n = 0;
	int m = 0;
	while (n < w*c*h) {
		if (n % class_num == 0) {
			m = n;
			n++;
			continue;
		}
		if (data[n] > class_num) {
			int bboxn = m / class_num;
			cv::Rect2f tmpRect;
			mars::DecodeBoxes(priod_box[bboxn], &roi[m / class_num * 4], tmpRect);
			sorcer[n - m].push_back(data[n]);
			bboxes[n - m].push_back(tmpRect);
		}
		n++;
	}
	for (int i = 1; i < classdetection.size(); i++)
	{
		mars::NMSBoxes(bboxes[i], sorcer[i], max_socer, 0.45f, classdetection[i], 1.0f, 100);
		for (int j = 0; j < classdetection[i].size(); j++)
		{
			result.push_back({ i, sorcer[i][classdetection[i][j]], bboxes[i][classdetection[i][j]] });
		}
	}
	std::sort(result.begin(), result.end(), [](const SSD_out &a, const SSD_out &b) {return a.box_score > b.box_score; });
	if (max_num > 0 && result.size() > max_num)
		result.resize(max_num);
	return HI_SUCCESS;
}

void FCW_NET::set_arsize()
{
	arsize.clear();
	for (auto i : net_size)
	{
		std::vector<float> tmpar;
		if (i == 19)
		{
			tmpar.push_back(sqrt(2));
			tmpar.push_back(sqrt(1.0 / 2));
		}
		else
		{
			tmpar.push_back(sqrt(2));
			tmpar.push_back(sqrt(1.0 / 2));
			tmpar.push_back(sqrt(3));
			tmpar.push_back(sqrt(1.0 / 3));
		}
		arsize.push_back(tmpar);
	}
}

void FCW_NET::init_priod_box()
{
	auto tmp_num = 0;
	for (int c = 0; c < net_size.size(); c++)
	{
		auto bbox_scale = 2;//max and min cube
		if (maxsize[c] != 0)
			bbox_scale = 1;//just min cube
		tmp_num += net_size[c] * net_size[c] * (bbox_scale + arsize[c].size());
	}
	priod_box.resize(tmp_num);
	int index = 0;
	for (int c = 0; c < net_size.size(); c++)
		for (int h = 0; h < net_size[c]; h++)
			for (int w = 0; w < net_size[c]; w++)
			{
				float cx = (w + 0.5) * 300 / net_size[c];
				float cy = (h + 0.5) * 300 / net_size[c];
				priod_box[index++] = cv::Rect2f(cx, cy, float(minsize[c]), float(minsize[c]));
				if (maxsize[c] != 0)
					priod_box[index++] = cv::Rect2f(cx, cy, sqrtf(minsize[c] * maxsize[c]), sqrtf(minsize[c] * maxsize[c]));
				for (auto ar : arsize[c])
					priod_box[index++] = cv::Rect2f(cx, cy, minsize[c] * ar, minsize[c] / ar);
			}
}
