#include "MarsFace.h"
#include "nms.h"


MarsFace::MarsFace()
{
}


MarsFace::~MarsFace()
{
}

HI_S32 MarsFace::Init_Detection(HI_CHAR * mtcnnfile)
{
	HI_S32 s32Ret;
	std::fstream mtcnnf(mtcnnfile, std::ios::binary | std::ios::in);
	mtcnnf.seekg(0, std::ios::beg);
	int psize;
	int hasno;
	int thesize;
	mtcnnf.read((char*)&psize, sizeof(int));
	p_nets.resize(psize);
	for (int i = 0; i < psize; i++)
	{
		for (int n = 0; n < 2; n++) {
			mtcnnf.read((char*)&hasno, sizeof(int));
			if (hasno == 0)
				continue;
			else if (hasno != 1)
				return HI_FAILURE;
			if (n == 0) {//net
				mtcnnf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				mtcnnf.read(buffer, thesize);
				s32Ret=p_nets[i].INIT(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, P_net init Failed!\n");

				delete[] buffer;
			}
			else {//mat
				mtcnnf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				mtcnnf.read(buffer, thesize);
				s32Ret=p_nets[i].Read_weight_bias(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, P_weight_bias init Failed!\n");

			}
		}
	}
	int rosize;
	mtcnnf.read((char*)&rosize, sizeof(int));
	o_nets.resize(rosize);
	for (int i = 0; i < rosize; i++)
	{
		for (int n = 0; n < 2; n++) {
			mtcnnf.read((char*)&hasno, sizeof(int));
//printf("i=%d n=%d has no %d\n",i,n,hasno);
			if (hasno == 0)
				continue;
			else if (hasno != 1)
				return HI_FAILURE;
			if (n == 0) {//net
				mtcnnf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				mtcnnf.read(buffer, thesize);
				s32Ret=o_nets[i].INIT(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, R_net init Failed!\n");

				delete[] buffer;
			}
			else {//mat
				mtcnnf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				mtcnnf.read(buffer, thesize);
				s32Ret=o_nets[i].Read_weight_bias(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, R_weight_bias init Failed!\n");

			}
		}
	}
	mtcnnf.close();

	return s32Ret;
}

HI_S32 MarsFace::Init_LandMark(HI_CHAR * landmarkfile)
{
	HI_S32 s32Ret;
	std::fstream ldf(landmarkfile, std::ios::binary | std::ios::in);
	ldf.seekg(0, SEEK_SET);
	int ldsize;
	int hasno;
	int thesize;
	ldf.read((char*)&ldsize, sizeof(int));
	landmark_net.resize(ldsize);
	for (int i = 0; i < ldsize; i++)
	{
		for (int n = 0; n < 2; n++) {
			ldf.read((char*)&hasno, sizeof(int));
			if (hasno == 0)
				continue;
			SAMPLE_SVP_CHECK_EXPR_RET(1 != hasno, hasno, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, Landmark_file Failed!\n");
			if (n == 0) {//net
				ldf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				ldf.read(buffer, thesize);
				s32Ret = landmark_net[i].INIT(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, Landmark_net init Failed!\n");
				delete[] buffer;
			}
			else {//mat
				ldf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				ldf.read(buffer, thesize);
				s32Ret = landmark_net[i].Read_weight_bias(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, Landmark bias Failed!\n");
			}
		}
	}
	ldf.close();
	return s32Ret;
}

std::vector<cv::Rect> MarsFace::Detection_Face(cv::Mat & cubeimg)
{
	//cv::Mat tmp=cubeimg.clone();
	std::vector<cv::Rect> bboxes;
	std::vector<float> scores;
	for (int i = 0; i < 3; i++){
		p_nets[i].BBoxes(cubeimg, bboxes, scores, thres[0]);
		//printf("%d has bboxes bf nms\n",bboxes.size());	
	}
	std::vector<int> indices;
	//printf("%d has bboxes afnms\n",indices.size());
	mars::NMSBoxes(bboxes, scores, thres[0], nmsthres[0], indices, 1.0, 50);

	for (int time = 3 - o_nets.size(); time < 3; time++) {//loop times = rnet.size()
		if (indices.size() == 0)
			break;
		std::vector<cv::Rect> inbboxes;
		inbboxes.clear();
		for (auto i : indices) {
			inbboxes.push_back(bboxes[i]);
			//cv::rectangle(tmp,bboxes[i],cv::Scalar(255), 2);
		}
		//cv::imwrite("../ld.bmp",tmp);
		bboxes.clear();
		scores.clear();
		for (int i = 0; i < indices.size(); i++) {
			cv::Rect rect;
			float socer;
			std::vector<float> t;
			cv::Mat timg = cubeimg(inbboxes[i]).clone();
			o_nets[time - 1].Confirm_bboxes(cubeimg, inbboxes[i], rect, socer, thres[time], 0, 2 - time);
			if (socer > thres[time]) {
				bboxes.push_back(rect);
				scores.push_back(socer);
			}
		}
		mars::NMSBoxes(bboxes, scores, thres[time], nmsthres[time], indices, 1.0, 10);
	}
	std::vector<cv::Rect>  Result(indices.size());
	for (int i = 0; i < indices.size(); i++) {
		Result[i] = bboxes[indices[i]];
	}
	return Result;

}

int MarsFace::LandPoint(cv::Mat & img, cv::Rect & bbox, std::vector<cv::Point2f>& pts, LandmarkType nettype)
{
	HI_S32 s32Ret;
	cv::Rect extend_bbox;
	float extend_pad = 0.1;

	float extend_bbox_w = extend_pad*bbox.width;
	float img_extend_x = img.cols*extend_pad;
	float img_extend_y = img.rows*extend_pad;
	float deta_y = (bbox.height - bbox.width) / 2.0;
	extend_bbox.x = bbox.x - extend_bbox_w + img_extend_x;
	extend_bbox.y = bbox.y - extend_bbox_w + img_extend_y + deta_y;
	extend_bbox.width = bbox.width + 2 * extend_bbox_w;
	extend_bbox.height = bbox.width + 2 * extend_bbox_w;
	cv::Mat extend_img = cv::Mat::zeros(img.rows*(extend_pad * 2 + 1), img.cols*(extend_pad * 2 + 1), img.type());
	if (extend_bbox.x < 0 || extend_bbox.y < 0 || extend_bbox.x + extend_bbox.width >= extend_img.cols || extend_bbox.y + extend_bbox.height >= extend_img.rows)
		return HI_FAILURE;
	img.copyTo(extend_img(cv::Rect(img_extend_x, img_extend_y, img.cols, img.rows)));
	pts.resize(0);
	cv::Mat tmp = extend_img(extend_bbox);

	s32Ret=landmark_net[nettype].stand_LandMark_point(tmp, pts);
	for (auto& pt : pts) {
		pt.x *= extend_bbox.width;
		pt.y *= extend_bbox.width;
		//pt -= cv::Point2f(img_extend_x, img_extend_y);

		pt += cv::Point2f(bbox.x - extend_bbox_w, bbox.y + deta_y - extend_bbox_w);
	}
	return s32Ret;
}
