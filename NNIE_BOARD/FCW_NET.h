#pragma once
#include "NNIE_Net.h"
/*ssd out
class_id: 1,head 2,hand 3,car 4,bus
*/
struct SSD_out
{
	int class_id;
	float box_score;
	cv::Rect2f _bbox;
};
class FCW_NET :
	protected NNIE_Net
{
public:
	FCW_NET();
	~FCW_NET();
	HI_U16 class_num = 5;
	HI_S32 FCW_INIT(HI_CHAR * pszModelFile);
	std::vector<SSD_out> Detection(const cv::Mat img, float max_socer = 800.f, int max_num =0);
private:
	HI_S32 detection_output_layer(std::vector<SSD_out>& result, float max_socer = 800.f, int max_num=0);
	void set_arsize();
	void init_priod_box();
	HI_U32 all_bbox_num;
	std::vector<std::vector<float> >arsize;
	std::vector<cv::Rect2f> priod_box = std::vector<cv::Rect2f>(1917);//1917=19*19*3+(10*10+5*5+3*3+2*2+1*1)*6
	std::vector< std::vector<int> > classdetection = std::vector< std::vector<int> >(class_num);
	std::vector< std::vector<float> > sorcer = std::vector< std::vector<float> >(class_num);
	std::vector< std::vector<cv::Rect2f> > bboxes = std::vector< std::vector<cv::Rect2f> >(class_num);
	std::vector<int> net_size = { 19,10,5,3,2,1 };
	std::vector<int> minsize = { 60,105,150,195,240,285 };
	std::vector<int> maxsize = { 0,150,195,240,285,300 };
};

