#include "MarsFace.h"
#include "nms.h"
#include"common_alignment.h"
static double dott(const cv::Point2f &lhs, const cv::Point2f &rhs) {
	auto dx = lhs.x - rhs.x;
	auto dy = lhs.y - rhs.y;
	return std::sqrt(dx * dx + dy * dy);
}
/**
* line for ax + by + c = 0
*/
class Line {
public:
	inline Line() = default;
	inline Line(double a, double b, double c)
		: a(a), b(b), c(c) {}

	inline Line(const cv::Point2f &a, const cv::Point2f &b) {
		auto x1 = a.x;
		auto y1 = a.y;
		auto x2 = b.x;
		auto y2 = b.y;
		// for (y2-y1)x-(x2-x1)y-x1(y2-y1)+y1(x2-x1)=0
		this->a = y2 - y1;
		this->b = x1 - x2;
		this->c = y1 * (x2 - x1) - x1 * (y2 - y1);
	}

	double distance(const cv::Point2f &p) const {
		return /*std::fabs*/(a * p.x + b * p.y + c) / std::sqrt(a * a + b * b);
	}

	static bool near_zero(double f) {
		return f <= DBL_EPSILON && -f <= DBL_EPSILON;
	}

	cv::Point2f projection(const cv::Point2f &p) const {
		if (near_zero(a)) {
			cv::Point2f result;
			result.x = p.x;
			result.y = -c / b;
			return  result;
		}
		if (near_zero(b)) {
			cv::Point2f result;
			result.x = -c / a;
			result.y = p.y;
			return result;
		}
		// y = kx + b  <==>  ax + by + c = 0
		auto k = -a / b;
		cv::Point2f o = { 0, float(-c / b) };
		cv::Point2f project = { 0 };
		project.x = (float)((p.x / k + p.y - o.y) / (1 / k + k));
		project.y = (float)(-1 / k * (project.x - p.x) + p.y);
		return project;
	}

	double a = 0;
	double b = 0;
	double c = 0;
};

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
				s32Ret = p_nets[i].INIT(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, P_net init Failed!\n");

				delete[] buffer;
			}
			else {//mat
				mtcnnf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				mtcnnf.read(buffer, thesize);
				s32Ret = p_nets[i].Read_weight_bias(buffer, thesize);
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
				s32Ret = o_nets[i].INIT(buffer, thesize);
				SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, R_net init Failed!\n");

				delete[] buffer;
			}
			else {//mat
				mtcnnf.read((char*)&thesize, sizeof(int));
				char *buffer = new char[thesize];
				mtcnnf.read(buffer, thesize);
				s32Ret = o_nets[i].Read_weight_bias(buffer, thesize);
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
	for (int i = 0; i < 3; i++) {
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

int MarsFace::Tracing(cv::Mat & cubeimg, std::vector<cv::Rect>& now_bboxex, std::vector<int>& now_index)
{
	now_bboxex = this->Detection_Face(cubeimg);
	now_index.resize(now_bboxex.size());
	float max_iou = 0;
	int max_iou_num = -1;
	for (auto i = 0; i < now_bboxex.size(); i++) {
		for (auto j = 0; j < trace_faces.size(); j++) {
			float tmp_iou = mars::rectOverlap(trace_faces[j], now_bboxex[i]);
			if (tmp_iou > max_iou) {
				max_iou = tmp_iou;
				max_iou_num = j;
			}
			if (max_iou > min_score) {//has this  face
				now_index[i] = trace_face_num[max_iou_num];
				if (max_iou < max_score) {
					now_bboxex[i].x = (now_bboxex[i].x + trace_faces[max_iou_num].x) / 2;
					now_bboxex[i].y = (now_bboxex[i].y + trace_faces[max_iou_num].y) / 2;
					now_bboxex[i].width = (now_bboxex[i].width + trace_faces[max_iou_num].width) / 2;
					now_bboxex[i].height = (now_bboxex[i].height + trace_faces[max_iou_num].height) / 2;
				}
			}
			else {
				now_index[i] = max_PID;
				max_PID++;
			}
		}
	}
	trace_faces = now_bboxex;
	trace_face_num = now_index;
	return now_bboxex.size();
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

	s32Ret = landmark_net[nettype].stand_LandMark_point(tmp, pts);
	for (auto& pt : pts) {
		pt.x *= extend_bbox.width;
		pt.y *= extend_bbox.width;
		//pt -= cv::Point2f(img_extend_x, img_extend_y);

		pt += cv::Point2f(bbox.x - extend_bbox_w, bbox.y + deta_y - extend_bbox_w);
	}
	return s32Ret;
}
float nop;
int MarsFace::FacePose(const std::vector<cv::Point2f>& pts, float & roll, float & yaw, float & pitch)
{
	static const float nose_center = 0.5f;
	// static const float roll0 = 1 / 6.0f;
	// static const float yaw0 = 0.2f;
	// static const float pitch0 = 0.2f;
	std::vector<cv::Point2f> points;
	points.clear();

	if (pts.size() == 5)
		points = pts;
	else if (pts.size() == 81) {
		points.push_back(pts[0]);
		points.push_back(pts[9]);
		points.push_back(pts[34]);
		points.push_back(pts[46]);
		points.push_back(pts[47]);
	}
	else {
		SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, pts num is wrong Failed!\n");
		return HI_FAILURE;
	}
	auto point_center_eye = (points[0] + points[1]) / 2;
	auto point_center_mouth = (points[3] + points[4]) / 2;

	Line line_eye_mouth(point_center_eye, point_center_mouth);

	auto vector_left2right = points[1] - points[0];

	auto rad = atan2(vector_left2right.y, vector_left2right.x);
	auto angle = rad * 180 / CV_PI;

	auto roll_dist = /*fabs(*/angle/*)*/ / 180;

	auto raw_yaw_dist = line_eye_mouth.distance(points[2]);
	auto yaw_dist = raw_yaw_dist / dott(points[0], points[1]);

	auto point_suppose_projection = point_center_eye * nose_center + point_center_mouth * (1 - nose_center);
	auto point_projection = line_eye_mouth.projection(points[2]);
	auto raw_pitch_dist = dott(point_projection, point_suppose_projection);
	auto pitch_dist = raw_pitch_dist / dott(point_center_eye, point_center_mouth);
	nop = pitch_dist;

	roll = roll_dist;
	yaw = yaw_dist;
	pitch = pitch_dist;
	return 0;
}

int MarsFace::crop_face(const cv::Mat & img, const std::vector<cv::Point2f>& pts, cv::Mat & crop_img)
{
	face_crop_core_ex(img, crop_img, pts);
	return 0;
}
bool check_rect(const cv::Mat& img,const cv::Rect& bbox) {
	if (bbox.x < 0 || bbox.y < 0 || bbox.y + bbox.height >= img.rows || bbox.x + bbox.width >= img.cols)
		return false;
	return true;
}
void cut_bbox(const cv::Mat & img,cv::Rect &bbox) {
	if (bbox.x < 0 || bbox.y < 0 || bbox.y + bbox.height >= img.rows || bbox.x + bbox.width >= img.cols) {
		cv::Rect tmp;
		tmp.x = bbox.x > 0 ? bbox.x : 0;
		tmp.y = bbox.y > 0 ? bbox.y : 0;
		tmp.height = bbox.y + bbox.height < img.rows ? bbox.y + bbox.height - tmp.y : img.rows - tmp.y;
		tmp.width = bbox.x + bbox.width < img.cols ? bbox.x + bbox.width - tmp.y : img.cols - tmp.x;
		bbox = tmp;
		return;
	}
	//return;
}
int MarsFace::get_cut_face(const cv::Mat & img, const std::vector<cv::Point2f>& pts,float pitch, float yaw,
	cv::Mat& smoke_img, cv::Mat* phone_img)
{
	static const int TFORM_SIZE = 9;
	SAMPLE_SVP_CHECK_EXPR_RET(pts.size() != 5 && pts.size() != 81, pts.size(), SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, pts num is wrong Failed!\n");
	double transformation[TFORM_SIZE];
	std::vector<cv::Point2f> rotate_p(pts.size());
	float t[4];
	transformation_maker(pts, transformation, 100, 100, 0.75,rotate_p,t);
	transformation[6] = transformation[7] = 0;
	transformation[8] = 1;
	cv::Mat crop,crop2,tm=(cv::Mat_<float>(2,3)<<t[0],t[1],t[2],-t[1],t[0],t[3]);
	cv::Mat tr(3, 3, CV_64FC1, transformation);
	cv::invert(tr, tr);
	cv::warpAffine(img, crop, tr.rowRange(0, 2), cv::Size(250, 200));
	cv::warpAffine(img, crop2, tm, cv::Size(250, 200));
	cv::imshow("crop", crop);
	cv::imshow("crop2", crop2);
	cv::waitKey();
	//face_crop_core_ex(img, crop_img, transformation, pts );
		/*std::vector<cv::Point2f> rotate_p(81);
		for (auto p_num=0;p_num<pts.size(); p_num++)
		{
			cv::Mat p = (cv::Mat_<double>(3, 1) << pts[p_num].x, pts[p_num].y, 1.0);
			cv::Mat rop = tr*p;//cv::circle(frame, cv::Point(point.x, point.y), 2, CV_RGB(128, 255, 128), -1);
			rotate_p[p_num].x = rop.at<double>(0);
			rotate_p[p_num].y = rop.at<double>(1);
			p_num++;
		}*/

	auto mousec = (rotate_p[46].y + rotate_p[47].y) / 2;
	auto eyec = (rotate_p[0].y + rotate_p[9].y) / 2;
	float nose_y = (pitch) * 3 * (mousec - eyec);
	cv::Rect smone_rect;
	float phone_w = rotate_p[63].x - rotate_p[62].x;
	float phone_h = (mousec - eyec)*1.2;
	float w = 0.8;
	float ww = phone_w*(w + abs(yaw) / 2);
	float ww1 = phone_w*(w + abs(yaw) / 4);
	cv::Rect phone_temp[2];
	if (yaw > 0) {
		phone_temp[1] = cv::Rect(rotate_p[63].x, mousec - nose_y, ww1, phone_h * 2);//right
		phone_temp[0] = cv::Rect(rotate_p[62].x + (yaw - w)*phone_w, mousec - nose_y - 0.8*phone_h, ww, phone_h * 2);
	}
	else {
		phone_temp[1] = cv::Rect(rotate_p[63].x + yaw*phone_w, mousec - nose_y - phone_h*0.5, ww, phone_h * 2);
		phone_temp[0] = cv::Rect(rotate_p[62].x - w*phone_w, mousec - nose_y - phone_h*0.8, ww1, phone_h * 2);
	}
	float mouse_w = rotate_p[47].x - rotate_p[46].x;
	float mouse_h = mouse_w* 1.5;
	cv::Rect rect_mouse;
	rect_mouse.x = rotate_p[46].x - mouse_w / 2;
	rect_mouse.y = rotate_p[46].y - mouse_h*0.4;
	rect_mouse.width = mouse_w * 2;
	rect_mouse.height = mouse_h;
	cut_bbox(crop, rect_mouse);

	smoke_img = crop(rect_mouse);
	for (int i = 0; i < 2; i++) {
		cut_bbox(crop, phone_temp[i]);
		phone_img[i] = crop(phone_temp[i]);
	}

	//cv::rectangle(crop, rect_mouse, cv::Scalar(200), 2);
	//cv::rectangle(crop, phone_temp[0], cv::Scalar(200), 2);
	//cv::rectangle(crop, phone_temp[1], cv::Scalar(200), 2);

	return HI_SUCCESS;
}
