// NNIE_BOARD.cpp : 定义控制台应用程序的入口点。
//
#include <stdio.h>
#include "FCW_NET.h"
#include "MarsFace.h"
#include <opencv2/opencv.hpp>
#include "nms.h"
//#include"LandMark_Net.h"
#include "FCW_NET.h"
#include "POSE_NET.h"
#include "Classificacion_Net.h"
using namespace cv;
cv::Mat img;
cv::TickMeter time_cnn;
void ttt(char* folder,int mode) {
	std::vector<String> files;
	glob(String(folder), files);
	cv::Mat show, img;
	MarsFace net;
	POSE_NET posenet;
	FCW_NET fcwnet;
	Classificacion_Net smokenet,phonenet;
	char* phonefile = "model//alexphone_no_group_inst.wk";
	char *faceModelFile = "model//face.dat";
	char* lafile = "model//landmark81+5.dat";
	char* posenetfile = "model//pose_inst_96_comp.wk";
	char* fcwnetfile = "model//mobile_ssd_inst_allDepthwiseConv.wk";
	char* smokefile = "model//alexsmoke_inst_nogroup.wk";

	switch (mode)
	{
	case 0://face
		time_cnn.reset();

		net.Init_Detection(faceModelFile);
		net.Init_LandMark(lafile);
		phonenet.CLASSIF_INIT(phonefile);
		smokenet.CLASSIF_INIT(smokefile);

		for (int i = 0; i < files.size(); i++) {
			img = cv::imread(files[i]);
			auto minwh = MIN(img.cols, img.rows);
			img = img(Rect(0, 0, minwh, minwh));
			show = img.clone();
			time_cnn.start();
			auto result = net.Detection_Face(img);
			//time_cnn.stop();

			for (auto re : result) {
				//cv::rectangle(show, re, cv::Scalar(255), 2);
				//cv::Mat fa_img = img(re);
				std::vector<cv::Point2f> stand_pts;
				//time_cnn.start();
				net.LandPoint(img, re, stand_pts);
				//time_cnn.stop();
				float pitch, yaw, roll;
				net.FacePose(stand_pts, roll, yaw, pitch);
				cv::Mat dstimg(Size(256, 256), img.type());
				cv::Mat smoke_img, phone_img[2];
				net.get_cut_face(img, stand_pts, pitch, yaw, smoke_img,phone_img);
				std::vector<float> scores_smoke,scores_phone[2];
				smokenet.Classificacion(smoke_img, scores_smoke);
				for (int p = 0; p < 2; p++) {
					phonenet.Classificacion(phone_img[p], scores_phone[p]);
				}
				time_cnn.stop();
				printf("scores_smoke");
				for (auto score : scores_smoke)
					printf(": %4.2f ", score);
				printf("\n");
				for (int p = 0; p < 2; p++) {
					printf("scores_phone %d",p);
					for (auto score : scores_phone[p])
						printf(" %4.2f ", score);
					printf("\n");

				}
				cv::imshow("fsdf", smoke_img); cv::imshow("l", phone_img[0]); cv::imshow("r", phone_img[1]);
				cv::waitKey(0);
				cv::rectangle(show, re, cv::Scalar(255), 2);
				for (auto pt : stand_pts)
					cv::circle(show, pt, 2, Scalar(0, 255, 255), -1);
			}
			printf("result num=%d\n\n", result.size());
			cv::imwrite("faceout" + files[i], show);
		}
		printf("face using average time =%4.2f ms \n\n", time_cnn.getTimeMilli()/ files.size());
		break;
	case 1://keypoint
		time_cnn.reset();

		posenet.POSE_INIT(posenetfile);
		for (int i = 0; i < files.size(); i++) {
			img = cv::imread(files[i]);
			auto minwh = MIN(img.cols, img.rows);
			img = img(Rect(0, 0, minwh, minwh));

			show = img.clone();
			time_cnn.start();
			posenet.KetPoint(img, 16);
			time_cnn.stop();
			posenet.show_keypoint(show);
			imshow("key", show);
			waitKey(0);
			cv::imwrite("poseout" + files[i], show);
		}
		printf("pose using average time =%4.2f ms \n\n", time_cnn.getTimeMilli() / files.size());
		break;
	case 2://Fcw
		time_cnn.reset();
		fcwnet.FCW_INIT(fcwnetfile);
		for (int i = 0; i < files.size(); i++) {
			img = cv::imread(files[i]);
			show = img.clone();
			time_cnn.start();
			auto fcw_res=fcwnet.Detection(img);
			time_cnn.stop();
			for (auto& fcw_re : fcw_res) {
				cv::rectangle(show, fcw_re._bbox, cv::Scalar(0, 60 * fcw_re.class_id, 200), 2);
				cv::putText(show, std::to_string(fcw_re.box_score), Point(fcw_re._bbox.x, fcw_re._bbox.y), 1, 1.2, cv::Scalar(255, 0, 120), 2);
			}
			cv::imwrite("poseout" + files[i], show);
		}
		printf("fcw using average time =%4.2f ms \n\n", time_cnn.getTimeMilli() / files.size());
		break;
	case 3://class smoke
		time_cnn.reset();
		smokenet.CLASSIF_INIT(smokefile);
		for (int i = 0; i < files.size(); i++) {
			img = cv::imread(files[i]);
			show = img.clone();
			time_cnn.start();
			std::vector<float> scores;
			smokenet.Classificacion(img, scores);
			time_cnn.stop();
			for (auto score : scores)
				printf(" %4.2f ", score);
			printf("\n");
			if (scores[1] > 0.9)
				cv::imwrite("smokeout" + files[i], show);
			else
				cv::imwrite("nosmokeout" + files[i], show);
		}
		printf("smoke using average time =%4.2f ms \n\n", time_cnn.getTimeMilli() / files.size());
		break;
	case 4:
		time_cnn.reset();
		phonenet.CLASSIF_INIT(phonefile);
		for (int i = 0; i < files.size(); i++) {
			img = cv::imread(files[i]);
			show = img.clone();
			time_cnn.start();
			std::vector<float> scores;
			phonenet.Classificacion(img, scores);
			time_cnn.stop();
			for (auto score : scores)
				printf(" %4.2f ", score);
			printf("\n");
			if (scores[1] > 0.9)
				cv::imwrite("lphoneout" + files[i], show);
			else if(scores[4] > 0.9)
				cv::imwrite("lphoneout" + files[i], show);
			else
				cv::imwrite("nophoneout" + files[i], show);
		}
		printf("smoke using average time =%4.2f ms \n\n", time_cnn.getTimeMilli() / files.size());
		break;

	default:
		break;
	}
}
int main(int argc, char *argv[])
{
	MarsFace mf;
	mf.Init_Detection("face.dat");
	cv::Mat im = cv::imread("COCO_val2014_000000000459.jpg");
	im = im(cv::Rect(0, 0, im.cols, im.cols));
	auto outs=mf.Detection_Face(im);
	for (auto out : outs) {
		cv::rectangle(im, out, cv::Scalar(200), 2);
	}
	cv::imshow("f", im);
	cv::waitKey();
	ttt(argv[1],atoi(argv[2]));
	return 0;
}

