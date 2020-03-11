// NNIE_BOARD.cpp : 定义控制台应用程序的入口点。
//
#include <stdio.h>
#include "FCW_NET.h"
#include "MarsFace.h"
#include <opencv2/opencv.hpp>
#include "nms.h"
#include"LandMark_Net.h"

using namespace cv;
cv::Mat img;
cv::TickMeter time_cnn;
void ttt(cv::Mat &img0) {
	char *pszModelFile = "model//face.dat";
	MarsFace net;
	net.Init_Detection(pszModelFile);
	net.Init_LandMark("model//landmark81+5.dat");
	//LandMark_Net ld;
	//ld.INIT("model//LD5_nnie_ins.wk", "model//landmark_5_inner4.dat");
	cv::Mat show, img;
	cv::Mat img1 = cv::imread("L.jpg");
	img1 = img1(Rect(0, 0, img1.cols, img1.cols));
	for (int i = 1; i < 2; i++) {
		if (i % 2)
			img = img1;
		else
			img = img0;
		show = img.clone();
		time_cnn.start();
		auto result = net.Detection_Face(img);
		time_cnn.stop();
		printf("result num=%d\n\n using time =%4.2f ms \n\n", result.size(), time_cnn.getTimeMilli());

		//cv::resize(show, show, cv::Size(300, 300));
		for (auto re : result) {
			std::cout << re << std::endl;
			cv::rectangle(show, re, cv::Scalar(255), 2);
			//cv::Mat fa_img = img(re);
			std::vector<cv::Point2f> stand_pts;
			net.LandPoint(img, re, stand_pts);
			for (auto pt : stand_pts)
				cv::circle(show, pt, 2, Scalar(0, 255, 255), -1);
		}
		//img = cv::imread("vlcsnap-2020-01-19-09h43m12s761.png");
		//cv::imshow("fadfa", show);
		cv::imwrite("fcwn" + std::to_string(i) + "ssd.jpg", show);
		//cv::waitKey();
	}
}
int main(int argc, char *argv[])
{
	//LandMark_Net ld;
	//ld.INIT("model//LD5_nnie_ins.wk","model//landmark_5_inner4.dat");
	img = cv::imread("vlcsnap-2020-01-19-09h43m12s761.png");
	ttt(img);
	return 0;
}

