#include "common_alignment.h"

#include <memory>
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>
#include <cfloat>
#include <cstring>
#include <cmath>
static bool caculate_final_points(
	const std::vector<cv::Point2f> points,
	const double *transformation,
	std::vector<cv::Point2f>& final_points);

template <typename T>
void CopyData(T *dst, const T *src, size_t _count)
{
#if _MSC_VER >= 1600
	memcpy_s(dst, sizeof(T) * _count, src, sizeof(T) * _count);
#else
	memcpy(dst, src, sizeof(T) * _count);
#endif
}



/**
 * \brief
 * \param crop_width Crop
 * \param crop_height Crop
 * \param points format {(x1, y1), (x2, y2), ...}
 * \param points_num
 * \param mean_shape
 * \param mean_shape_width
 * \param mean_shape_height
 * \param transformation size: 
 * \return
 */
bool transformation_maker(
	const std::vector <cv::Point2f> pts,
	double *transformation, int crop_width , int crop_height, float pad_left,std::vector <cv::Point2f> & final_points,float *A)
{
	cv::Point2f mean_shape[5] = { cv::Point2f(89.3095, 72.9025), cv::Point2f(169.3095, 72.9025),
		cv::Point2f(127.8949, 127.0441), cv::Point2f(96.8796, 184.8907),cv::Point2f(159.1065, 184.7601) };
	const static int points_num = 5;
	std::vector<cv::Point2f> points;
	points.clear();
	if (pts.size() == points_num)
		points = pts;
	else if (pts.size() == 81) {
		points.push_back(pts[0]);
		points.push_back(pts[9]);
		points.push_back(pts[34]);
		points.push_back(pts[46]);
		points.push_back(pts[47]);
	}

	float mean_shape_width = 256, mean_shape_height = 256;
	cv::Point2f std_points[points_num];
	for (int i = 0; i < points_num; ++i)
	{
		std_points[i] = mean_shape[i] * crop_width / mean_shape_width + cv::Point2f(crop_width*pad_left,0.0f);
	}
	cv::TickMeter ti;
	
	float sum_x = 0, sum_y = 0;
	float sum_u = 0, sum_v = 0;
	float sum_xx_yy = 0;
	float sum_ux_vy = 0;
	float sum_vx__uy = 0;
	float sum_uu_vv = 0;



	double sum_xf = 0.0, sum_yf = 0.0;
	double sum_uf = 0.0, sum_vf = 0.0;
	double sum_ux_vyf = 0.0;
	double sum_vx__uyf = 0.0;
	double sum_uu_vvf = 0.0;
	ti.start();
	for (int c = 0; c < points_num; ++c)
	{
		sum_xf += std_points[c].x;
		sum_yf += std_points[c].y;
		sum_uf += points[c].x;
		sum_vf += points[c].y;
		//sum_xx_yy += std_points[c].dot(std_points[c]);
		sum_ux_vyf += std_points[c].dot(points[c]);
		sum_uu_vvf += points[c].dot(points[c]);
		sum_vx__uyf += points[c].y * std_points[c].x - points[c].x * std_points[c].y;
	}
	//if (sum_uu_vvf <= FLT_EPSILON) return false;
	double qf = sum_xf - sum_uf * sum_ux_vyf / sum_uu_vvf
		- sum_vf * sum_vx__uyf / sum_uu_vvf;
	double pf = sum_yf - sum_vf * sum_ux_vyf / sum_uu_vvf
		+ sum_uf * sum_vx__uyf / sum_uu_vvf;

	double rf = points_num - (sum_uf * sum_uf + sum_vf * sum_vf) / sum_uu_vvf;

	//if (!(rf > FLT_EPSILON || rf < -FLT_EPSILON)) return false;

	double af = (sum_ux_vyf - sum_uf * qf / rf - sum_vf * pf / rf) / sum_uu_vvf;

	double bf = (sum_vx__uyf - sum_vf * qf / rf + sum_uf * pf / rf) / sum_uu_vvf;

	double cf = qf / rf;

	double df = pf / rf;
	/*cv::Mat X=(cv::Mat_<float>(4, 4)<< sum_uu_vvf,0, sum_uf, sum_vf,
	0, sum_uu_vvf, sum_vf,-sum_uf,
	sum_uf, sum_vf,5,0,
	sum_vf,-sum_uf,0,5), Y=(cv::Mat_<float>(4, 1)<< sum_ux_vyf, sum_vx__uyf, sum_xf, sum_yf), AA(4, 1, CV_32FC1, A);


	for (int i = 0; i < 10; i++) {
	float *xp = (float*)X.ptr(i);
	float *yp = (float*)Y.ptr(i);
	if (i < 5) {
	xp[0] = points[i].x;
	xp[1] = points[i].y;
	xp[2] = 1;
	xp[3] = 0;
	yp[0] = std_points[i].x;
	}
	else {
	xp[0] = points[i - 5].y;
	xp[1] = -points[i - 5].x;
	xp[2] = 0;
	xp[3] = 1;
	yp[0] = std_points[i - 5].y;
	}
	}
	cv::solve(X, Y, AA, cv::DECOMP_LU);*/ti.stop();
	
	std::cout << ti.getTimeMilli() << std::endl;
	ti.reset();
	ti.start();
	for (int c = 0; c < points_num; ++c)
	{
		sum_x += std_points[c].x;
		sum_y += std_points[c].y;
		sum_u += points[c].x;
		sum_v += points[c].y;
		sum_xx_yy += std_points[c].dot(std_points[c]);
		sum_ux_vy += std_points[c].dot(points[c]);
		//sum_uu_vv+= points[c].dot(points[c]);
		sum_vx__uy += points[c].y * std_points[c].x - points[c].x * std_points[c].y;
	}

	//A = (double*)AA.data;
	//if (sum_xx_yy <= FLT_EPSILON) return false;
	float q = sum_u - sum_x * sum_ux_vy / sum_xx_yy
		+ sum_y * sum_vx__uy / sum_xx_yy;
	float p = sum_v - sum_y * sum_ux_vy / sum_xx_yy
		- sum_x * sum_vx__uy / sum_xx_yy;

	float r = points_num - (sum_x * sum_x + sum_y * sum_y) / sum_xx_yy;

	//if (!(r > FLT_EPSILON || r < -FLT_EPSILON)) return false;

	float a = (sum_ux_vy - sum_x * q / r - sum_y * p / r) / sum_xx_yy;

	float b = (sum_vx__uy + sum_y * q / r - sum_x * p / r) / sum_xx_yy;

	float c = q / r;

	float d = p / r;
ti.stop();
	transformation[0] = transformation[4] = a;
	transformation[1] = -b;
	transformation[3] = b;
	transformation[2] = c;
	transformation[5] = d;
	




	
	A[0] = af, A[1] = bf,A[2]=cf,A[3]=df;
	
	std::cout << ti.getTimeMilli() << std::endl;



	bool check3=true;
	if (final_points != std::vector<cv::Point2f>())
	{
		check3 = caculate_final_points(pts, transformation,final_points);
	}
	if (!check3) return false;
	return true;
}

static inline double Cubic(double x)
{
	double ax = std::fabs(x), ax2, ax3;
	ax2 = ax * ax;
	ax3 = ax2 * ax;
	if (ax <= 1) return 1.5 * ax3 - 2.5 * ax2 + 1;
	if (ax <= 2) return -0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2;
	return 0;
}

static inline void Norm(std::vector<double> &weights)
{
	double sum = 0;
	for (double w : weights) sum += w;;
	for (double &w : weights) w /= sum;
}

static void near_sampling(
	const uint8_t *image_data, int image_width, int image_height, int image_channels,
	int x, int y, uint8_t *pixel)
{
	if (x < 0) x = 0;
	if (x >= image_height) x = image_height - 1;
	if (y < 0) y = 0;
	if (y >= image_width) y = image_width - 1;
	int offset = (x * image_width + y) * image_channels;
	for (int c = 0; c < image_channels; ++c)
	{
		pixel[c] = image_data[offset + c];
	}
}

#define BICUBIC_KERNEL 4

static void sampling(
	const uint8_t *image_data, int image_width, int image_height, int image_channels,
	double scale,
	double x, double y, uint8_t *pixel,
	std::vector<double> &weights_x, std::vector<double> &weights_y,
	std::vector<int> &indices_x, std::vector<int> &indices_y,
	SAMPLING_TYPE type = LINEAR,
	PADDING_TYPE ptype = ZERO_PADDING)
{
	if (type == LINEAR)
	{
		// bilinear subsampling
		int ux = static_cast<int>(std::floor(x)), uy = static_cast<int>(std::floor(y));
		if (ux >= 0 && ux < image_height - 1 && uy >= 0 && uy < image_width - 1)
		{
			double cof_x = x - ux;
			double cof_y = y - uy;
			for (int c = 0; c < image_channels; ++c)
			{
				double ans = 0;
				int offset = (ux * image_width + uy) * image_channels + c;
				ans = (1 - cof_y) * image_data[offset] + cof_y * image_data[offset + image_channels];
				ans = (1 - cof_x) * ans + cof_x * ((1 - cof_y) * image_data[offset + image_width * image_channels]
					+ cof_y * image_data[offset + image_width * image_channels + image_channels]);
				pixel[c] = static_cast<uint8_t>(std::max<double>(0.0f, std::min<double>(255.0f, ans)));
			}
		}else{
			switch (ptype)
			{
			default:
				memset(pixel, 0, sizeof(uint8_t) * image_channels);
				break;
			case NEAREST_PADDING:
				near_sampling(image_data, image_width, image_height, image_channels, ux, uy, pixel);
				break;
			}
		}
	}
	else
		if (type == BICUBIC)
		{
			// bicubic subsampling
			if (x >= 0 && x < image_height && y >= 0 && y < image_width)
			{
				scale = std::min<double>(scale, double(1.0));
				double kernel_width = std::max<double>(BICUBIC_KERNEL, BICUBIC_KERNEL / scale); // bicubic kernel width
				//std::vector<double> weights_x, weights_y;
				//std::vector<int>  indices_x, indices_y;
				//weights_x.reserve(kernel_width), indices_x.reserve(kernel_width);
				//weights_y.reserve(kernel_width), indices_y.reserve(kernel_width);
				weights_x.clear();
				indices_x.clear();
				weights_y.clear();
				indices_y.clear();
				// get indices and weight along x axis
				int ux_left = std::max<int>(0, static_cast<int>(std::ceil(x - kernel_width / 2)));
				int ux_right = std::min<int>(image_height - 1, static_cast<int>(std::floor(x + kernel_width / 2)));
				for (int ux = ux_left; ux <= ux_right; ++ux)
				{
					double weight = Cubic((x - ux) * scale);
					// if (weight == 0) continue;
					indices_x.push_back(ux);
					weights_x.push_back(weight);
				}
				// get indices and weight along y axis
				int uy_left = std::max<int>(0, static_cast<int>(std::ceil(y - kernel_width / 2)));
				int uy_right = std::min<int>(image_width - 1, static_cast<int>(std::floor(y + kernel_width / 2)));
				for (int uy = uy_left; uy <= uy_right; ++uy)
				{
					double weight = Cubic((y - uy) * scale);
					// if (weight == 0) continue;
					indices_y.push_back(uy);
					weights_y.push_back(weight);
				}
				// normalize the weights
				Norm(weights_x);
				Norm(weights_y);
				size_t lx = weights_x.size(), ly = weights_y.size();
				for (int c = 0; c < image_channels; ++c)
				{
					double ans = 0;
					double val = 0;
					for (size_t i = 0; i < lx; ++i)
					{
						val = 0;
						int offset = indices_x[i] * image_width * image_channels;
						for (size_t j = 0; j < ly; ++j)
						{
							val += image_data[offset + indices_y[j] * image_channels + c] * weights_y[j];
						}
						ans += val * weights_x[i];
					}
					pixel[c] = static_cast<uint8_t>(std::max<double>(0.0f, std::min<double>(255.0f, ans)));
				}
			}
			else
			{
				switch (ptype)
				{
				default:
					memset(pixel, 0, sizeof(uint8_t) * image_channels);
					break;
				case NEAREST_PADDING:
					near_sampling(image_data, image_width, image_height, image_channels, int(x), int(y), pixel);
					break;
				}
			}
		}
		else
		{
			int ux = static_cast<int>(x + 0.5), uy = static_cast<int>(y + 0.5);
			if (ux >= 0 && ux < image_height && uy >= 0 && uy < image_width)
			{
				int offset = (ux * image_width + uy) * image_channels;
				CopyData(pixel, &image_data[offset], image_channels);
			}
			else
			{
				switch (ptype)
				{
				default:
					memset(pixel, 0, sizeof(uint8_t) * image_channels);
					break;
				case NEAREST_PADDING:
					near_sampling(image_data, image_width, image_height, image_channels, ux, uy, pixel);
					break;
				}
			}
		}
}

static bool spatial_transform(
	const cv::Mat img, cv::Mat &dstimg,
	const double *transformation,
	SAMPLING_TYPE type = LINEAR,
	PADDING_TYPE dtype = ZERO_PADDING)
{
	// const double *theta_data = transformation;
	// int src_w = image_width;
	// int src_h = image_height;
	int channels = img.channels();
	int dst_h = dstimg.cols;
	int dst_w = dstimg.rows;
	uint8_t *output_data = (uint8_t *)dstimg.data;
	//bool normalized_tform_ = false;   // @todo it does not work now
	std::vector<double> weights_x, weights_y;
	std::vector<int>  indices_x, indices_y;

	double scale = std::sqrt(transformation[0] * transformation[0] + transformation[3] * transformation[3]);
	for (int x = 0; x < dst_h; ++x)
	{
		for (int y = 0; y < dst_w; ++y)
		{
			// Convet the point into crop axis
			// Get the source position of each point on the destination feature map.
			double src_y = transformation[0] * y + transformation[1] * x + transformation[2];
			double src_x = transformation[3] * y + transformation[4] * x + transformation[5];
			uint8_t *current_channel_data = &output_data[x * dst_w * channels + y * channels];
			sampling(img.data, img.cols, img.rows, channels, 1.0 / scale,
				src_x, src_y, current_channel_data,
				weights_x, weights_y, indices_x, indices_y,
				type,dtype);
		}
	}
	return true;
}

static bool caculate_final_points(
	const std::vector<cv::Point2f> points,
	const double *transformation,
	std::vector<cv::Point2f>& final_points)
{
	final_points.resize(points.size());
	const double *t = transformation;
	double t3t1_t0t4 = t[3] * t[1] - t[0] * t[4];
	if (t3t1_t0t4 < FLT_EPSILON && t3t1_t0t4 > -FLT_EPSILON) t3t1_t0t4 = FLT_EPSILON * 2;
	for (int i = 0; i < points.size(); ++i)
	{
		float x = points[i].x;
		float y = points[i].y;
		double fy = ((t[3] * x - t[0] * y) - (t[3] * t[2] - t[0] * t[5])) / t3t1_t0t4;
		double fx = ((t[1] * y - t[4] * x) - (t[1] * t[5] - t[4] * t[2])) / t3t1_t0t4;
		final_points[i].x = static_cast<float>(fx);
		final_points[i].y = static_cast<float>(fy);
	}
	return true;
}

bool face_crop_core_ex(
	const cv::Mat img, cv::Mat& dstimg, 
	const std::vector<cv::Point2f> points, std::vector<cv::Point2f>& final_points,
	SAMPLING_TYPE type,PADDING_TYPE ptype)
{
	const int TFORM_SIZE = 6;
	//std::unique_ptr<double[]> transformation(new double[TFORM_SIZE]);
	double transformation[TFORM_SIZE];
	bool check1 = transformation_maker(points, transformation, dstimg.cols, dstimg.rows);
	//std::cout << transformation << std::endl;
	if (!check1) return false;
	bool check2 = spatial_transform(img, dstimg, transformation,
		type,
		ptype);
	if (!check2) return false;
	//bool check3 = true;
	//if (final_points != std::vector<cv::Point2f>())
	//{
	//	check3 = caculate_final_points(points, transformation,final_points);
	//}
	//if (!check3) return false;
	return true;
}
/*
bool face_crop_core(
	const cv::Mat img, cv::Mat& dstimg,
	const std::vector<cv::Point2f> points,
	const float *mean_shape, int mean_shape_width, int mean_shape_height,
	int pad_top, int pad_bottom, int pad_left, int pad_right,
	float *final_points,
	SAMPLING_TYPE type)
{
	return face_crop_core_ex(
		img, dstimg,
		points,
		mean_shape, mean_shape_width, mean_shape_height,
		pad_top, pad_bottom, pad_left, pad_right,
		final_points,
		type,
		ZERO_PADDING);
}*/
