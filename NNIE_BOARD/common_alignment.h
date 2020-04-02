#ifndef _SEETA_COMMON_ALIGNMENT_H
#define _SEETA_COMMON_ALIGNMENT_H
#include <cstdint>
#include<opencv2/opencv.hpp>
/**
 * \brief sample type
 */
enum SAMPLING_TYPE
{
    LINEAR, ///< 
    BICUBIC ///< 
};

/**
 * \brief type of padding value if sample out of image
 */
enum PADDING_TYPE
{
    ZERO_PADDING,           ///< 0 padding
    NEAREST_PADDING,        ///< padding with value of nearest pixel
};

/**
 * \brief Crop face
 * \param [in] image_data input image, format like height * width * channels
 * \param [in] image_width image width
 * \param [in] image_height image height
 * \param [in] image_channels image channels
 * \param [in] crop_data output [crop_width + pad_left + pad_right, crop_height + pad_top + pad_bottom, image_channels]
 * \param [in] crop_width output crop face width, based on meanshape, \b not output image size, infected by pad value
 * \param [in] crop_height output crop face height, based on meanshape, \b not output image size, infected by pad value
 * \param [in] points face landmark, format {(x1, y1), (x2, y2), ...}
 * \param [in] points_num number of landmark
 * \param [in] mean_shape meanshape, fomat {(x1, y1), (x2, y2), ...}
 * \param [in] mean_shape_width meanshape face width
 * \param [in] mean_shape_height meanshape face height
 * \param [in] pad_top pad on top, can be neg value
 * \param [in] pad_bottom pad on bottom, can be neg value
 * \param [in] pad_left pad on left, can be neg value
 * \param [in] pad_right pad on right, can be neg value
 * \param [out] final_points landmarks on cropped face {(x1, y1), (x2, y2), ...}, can be NULL.
 * \param [in] type method of sample
 * \return ture if succeed
 * \note final face data size should be [crop_width + pad_left + pad_right, crop_height + pad_top + pad_bottom]
 */
bool face_crop_core_ex(
	const cv::Mat img, cv::Mat& dstimg,
	const std::vector<cv::Point2f> points, std::vector<cv::Point2f>& final_points = std::vector<cv::Point2f>(),
	SAMPLING_TYPE type = LINEAR, PADDING_TYPE ptype = ZERO_PADDING);
/*
bool face_crop_core_ex(
	const cv::Mat img, cv::Mat& dstimg,
	const std::vector<cv::Point2f> points,
    int pad_top = 0, int pad_bottom = 0, int pad_left = 0, int pad_right = 0,
    float *final_points = nullptr,
    SAMPLING_TYPE type = LINEAR,
    PADDING_TYPE ptype = ZERO_PADDING );*/
bool transformation_maker(
	const std::vector <cv::Point2f> pts,
	double *transformation, int crop_width = 256, int crop_height = 256, float pad_left = 0,
	std::vector <cv::Point2f>& final_points = std::vector <cv::Point2f>(), float* A = NULL);
#endif // _SEETA_COMMON_ALIGNMENT_H
