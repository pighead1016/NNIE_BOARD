#include "POSE_NET.h"
template<typename T>
inline char charRound(const T a)
{
	return char(a + 0.5f);
}
template<typename T>
inline signed char sCharRound(const T a)
{
	return (signed char)(a + 0.5f);
}

template<typename T>
inline int intRound(const T a)
{
	return int(a + 0.5f);
}

template<typename T>
inline long longRound(const T a)
{
	return long(a + 0.5f);
}

template<typename T>
inline long long longLongRound(const T a)
{
	return (long long)(a + 0.5f);
}
// Unsigned
template<typename T>
inline unsigned char uCharRound(const T a)
{
	return (unsigned char)(a + 0.5f);
}

template<typename T>
inline unsigned int uIntRound(const T a)
{
	return (unsigned int)(a + 0.5f);
}

template<typename T>
inline unsigned long ulongRound(const T a)
{
	return (unsigned long)(a + 0.5f);
}

template<typename T>
inline unsigned long long uLongLongRound(const T a)
{
	return (unsigned long long)(a + 0.5f);
}
// Max/min functions
template<typename T>
inline T fastMax(const T a, const T b)
{
	return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
	return (a < b ? a : b);
}

template<class T>
inline T fastTruncate(T value, T min = 0, T max = 1)
{
	return fastMin(max, fastMax(min, value));
}
std::vector<unsigned int> POSE_COCO_PAIRS;
std::vector<unsigned int> POSE_COCO_MAP_IDX;
std::vector<unsigned int> TIRED_RENDER;
struct BlobData {
	int count;
	float* list;
	int num;
	int channels;
	int height;
	int width;
	int capacity_count;
};
//cv::Mat getImage(const cv::Mat& im, cv::Size baseSize = cv::Size(656, 368), float* scale = 0) {
//	int w = baseSize.width;
//	int h = baseSize.height;
//	int nh = h;
//	float s = h / (float)im.rows;;
//	int nw = im.cols * s;
//
//	if (nw > w) {
//		nw = w;
//		s = w / (float)im.cols;
//		nh = im.rows * s;
//	}
//
//	if (scale)*scale = 1 / s;
//	cv::Rect dst(0, 0, nw, nh);
//	cv::Mat bck = cv::Mat::zeros(h, w, CV_8UC3);
//	cv::resize(im, bck(dst), cv::Size(nw, nh));
//	return bck;
//}


float cos_vector(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4)
{
	float v1x = p1.x - p2.x;
	float v1y = p1.y - p2.y;
	float v2x = p3.x - p4.x;
	float v2y = p3.y - p4.y;
	return (v1x*v2x + v1y*v2y) / sqrt((v1x*v1x + v1y*v1y)*(v2x*v2x + v2y*v2y));
}

void nms(BlobData* bottom_blob, BlobData* top_blob, float threshold) {
	int w = bottom_blob->width;
	int h = bottom_blob->height;
	int plane_offset = w * h;
	float* ptr = bottom_blob->list;
	float* top_ptr = top_blob->list;
	int top_plane_offset = top_blob->width * top_blob->height;
	int max_peaks = top_blob->height - 1;

	for (int n = 0; n < bottom_blob->num; ++n) {
		for (int c = 0; c < bottom_blob->channels - 1; ++c) {

			int num_peaks = 0;
			for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y) {
				for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x) {
					float value = ptr[y*w + x];
					if (value > threshold) {
						const float topLeft = ptr[(y - 1)*w + x - 1];
						const float top = ptr[(y - 1)*w + x];
						const float topRight = ptr[(y - 1)*w + x + 1];
						const float left = ptr[y*w + x - 1];
						const float right = ptr[y*w + x + 1];
						const float bottomLeft = ptr[(y + 1)*w + x - 1];
						const float bottom = ptr[(y + 1)*w + x];
						const float bottomRight = ptr[(y + 1)*w + x + 1];
						if (value > topLeft && value > top && value > topRight
							&& value > left && value > right
							&& value > bottomLeft && value > bottom && value > bottomRight)
						{
							float xAcc = 0;
							float yAcc = 0;
							float scoreAcc = 0;
							for (int kx = -3; kx <= 3; ++kx) {
								int ux = x + kx;
								if (ux >= 0 && ux < w) {
									for (int ky = -3; ky <= 3; ++ky) {
										int uy = y + ky;
										if (uy >= 0 && uy < h) {
											float score = ptr[uy * w + ux];
											xAcc += ux * score;
											yAcc += uy * score;
											scoreAcc += score;
										}
									}
								}
							}

							xAcc /= scoreAcc;
							yAcc /= scoreAcc;
							scoreAcc = value;
							top_ptr[(num_peaks + 1) * 3 + 0] = xAcc;
							top_ptr[(num_peaks + 1) * 3 + 1] = yAcc;
							top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
							num_peaks++;
						}
					}
				}
			}
			top_ptr[0] = num_peaks;
			ptr += plane_offset;
			top_ptr += top_plane_offset;
		}
	}
}
float p2p_distance(cv::Point2f p1, cv::Point2f p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}
#define M_PI       3.14159265358979323846

BlobData* createBlob_local(int num, int channels, int height, int width) {
	BlobData* blob = new BlobData();
	blob->num = num;
	blob->width = width;
	blob->channels = channels;
	blob->height = height;
	blob->count = num*width*channels*height;
	blob->list = new float[blob->count];
	blob->capacity_count = blob->count;
	return blob;
}

BlobData* createEmptyBlobData() {
	BlobData* blob = new BlobData();
	memset(blob, 0, sizeof(*blob));
	return blob;
}

void releaseBlob_local(BlobData** blob) {
	if (blob) {
		BlobData* ptr = *blob;
		if (ptr) {
			if (ptr->list)
				delete[] ptr->list;

			delete ptr;
		}
		*blob = 0;
	}
}


POSE_NET::POSE_NET()
{
}


POSE_NET::~POSE_NET()
{
}

HI_S32 POSE_NET::POSE_INIT(HI_CHAR * pszModelFile)
{
	HI_S32 s32Ret = NNIE_NET_INIT(pszModelFile);
	
	/*check*/
	SAMPLE_SVP_CHECK_EXPR_RET(stModel.u32NetSegNum != 1, HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,Pose seg is only 1 failed!\n");
	SAMPLE_SVP_CHECK_EXPR_RET(stModel.astSeg[0].u16DstNum != 1, HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,Pose seg 0 dstnode is only 1 failed!\n");
	HI_S32 c = astSegData[0].astDst[0].unShape.stWhc.u32Chn;
	SAMPLE_SVP_CHECK_EXPR_RET(c%3!=0, c, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,Poseout channel is not treble failed!\n");
	numberBodyParts = c / 3 - 1;
	HI_U32 stride = this->astSegData[0].astDst[0].u32Stride;
	HI_U32 width = astSegData[0].astDst[0].unShape.stWhc.u32Width;
	SAMPLE_SVP_CHECK_EXPR_RET(stride != width*4, width, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,Poseinupt WK is just octuple failed!\n");

	return s32Ret;
}

void POSE_NET::show_keypoint(cv::Mat & image)
{
	int numberKeypoints = shape[1];
	float scalex = float(image.cols) / this->BaseSize.width;
	float scaley = float(image.rows) / this->BaseSize.height;
	for (auto person = 0; person < shape[0]; person++)
		for (size_t pn = 0; pn < POSE_COCO_PAIRS.size(); pn += 2)
		{
			const int index1 = (person*numberKeypoints + POSE_COCO_PAIRS[pn])*shape[2];
			const int index2 = (person*numberKeypoints + POSE_COCO_PAIRS[pn + 1])*shape[2];
			if (keypoints[index1] > 0.05&&keypoints[index2] > 0.05)
			{
				const cv::Point kp1(intRound(keypoints[index1] * scalex), intRound(keypoints[index1 + 1] * scaley));
				const cv::Point kp2(intRound(keypoints[index2] * scalex), intRound(keypoints[index2 + 1] * scaley));
				line(image, kp1, kp2, cv::Scalar(0, 128, 255), 2);
			}
		}

}

HI_S32 POSE_NET::KetPoint(const cv::Mat img, float scale)
{
	if (this->stModel.astSeg[0].u16DstNum != 1)
		return HI_FAILURE;
	SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S input_index = { 0,0 };
	SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S process_index = { 0,0 };
	SVP_FillSrcData_Mat(&input_index, img);
	SAMPLE_SVP_NNIE_Forward(&input_index, &process_index);
	auto blob = this->astSegData[0].astDst[0];
	int outc = blob.unShape.stWhc.u32Chn;
	int outh = blob.unShape.stWhc.u32Height;
	int outw = blob.unShape.stWhc.u32Width;
	BaseSize = cv::Size(outw*scale, outh*scale);
	numberBodyParts = outc / 3 - 1;
	int stride = blob.u32Stride;
	if (stride != outw * 4) {
		return HI_FAILURE;
	}
	BlobData* nms_out = createBlob_local(1, outc - 1, POSE_MAX_PEOPLE + 1, 3);
	BlobData* input = createBlob_local(1, outc, BaseSize.height, BaseSize.width);

	BlobData* net_output = createBlob_local(1, outc, outh, outw);
	HI_S32* data = (HI_S32*)blob.u64VirAddr;
	for (int i = 0; i < outc; ++i) {
		cv::Mat um(BaseSize.height, BaseSize.width, CV_32F, input->list + i*BaseSize.height*BaseSize.width);
		cv::Mat tmp32 = cv::Mat(net_output->height, net_output->width, CV_32S, data + i*outh*stride / 4);
		cv::Mat tmpf;
		tmp32.convertTo(tmpf, CV_32F, 1 / 4096.0);
		cv::resize(tmpf, um, BaseSize, 0, 0, CV_INTER_CUBIC);
	}
	nms(input, nms_out, 0.05);
	connectBodyPartsCpu(input->list, nms_out->list, BaseSize, POSE_MAX_PEOPLE, 9, 0.05, 6, 0.4, 1);
	releaseBlob_local(&net_output);
	releaseBlob_local(&input);
	releaseBlob_local(&nms_out);
	return HI_SUCCESS;
}

void POSE_NET::connectBodyPartsCpu(const float * const heatMapPtr, const float * const peaksPtr, const cv::Size & heatMapSize, const int maxPeaks, const int interMinAboveThreshold, const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor)
{
	shape.resize(3);
	int backear;
	if (numberBodyParts == 25) {
		backear = 19;
		POSE_COCO_PAIRS = { 1, 8, 1, 2, 1, 5,
			2, 3, 3, 4, 5, 6, 6, 7,
			8, 9, 8, 12,
			9, 10, 10, 11,
			12, 13, 13, 14,
			1, 0, 0, 16, 0, 15, 15, 17, 16, 18,
			2, 17, 5, 18,
			14, 19, 19, 20, 14, 21,
			11, 22, 22, 23, 11, 24
		};
		POSE_COCO_MAP_IDX = { 26, 27, 40, 41, 48, 49,
			42, 43, 44, 45, 50, 51, 52, 53,
			32, 33, 34, 35,
			28, 29, 30, 31,
			36, 37, 38, 39,
			56, 57, 60, 61, 58, 59, 62, 63, 64, 65,
			46, 47, 54, 55,
			66, 67, 68, 69, 70, 71,
			72, 73, 74, 75, 76, 77
		};
		TIRED_RENDER = { 0,2,5,18 };
	}
	else {
		backear = 17;
		POSE_COCO_PAIRS = { 1, 5 , 1, 2,
			2, 3, 3, 4, 5, 6, 6, 7,
			1 , 8, 8, 9, 9,10,
			1, 11,11,12,12,13,
			1, 0, 0,14, 14,16,0,15,15,17,
			2,16, 5,17
			//14,19,19,20,14,21,
			//11,22,22,23,11,24
		};
		POSE_COCO_MAP_IDX = { 39,40,31,32,
			33,34,35,36,41,42,43,44,
			19,20,21,22,23,24,
			25,26,27,28,29,30,
			47,48,49,50,53,54,51,52,55,56,
			37,38,45,46
			//66,67,68,69,70,71,
			//72,73,74,75,76,77
		};
		TIRED_RENDER = { 0,2,5,17 };
	}
	const auto& bodyPartPairs = POSE_COCO_PAIRS;
	const auto& mapIdx = POSE_COCO_MAP_IDX;

	const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

	std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
	const auto subsetCounterIndex = numberBodyParts;
	const auto subsetSize = numberBodyParts + 1;
	const auto peaksOffset = 3 * (maxPeaks + 1);
	const auto heatMapOffset = heatMapSize.area();
	for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
	{
		const auto bodyPartA = bodyPartPairs[2 * pairIndex];
		const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
		const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
		const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
		const auto nA = intRound(candidateA[0]);
		const auto nB = intRound(candidateB[0]);
		// add parts into the subset in special case
		if (nA == 0 || nB == 0)
		{
			// Change w.r.t. other
			if (nA == 0) // nB == 0 or not
			{
				for (auto i = 1; i <= nB; i++)
				{
					bool num = false;
					const auto indexB = bodyPartB;
					for (auto j = 0u; j < subset.size(); j++)
					{
						const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
						if (subset[j].first[indexB] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> rowVector(subsetSize, 0);
						rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2; //store the index
						rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
						const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
						subset.emplace_back(std::make_pair(rowVector, subsetScore));
					}
				}
			}
			else // if (nA != 0 && nB == 0)
			{
				for (auto i = 1; i <= nA; i++)
				{
					bool num = false;
					const auto indexA = bodyPartA;
					for (auto j = 0u; j < subset.size(); j++)
					{
						const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
						if (subset[j].first[indexA] == off)
						{
							num = true;
							break;
						}
					}
					if (!num)
					{
						std::vector<int> rowVector(subsetSize, 0);
						rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2; //store the index
						rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
						const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
						subset.emplace_back(std::make_pair(rowVector, subsetScore));
					}
				}
			}
		}
		else // if (nA != 0 && nB != 0)
		{
			std::vector<std::tuple<double, int, int>> temp;
			const auto numInter = 10;
			const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
			const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
			for (auto i = 1; i <= nA; i++)
			{
				for (auto j = 1; j <= nB; j++)
				{
					const auto dX = candidateB[j * 3] - candidateA[i * 3];
					const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
					const auto normVec = float(std::sqrt(dX*dX + dY*dY));
					// If the peaksPtr are coincident. Don't connect them.
					if (normVec > 1e-6)
					{
						const auto sX = candidateA[i * 3];
						const auto sY = candidateA[i * 3 + 1];
						const auto vecX = dX / normVec;
						const auto vecY = dY / normVec;

						auto sum = 0.;
						auto count = 0;
						for (auto lm = 0; lm < numInter; lm++)
						{
							const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm*dX / numInter));
							const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm*dY / numInter));
							//checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
							//checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
							const auto idx = mY * heatMapSize.width + mX;
							const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
							if (score > interThreshold)
							{
								sum += score;
								count++;
							}
						}
						// parts score + connection score
						if (count > interMinAboveThreshold)
							temp.emplace_back(std::make_tuple(sum / count, i, j));
					}
				}
			}

			// select the top minAB connection, assuming that each part occur only once
			// sort rows in descending order based on parts + connection score
			if (!temp.empty())
				std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

			std::vector<std::tuple<int, int, double>> connectionK;

			const auto minAB = fastMin(nA, nB);
			std::vector<int> occurA(nA, 0);
			std::vector<int> occurB(nB, 0);
			auto counter = 0;
			for (auto row = 0u; row < temp.size(); row++)
			{
				const auto score = std::get<0>(temp[row]);
				const auto x = std::get<1>(temp[row]);
				const auto y = std::get<2>(temp[row]);
				if (!occurA[x - 1] && !occurB[y - 1])
				{
					connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
						bodyPartB*peaksOffset + y * 3 + 2,
						score));
					counter++;
					if (counter == minAB)
						break;
					occurA[x - 1] = 1;
					occurB[y - 1] = 1;
				}
			}
			// Cluster all the body part candidates into subset based on the part connection
			// initialize first body part connection 15&16
			if (pairIndex == 0)
			{
				for (const auto connectionKI : connectionK)
				{
					std::vector<int> rowVector(numberBodyParts + 3, 0);
					const auto indexA = std::get<0>(connectionKI);
					const auto indexB = std::get<1>(connectionKI);
					const auto score = std::get<2>(connectionKI);
					rowVector[bodyPartPairs[0]] = indexA;
					rowVector[bodyPartPairs[1]] = indexB;
					rowVector[subsetCounterIndex] = 2;
					// add the score of parts and the connection
					const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
					subset.emplace_back(std::make_pair(rowVector, subsetScore));
				}
			}
			// Add ears connections (in case person is looking to opposite direction to camera)
			else if (pairIndex == backear || pairIndex == 18)
			{
				for (const auto& connectionKI : connectionK)
				{
					const auto indexA = std::get<0>(connectionKI);
					const auto indexB = std::get<1>(connectionKI);
					for (auto& subsetJ : subset)
					{
						auto& subsetJFirst = subsetJ.first[bodyPartA];
						auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
						if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
							subsetJFirstPlus1 = indexB;
						else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
							subsetJFirst = indexA;
					}
				}
			}
			else {
				if (!connectionK.empty()) {
					// A is already in the subset, find its connection B
					for (auto i = 0u; i < connectionK.size(); i++) {
						const auto indexA = std::get<0>(connectionK[i]);
						const auto indexB = std::get<1>(connectionK[i]);
						const auto score = std::get<2>(connectionK[i]);
						auto num = 0;
						for (auto j = 0u; j < subset.size(); j++) {
							if (subset[j].first[bodyPartA] == indexA)
							{
								subset[j].first[bodyPartB] = indexB;
								num++;
								subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
								subset[j].second = subset[j].second + peaksPtr[indexB] + score;
							}
						}
						// if can not find partA in the subset, create a new subset
						if (num == 0)
						{
							std::vector<int> rowVector(subsetSize, 0);
							rowVector[bodyPartA] = indexA;
							rowVector[bodyPartB] = indexB;
							rowVector[subsetCounterIndex] = 2;
							const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
							subset.emplace_back(std::make_pair(rowVector, subsetScore));
						}
					}
				}
			}
		}
	}

	// Delete people below the following thresholds:
	// a) minSubsetCnt: removed if less than minSubsetCnt body parts
	// b) minSubsetScore: removed if global score smaller than this
	// c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
	auto numberPeople = 0;
	std::vector<int> validSubsetIndexes;
	validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
	for (auto index = 0u; index < subset.size(); index++)
	{
		const auto subsetCounter = subset[index].first[subsetCounterIndex];
		const auto subsetScore = subset[index].second;
		if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
		{
			numberPeople++;
			validSubsetIndexes.emplace_back(index);
			if (numberPeople == POSE_MAX_PEOPLE)
				break;
		}
		else if (subsetCounter < 1)
			printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
	}

	// Fill and return poseKeypoints
	shape = { numberPeople, (int)numberBodyParts, 3 };
	if (numberPeople > 0)
		keypoints.resize(numberPeople * (int)numberBodyParts * 3);
	else
		keypoints.clear();

	for (auto person = 0u; person < validSubsetIndexes.size(); person++) {
		const auto& subsetI = subset[validSubsetIndexes[person]].first;
		for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++) {
			const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
			const auto bodyPartIndex = subsetI[bodyPart];
			if (bodyPartIndex > 0) {
				keypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
				keypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
				keypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
			}
			else {
				keypoints[baseOffset] = 0.f;
				keypoints[baseOffset + 1] = 0.f;
				keypoints[baseOffset + 2] = 0.f;
			}
		}
	}

}
