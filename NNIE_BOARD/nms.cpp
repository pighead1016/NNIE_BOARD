#include "nms.h"
namespace mars {
	
	// Get max scores with corresponding indices.
	//    scores: a set of scores.
	//    threshold: only consider scores higher than the threshold.
	//    top_k: if -1, keep all; otherwise, keep at most top_k.
	//    score_index_vec: store the sorted (score, index) pair.
	void DecodeBoxes(const cv::Rect2f& p, const int *t, cv::Rect2f& b, DecodeEnum Decodetype)
	{
		if (Decodetype == CENTER_SIZE) {
			float cbx = p.x + t[0] * 0.1*p.width / 4096;
			float cby = p.y + t[1] * 0.1*p.height / 4096;
			b.width = exp(0.2*t[2] / 4096)*p.width;
			b.height = exp(0.2*t[3] / 4096)*p.height;
			b.x = cbx - b.width / 2;
			b.y = cby - b.height / 2;
		}
	}
}