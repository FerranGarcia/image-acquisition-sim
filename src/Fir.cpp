#include <opencv2/highgui/highgui.hpp>

#include "Fir.h"

using namespace cv;
using namespace std;

Fir::Fir()
{
}

/**
*   @brief  Overloaded constructor
*
*   @param  coeff is the reference to a vector of float
*/
Fir::Fir(std::vector<float> &coeff)
{
	coefficients = coeff;
}

Fir::~Fir()
{
}

/**
*   @brief  Filters the given image aplying a convolution
*
*   @param  img is the reference to the image provided
*   @return The image filtered as result
*/
cv::Mat Fir::filter(cv::Mat &img)
{		
	int numFrames = img.size[0];
	int featuresPrevious = img.size[1];
	Mat filteredTrajectory = cv::Mat ::zeros(numFrames, featuresPrevious, CV_32F);

	for (int j = 0; j<featuresPrevious; j++)
	{
		for (int i = 0; i<numFrames ; i++)
		{
			float tmp = 0;
			for (int k = 0; k < coefficients.size(); k++) {
				if (i+k < numFrames) {
					tmp += (img.at<float>(i + k, j) * coefficients[k]);
				}
			}
			filteredTrajectory.at<float>(i, j) = tmp;
		}
	}
	return filteredTrajectory;
}