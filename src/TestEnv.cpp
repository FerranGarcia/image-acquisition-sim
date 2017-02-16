#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include "TestEnv.h"
#include "Fir.h";

using namespace cv;
using namespace std;
using namespace chrono;

TestEnv::TestEnv() {
}

/**
*   @brief  Overloaded constructor
*
*   @param  coeff is the reference to a vector of float
*/
TestEnv::TestEnv(vector<float>& coeff) {
	_coeff = coeff;
}

TestEnv::~TestEnv()
{
}

/**
*   @brief  Loads an image given its path location
*
*   @param  imagePath is a pointer to the string
*/
void TestEnv::loadImage(char* imagePath) {

	_image = imread(imagePath, CV_LOAD_IMAGE_GRAYSCALE);								/// Load the image from the path provided		

	if (!_image.data)																	/// Check for invalid input
	{
		cout << "Could not open or find the image" << endl;
	}
}

/**
*   @brief  Calculate the L2 relative error between images
*
*   @param  reference to two images
*   @return unit between 0 and 1, where 0 means same image
*/
float TestEnv::sumOfSquaredError(Mat &image_1, Mat &image_2) {
	if (image_1.rows > 0 && image_1.rows == image_2.rows && 
		image_1.cols > 0 && image_1.cols == image_2.cols) {
																						
		float errorL2 = norm(image_1, image_2, CV_L2);
		float error = (errorL2 / (float)(image_1.rows * image_1.cols));				/// Convert to a reasonable scale
		return error;
	}
	else {																				///Images have a different size
		return 0; 
	}
}

/**
*   @brief  Calculate the correlation between images
*
*   @param  reference to two images
*   @return unit between 0 and 1, where 1 means same image
*/
float TestEnv::imgCorrelation(Mat &image_1, Mat &image_2) {
	int n_pixels = image_1.rows * image_1.cols;
	
	cv::Scalar im1_Mean, im1_Std, im2_Mean, im2_Std;									/// Compute mean and standard deviation
	meanStdDev(image_1, im1_Mean, im1_Std);
	meanStdDev(image_2, im2_Mean, im2_Std);

	float covar = (image_1 - im1_Mean).dot(image_2 - im2_Mean) / n_pixels;				/// Compute covariance and correlation
	float correl = covar / (im1_Std[0] * im2_Std[0]);

	return correl;
}

/**
*   @brief  Shows the original, openCV and Fir filtered images and its substraction 
*			A black image as substraction means same images
*
*   @param  reference to three images
*/
void TestEnv::visualComparison(Mat &imageCast, Mat &opencvFilRes, Mat &imageFilRes) {
	namedWindow("Original", WINDOW_AUTOSIZE);											/// Create a window for display
	imshow("Original", imageCast);														/// Show our image inside it

	namedWindow("Fir Filtered", WINDOW_AUTOSIZE);										
	imshow("Fir Filtered", imageFilRes);												

	namedWindow("OpenCV Filtered", WINDOW_AUTOSIZE);									
	imshow("OpenCV Filtered", (opencvFilRes));											

	namedWindow("OpenCV - Fir", WINDOW_AUTOSIZE);										
	imshow("OpenCV - Fir", (opencvFilRes - imageFilRes));								
}

/**
*   @brief  Runs the qualitative and quantitative tests for comparison of
*			the two different methods used for filtering given the same image
*			and shows the results in console
*/
void TestEnv::sanityCheck() {

	Mat imageCast, imageFilRes, opencvFilRes;

	_image.convertTo(imageCast, CV_32F, 1.0 / 255.0);									/// Need to normalize
	
	Fir *firModule = new Fir(_coeff);
	imageFilRes = firModule->filter(imageCast);
		
	float* fircoeffsArray = &_coeff[0];
	Mat kernel = Mat(5, 1, CV_32F, fircoeffsArray);
	Point anchor = Point(0, 0);															
	float delta = 0;
	int ddepth = -1;
	
	// Adding BORDER_CONSTANT, the expected output matches the one computed using Fir
	// OpenCV filter2D uses convolution through different threads on the same image -> Better performance
	filter2D(imageCast, opencvFilRes, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
	
	visualComparison(imageCast, opencvFilRes, imageFilRes);								/// Visual comparison

	float squaredError = sumOfSquaredError(opencvFilRes, imageFilRes);					/// Calculate the L2 error
	
	float correlation = imgCorrelation(opencvFilRes, imageFilRes);						/// Correlation

	cout << ">Visual comparison:"<< endl;												/// Display in console
	cout << "'OpenCV - Fir' shows the difference between both images. If black, exactly the same image" << endl;
	cout << ">Squared Error: " << squaredError << endl;
	cout << ">Correlation: " << correlation << endl;
}

/**
*   @brief  Loads an image and applies the correspondent filter using Fir class
*			
*	@param  imagePath is a pointer to the string
*/
void TestEnv::fullIteration(char* imagePath) {

	this->loadImage(imagePath);

	Mat imageCast, imageFilRes;

	_image.convertTo(imageCast, CV_32F, 1.0 / 255.0);

	Fir *firModule = new Fir(_coeff);
	imageFilRes = firModule->filter(imageCast);
}

/**
*   @brief  Initializes X threads and aplies the Fir filter to the images provided.
*			For the sake of illustration the very same image (in gray) is provided to each thread,
*			so the actual number of images provided is: N * numberThreads.
*
*			Another approach could be to provide each image column to a different thread instead.
*
*			As it is stated the current iteration is not done till all threads join the main one
*			which may not be the most efficient way to process the greatest amount of images
*
*	@param  numberThreads and N provide a the number of threads and iterations respectively and
*			imagePath is a pointer to the string
*/
void TestEnv::performanceCheck(int numberThreads, int N, char* imagePath) {
	static const int numThreads = numberThreads;
	static const int images = N;
	vector<thread> t(numThreads);

	high_resolution_clock::time_point t1 = high_resolution_clock::now();

	int image = 0;
	while (image < images) {
		for (int i = 0; i < numThreads; ++i) {											///Launch a group of threads
			t.at(i) = thread(&TestEnv::fullIteration, this, imagePath);
		}

		for (int i = 0; i < numThreads; ++i) {											///Join the threads with the main thread
			t.at(i).join();
		}
		image++;
	}

	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(t2 - t1).count();

	float imageSecond = (images * numThreads) / (float)(duration / 1000);

	cout << ">Computational time: " << imageSecond << " img/s" << endl;
}

/**
*   @brief  Sets a new vector of coefficients for filtering
*
*	@param  coeff is a reference to the coefficients
*/
void TestEnv::setCoeff(vector<float> &coeff) {
	_coeff = coeff;
}