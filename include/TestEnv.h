#pragma once

using namespace cv;
using namespace std;

/**
*  @brief	Performance and correctness testing class for FIR filters:
*			Given an image, applies the provided filter using two different implementations
*			and compare the results. It also provides the computing time with a previously
*			defined number of threads
*/
class TestEnv
{
private:
	vector<float> _coeff;	/// Vector of filter coefficients
	Mat _image;				/// Original Image

	// One single iteration of loading and filter application
	void TestEnv::fullIteration(char*);

	// Calculates the correlation between two images
	float TestEnv::imgCorrelation(Mat &, Mat &);

	// Displays the two compared images and its substraction
	void TestEnv::visualComparison(Mat &, Mat &, Mat &);

	// Calculates square-root of sum of squared error between two images
	float TestEnv::sumOfSquaredError(Mat &, Mat &);	

public:

	// Default constructor
	TestEnv::TestEnv();

	// Overloaded constructor  
	TestEnv(vector<float>&);
	
	// Destructor
	~TestEnv();
	
	// _coeff setter
	void TestEnv::setCoeff(vector<float>&);

	// Loads an image given its location path
	void TestEnv::loadImage(char*);

	// Computes the summary of the result correctness
	void TestEnv::sanityCheck();

	// Computes the benchmark for the FIR filter implementation
	void TestEnv::performanceCheck(int, int, char*);
};