#include <iostream>
#include <opencv2/highgui/highgui.hpp>

#include "TestEnv.h"
#include "Fir.h";

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{	
	try
	{
		if (argc != 3)															/// Check the number of arguments
		{
			cout << " Usage: test_fir <ImagePath> <numberOfThreads>" << endl;
			return -1;
		}

		if (atoi(argv[2]) < 0 || atoi(argv[2]) > 100) {							/// Define some limitations in the threads
			cout << "The number of threads must be 0<x<100" << std::endl;
			return -1;
		}

		vector<float> firCoeffs = { 0.5, 0.5, 0.5, 0.5, 0.5 };
		int N = 10;
		TestEnv *test = new TestEnv(firCoeffs);

		test->loadImage(argv[1]);

		test->sanityCheck();

		test->performanceCheck(atoi(argv[2]), N, argv[1]);
		
		waitKey(0);
		return 0;
	}
	catch (std::exception &ex)
	{
		cout << "Exception :" << ex.what() << endl;
	}
}