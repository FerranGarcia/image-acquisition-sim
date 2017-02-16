#pragma once

/**
*  @brief Class that filters an image applying a 1D kernel
*/
class Fir
{

private:
	std::vector<float> coefficients = {};		/// filter coefficients to apply

public:

	// Default constructor
	Fir::Fir();

	// Overloaded constructor  
	Fir::Fir(std::vector<float> &coeff);

	// Destructor
	~Fir();

	// Given an image applies the filter defined
	cv::Mat Fir::filter(cv::Mat &img);
};

