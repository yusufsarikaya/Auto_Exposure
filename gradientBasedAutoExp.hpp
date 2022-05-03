//
//  gradientBasedAutoExp.hpp
//
//  Created by Yusuf Sarıkaya on 4.11.2020.
//  
//
#ifndef autoExp_hpp
#define autoExp_hpp

# define M_PIl          3.141592653589793238462643383279502884L

#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <numeric>
#include <execution>
#include <math.h>

#include "PolynomialRegression.h"

using namespace cv;

class gradientBasedAutoExp {
    
public:
	double calcNewExposureTime(cv::Mat *img, double currExposureTime, double kp, double d);

private:
    std::vector<cv::Mat> gammaCorrection(const cv::Mat img, std::vector<double> gamma_);
    std::vector<double> calcGradientMagnitude(std::vector<cv::Mat> img);
	std::vector<double> derivePolynomial(std::vector<double> coeffs, int numCoeffs);
	bool findCubicPolynomRoot(std::vector<double> input, double &root);

	PolynomialRegression <double> PolynomialFittingSolution;

};


#endif /* gradientBasedAutoExp_hpp */