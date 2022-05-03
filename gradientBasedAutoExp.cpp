//
//  gradientBasedAutoExp.cpp
//
//  Created by Yusuf Sarıkaya on 4.11.2020.
//  
//
#include "gradientBasedAutoExp.hpp"

//kp -> control the speed to convergence
//d  -> parameters used in the nonliear function in Shim's 2018 paper
double gradientBasedAutoExp::calcNewExposureTime(cv::Mat *img, double currExposureTime, double kp, double d) {
	cv::Mat cv_img_resized;
	std::vector<double> coeffs;
	double R, max_gamma, alpha, newExposureTime;

	cv::resize(*img, cv_img_resized, cv::Size(320, 320 / (double)img->size().aspectRatio()));
	std::vector<double> gammaValues = { 0.1, 0.5, 0.8, 1.0, 1.2, 1.5, 1.9 };
	std::vector<cv::Mat> gammaCorrected = gradientBasedAutoExp::gammaCorrection(cv_img_resized, gammaValues);

	std::vector<double> gradientMagnitude = gradientBasedAutoExp::calcGradientMagnitude(gammaCorrected);

	//Find the coefficients of the fitting polynom
	PolynomialFittingSolution.fitIt(gammaValues, gradientMagnitude, 4, coeffs);
	coeffs = gradientBasedAutoExp::derivePolynomial(coeffs, 5);

	double maxgradientMagnitude = *max_element(gradientMagnitude.begin(), gradientMagnitude.end());
	double mingradientMagnitude = *min_element(gradientMagnitude.begin(), gradientMagnitude.end());

	if ((gradientMagnitude[0] == maxgradientMagnitude)) {
		max_gamma = gammaValues[0];
	}
	else if ((gradientMagnitude[6] == maxgradientMagnitude)) {
		max_gamma = gammaValues[6];
	}
	else {
		gradientBasedAutoExp::findCubicPolynomRoot(coeffs, max_gamma);
	}

	// alpha value refers to Shim's 2018 paper
	if (max_gamma < 1)
	{
		alpha = 1.0;
	}
	if (max_gamma >= 1)
	{
		alpha = 0.5;
	}

	// This update function was implemented in Shim's 2018 paper which is an update version of his 2014 paper
	R = d * tan((2 - max_gamma) * atan2(1, d) - atan2(1, d)) + 1;
	newExposureTime = (1 + alpha * kp * (R - 1)) * currExposureTime;

	return newExposureTime;
}

std::vector<cv::Mat> gradientBasedAutoExp::gammaCorrection(const cv::Mat img, std::vector<double> gamma_)
{
	std::vector<cv::Mat> gammaCorrectionVec;
    cv::Mat img_gamma_corrected;
    CV_Assert(gamma_.size() >= 0);
    //! [changing-contrast-brightness-gamma-correction]
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
	for (int gammaIndex = 0; gammaIndex < 7; gammaIndex++)
	{
		for (int i = 0; i < 256; ++i)
			p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma_[gammaIndex]) * 255.0);

		Mat res = img.clone();
		LUT(img, lookUpTable, res);
		gammaCorrectionVec.push_back(res);
	}
    
    //hconcat(img, res, img_gamma_corrected);
    //imshow("Gamma correction", img_gamma_corrected);
    return gammaCorrectionVec;

}

std::vector<double> gradientBasedAutoExp::calcGradientMagnitude(std::vector<cv::Mat> img){
    cv::Mat gray;
	std::vector<double> sum;

	for (int index = 0; index < img.size(); index++)
	{
		if (img[index].type() == CV_8UC3) {
			cv::cvtColor(img[index], gray, cv::COLOR_BGR2GRAY);
		}
		else {
			gray = img[index];
		}

		//Compute dx and dy derivatives
		cv::Mat1f dx, dy;
		Sobel(gray, dx, CV_32F, 1, 0);
		Sobel(gray, dy, CV_32F, 0, 1);

		//Compute gradient
		cv::Mat1f magn;
		magnitude(dx, dy, magn);
		sum.push_back(cv::sum(magn)[0]);
	}
    
    return sum;
}

std::vector<double> gradientBasedAutoExp::derivePolynomial(std::vector<double> coeffs, int numCoeffs) {
	double result;
	std::vector<double> newcoeffs;

	for (int i = 1; i < numCoeffs; i++) //start with 1 because the first element is constant.
	{
		result = coeffs[i] * i;
		newcoeffs.push_back(result);
		result = 0;
	}

	return newcoeffs;
}


bool gradientBasedAutoExp::findCubicPolynomRoot(std::vector<double> input, double &root)
{
	double d = input.at(0);
	double c = input.at(1);
	double b = input.at(2);
	double a = input.at(3);

	if (input.size() != 4)
		return false;

	if (a == 0.0 || abs(a / b) < 1.0e-6) {  // Quadratic case, ax^2+bx+c=0
		a = b; b = c; c = d;
		if (abs(a) < 1e-8) { // Linear case, ax+b=0
			a = b; b = c;
			if (abs(a) < 1e-8) // Degenerate case
				return false;
			root = (-b / a);
			return true;
		}

		double D = b * b - 4 * a*c;
		if (abs(D) < 1e-8)
		{
			root = (-b / (2 * a));
		}
		else if (D > 0)
		{
			root = (-b + sqrt(D)) / (2 * a);
			root = (-b - sqrt(D)) / (2 * a);
		}
		return true;
	}

	double B = b / a, C = c / a, D = d / a;

	double Q = (B*B - C * 3.0) / 9.0, QQQ = Q * Q*Q;
	double R = (2.0*B*B*B - 9.0*B*C + 27.0*D) / 54.0, RR = R * R;

	// 3 real roots
	if (RR < QQQ)
	{
		/* This sqrt and division is safe, since RR >= 0, so QQQ > RR,    */
		/* so QQQ > 0.  The acos is also safe, since RR/QQQ < 1, and      */
		/* thus R/sqrt(QQQ) < 1.                                     */
		double theta = acos(R / sqrt(QQQ));
		/* This sqrt is safe, since QQQ >= 0, and thus Q >= 0             */
		double r1, r2, r3;
		r1 = r2 = r3 = -2.0*sqrt(Q);
		r1 *= cos(theta / 3.0);
		r2 *= cos((theta + 2 * M_PIl) / 3.0);
		r3 *= cos((theta - 2 * M_PIl) / 3.0);

		r1 -= B / 3.0;
		r2 -= B / 3.0;
		r3 -= B / 3.0;

		root = 1000000.0;

		if (r1 >= 0.0) root = r1;
		if (r2 >= 0.0 && r2 < root) root = r2;
		if (r3 >= 0.0 && r3 < root) root = r3;

		return true;
	}
	// 1 real root
	else
	{
		double A2 = -pow(fabs(R) + sqrt(RR - QQQ), 1.0 / 3.0);
		if (A2 != 0.0) {
			if (R < 0.0) A2 = -A2;
			root = A2 + Q / A2;
		}
		root -= B / 3.0;
		return true;
	}
}