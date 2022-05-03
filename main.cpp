#define EXPORT extern "C" __declspec(dllexport)

#include <iostream>
#include "gradientBasedAutoExp.hpp"

gradientBasedAutoExp autoExp;

EXPORT double calcNewExp(cv::Mat *img, double currExposureTime, double kp = 0.75, double d = 0.25) {
	return autoExp.calcNewExposureTime(img, currExposureTime, kp, d);
}
int main() {

	return EXIT_SUCCESS;
}