#ifndef HelipadDetector_H
#define HelipadDetector_H

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <valarray>
#include <fstream>


using namespace cv;
using namespace std;

namespace heli{

class HelipadDetector {

  public:

	HelipadDetector();

	~HelipadDetector();

	int readParameters();

	void drawContours(Mat image, vector<Point> TheApproxCurve);

	void drawPoints(Mat image, vector<Point> TheApproxMarker);

	bool read();

	int perimeter(vector< Point > &a);

	bool FindHelipad(const Mat &in, bool &rotated);

	bool GetFrontHeliCandidate(Mat &in, Mat &out, Size size, vector< Point > points) throw(cv::Exception);

	void detectRectangles(Mat &thresImgv, vector< vector< Point > > &OutMarkerCanditates);

	int detect(Mat source, vector < vector < Point > > Helipads);

	void GetPose(float markerSizeX, float markerSizeY, cv::Mat camMatrix, cv::Mat distCoeff, Mat &Rvec, Mat &Tvec, vector < Point > Helipad);

	Mat set_P_matrix(Mat _R_matrix, Mat t_matrix);

};
}

#endif
