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

	int Th;//This Param must be measured in the real Helipad. It represents the total height
		   //The units are not important, just keep in mind you must use the same ones you are going to use in all the measurements.
	int Tw;//Total width
	int Mhb;//The height of the black line under the "H"
	int Mwb;//The width of the other black line

	int Twh; //Height of the white square
	int Tww; //width of the white square
	int Mhw; //Height of the white "line" under the "H"
	int Mww; //width of the other white "line"
	int Hw; //width of the "H"
	int Hsw; //width of the H's vertical lines


	int _thresParam1_range; //used for the threshold step
	int _markerWarpSizex; //Size of the columns and rows of the Cannonical Marker Image
	int _markerWarpSizey; //Size of the columns and rows of the Cannonical Marker Image


	double _minSize; //used to identify candidate markers
	double _maxSize; //used to identify candidate markers

	float tooNearCandidatesDistance; //Used to remove too near marker candidates.

	float markerSizeX;
	float markerSizeY;

	int _thresParam1; //Params for the threshold
	int _thresParam2; //this one is only used in the adaptative threshold.
	int thresMethod; // 2: Adaptative threshold

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
