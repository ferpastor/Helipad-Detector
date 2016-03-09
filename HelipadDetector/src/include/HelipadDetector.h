#ifndef HelipadDetector_H
#define HelipadDetector_H

//Prevoius includes. TODO make it clear
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
#include "pugixml.hpp"

//ROS
#include "ros/ros.h"


//Drone Module
#include "droneModuleROS.h"

//Drone Msgs
#include "droneMsgsROS/obsVector.h"


//ROS Images
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include "referenceFrames.h"

#include "nodes_definition.h"


///Consts
const double DRONE_HELIPAD_EYE_RATE = 30.0;

using namespace cv;
using namespace std;

/////////////////////////////////////////
// Class HelipadDetectorModule
//
//   Description
//
/////////////////////////////////////////
class HelipadDetector : public DroneModule {

  //Helipads Detected
  protected:
      std::string droneImageTopicName;
      //Front image msgs
      cv_bridge::CvImagePtr cvDroneImage;
      //ros::Time frontImage_timesamp;
      //uint32_t frontImage_seq;
      cv::Mat droneImage;
      //Subscriber
      ros::Subscriber droneFrontImageSubs;
      void droneImageCallback(const sensor_msgs::ImageConstPtr& msg);
      bool flagNewImage;

    //Helipads detected
  protected:
      std::string droneArucoListTopicName;
      ros::Publisher droneArucoListPubl; ////Publishers
      droneMsgsROS::obsVector droneArucoListMsg; //Messages
      bool publishArucoList();

  //Init and close
  public:
     void init();
     void close();

  //Open
  public:
     void open(ros::NodeHandle & nIn);

  //Reset
  protected:
     bool resetValues();

  //Start
  protected:
     bool startVal();

  //Stop
  protected:
     bool stopVal();

  //Run
  public:
     bool run();

  //Helipad Detector
  public:

        Mat InputImage;

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

        int setInputImage(Mat InputImageIn);

	void drawContours(Mat image, vector<Point> TheApproxCurve);

	void drawPoints(Mat image, vector<Point> TheApproxMarker);

	bool read();

	int perimeter(vector< Point > &a);

	bool FindHelipad(const Mat &in, bool &rotated);

	bool GetFrontHeliCandidate(Mat &in, Mat &out, Size size, vector< Point > points) throw(cv::Exception);

	void detectRectangles(Mat &thresImgv, vector< vector< Point > > &OutMarkerCanditates);

	int detect(Mat source, vector < vector < Point > > Helipads, Mat &Rvec, Mat &Tvec);

	void GetPose(float markerSizeX, float markerSizeY, cv::Mat camMatrix, cv::Mat distCoeff, Mat &Rvec, Mat &Tvec, vector < Point > Helipad);

	Mat set_P_matrix(Mat _R_matrix, Mat t_matrix);

};

#endif
