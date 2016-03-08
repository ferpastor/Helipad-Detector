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
