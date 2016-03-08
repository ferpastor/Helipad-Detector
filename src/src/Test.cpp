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
#include <HelipadDetector.h>

using namespace cv;
using namespace std;
using namespace heli;

 int main(int argc,char **argv)
 {
     try
     {

         Mat inImage;

         HelipadDetector Detector;

         VideoCapture Cap(0);

         if(!Cap.isOpened())
         {
             cout << "Unable to open video capturer" << endl;
             return 0;
         }

         while(true)
         {
             bool bSuccess = Cap.read(inImage); // read a new frame from video

             if (!bSuccess) //if not success, break loop
             {
                  cout << "Cannot read a frame from video stream" << endl;
                  break;
             }

             vector < vector < Point > > Helipad;

             //HelipadDetector Detector;

             int nHelis = 0;

             nHelis = Detector.detect(inImage, Helipad);

             if(waitKey(30) == 27)
             {
                 break;
             }

         }

     }
     catch(Exception e)
     {
         cout << e.what() << endl;
         return 0;
     }

    //cout << nHelis <<endl;
    return 0;

 }
