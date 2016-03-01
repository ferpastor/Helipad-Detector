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
#include <sstream>
#include <time.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;

int _thresParam1 = 7; //Params for the threshold
int _thresParam2 = 7; //this one is only used in the adaptative threshold.
int thresMethod = 2; // 2: Adaptative threshold


int Th = 39;//This Param must be measured in the real Helipad. It represents the total height
           //The units are not important, just keep in mind you must use the same ones you are going to use in all the measurements.
int Tw = 37;//Total width
int Mhb = 4;//The height of the black line under the "H"
int Mwb = 4;//The width of the other black line

int Twh = 31; //Height of the white square
int Tww = 28; //width of the white square
int Mhw = 6; //Height of the white "line" under the "H"
int Mww = 7; //width of the other white "line"
int Hw = 37; //width of the "H"
int Hsw = 5; //width of the H's vertical lines

/*
int Th = 96;//This Param must be measured in the real Helipad. It represents the total height
           //The units are not important, just keep in mind you must use the same ones you are going to use in all the measurements.
int Tw = 91;//Total width
int Mhb = 10;//The height of the black line under the "H"
int Mwb = 11;//The width of the other black line

int Twh = 76; //Height of the white square
int Tww = 69; //width of the white square
int Mhw = 15; //Height of the white "line" under the "H"
int Mww = 16; //width of the other white "line"
int Hw = 37; //width of the "H"
int Hsw = 5; //width of the H's vertical lines
*/

int _thresParam1_range = 0; //used for the threshold step
int _markerWarpSizex = 250; //Size of the columns and rows of the Cannonical Marker Image
int _markerWarpSizey = 250; //Size of the columns and rows of the Cannonical Marker Image


double _minSize = 0.04; //used to identify candidate markers
double _maxSize = 0.5; //used to identify candidate markers

float tooNearCandidatesDistance = 10; //Used to remove too near marker candidates.

float markerSizeX = 0.096;
float markerSizeY = 0.091;

Mat CamMatrix(3, 3, CV_32FC1);

Mat DistMatrix(5,1, CV_32FC1);

Mat Pose;

double _borderDistThres = 0.025; // corners in a border of 2.5% of image  are ignored

bool _useLockedCorners = false;

//TODO Leer estos parámetros desde fichero.

namespace heli{

HelipadDetector::HelipadDetector()
{
    // TODO Leer parámetros de cámara desde fichero
    CamMatrix.at < float > (0, 0) = 6.8966032179638808e+02;
    CamMatrix.at < float > (0, 1) = 0;
    CamMatrix.at < float > (0, 2) = 3.1950000000000000e+02;
    CamMatrix.at < float > (1, 0) = 0;
    CamMatrix.at < float > (1, 1) = 6.8966032179638808e+02;
    CamMatrix.at < float > (1, 2) = 2.3950000000000000e+02;
    CamMatrix.at < float > (2, 0) = 0;
    CamMatrix.at < float > (2, 1) = 0;
    CamMatrix.at < float > (2, 2) = 1;

    CamMatrix.at < float > (0, 0) = 7.7515813387338650e-02;
    CamMatrix.at < float > (0, 1) = 2.8459888465880390e-01;
    CamMatrix.at < float > (0, 2) = 0;
    CamMatrix.at < float > (0, 3) = 0;
    CamMatrix.at < float > (0, 4) = -2.5553636828393729e+00;
}

HelipadDetector::~HelipadDetector() {}

void HelipadDetector::drawContours(Mat image, vector<Point> TheApproxCurve)
{
    line(image, TheApproxCurve[0], TheApproxCurve[1], CV_RGB(200, 100, 100), 2);
    line(image, TheApproxCurve[1], TheApproxCurve[2], CV_RGB(200, 100, 100), 2);
    line(image, TheApproxCurve[2], TheApproxCurve[3], CV_RGB(200, 100, 100), 2);
    line(image, TheApproxCurve[3], TheApproxCurve[0], CV_RGB(200, 100, 100), 2);
}

void HelipadDetector::drawPoints(Mat image, vector<Point> TheMarker)
{
    circle(image, TheMarker[0], 8, CV_RGB(250, 0, 0), -1);
    circle(image, TheMarker[1], 8, CV_RGB(0, 250, 0), -1);
    circle(image, TheMarker[2], 8, CV_RGB(0, 0, 250), -1);
}

int HelipadDetector::perimeter(vector< Point > &a) {
    int sum = 0;
    for (unsigned int i = 0; i < a.size(); i++) {
        int i2 = (i + 1) % a.size();
        sum += sqrt((a[i].x - a[i2].x) * (a[i].x - a[i2].x) + (a[i].y - a[i2].y) * (a[i].y - a[i2].y));
    }
    return sum;
}

bool HelipadDetector::FindHelipad(const Mat &in, bool &rotated) { //, int &nRotations) {

    Mat grey;
    if (in.type() == CV_8UC1)
        grey = in;
    else
        cv::cvtColor(in, grey, CV_BGR2GRAY);
    // threshold image
    threshold(grey, grey, 125, 255, THRESH_BINARY | THRESH_OTSU);

    // now, analyze the interior in order to get the id
    //imshow("thres Helipad Cand", grey);

   // Markers  are divided in 7x7 regions, of which the inner 5x5 belongs to marker info
   // the external border shoould be entirely black

   // External Border.
   int nDivH = Th/Mhb;
   int divisionsH = grey.rows / nDivH;
   int nDiv = Tw/Mwb;
   int divisionsW = grey.cols / nDiv;
   for (int y = 0; y < nDivH; y++) {
       int inc = nDiv - 1;
       if (y == 0 || y == nDiv - 1)
           inc = 1; // for first and last row, check the whole border
       for (int x = 0; x < nDiv; x += inc) {
           int Xstart = (x) * (divisionsW);
           int Ystart = (y) * (divisionsH);
           Mat square = grey(Rect(Xstart, Ystart, divisionsW, divisionsH));
           int nZ = countNonZero(square);
           if (nZ > (divisionsW * divisionsH) / 2) {
               return false; // can not be a helipad because the border element is not black!
           }
       }
   }
   //imshow("External Border OK", grey);

   // Inner Border
   Mat InBorder = grey(Rect(divisionsW, divisionsH, grey.rows - 2*divisionsW, grey.cols - 2*divisionsW));
   //imshow("in border", InBorder);
   nDivH = Twh/Mhw;
   divisionsH = InBorder.rows / nDivH;
   nDiv = Tww/Mww;
   divisionsW = InBorder.cols / nDiv;
   for (int y = 0; y < nDivH; y++) {
       int inc = nDiv - 1;
       if (y == 0 || y == nDiv - 1)
           inc = 1; // for first and last row, check the whole border
       for (int x = 0; x < nDiv; x += inc) {
           int Xstart = (x) * (divisionsW);
           int Ystart = (y) * (divisionsH);
           Mat square = InBorder(Rect(Xstart, Ystart, divisionsW, divisionsH));
           int nZ = countNonZero(square);
           if (nZ < (divisionsW * divisionsH) / 2) {
               return false; // can not be a helipad because the second border element is not white!
           }
       }
   }

   rotated = false;
   bool finded = true;

   Mat H = InBorder(Rect(divisionsW, divisionsH, InBorder.rows - 2*divisionsW, InBorder.cols - 2*divisionsW));
   //imshow("H", H);
   int nDivh = Hw/Hsw;
   int divisionsWH = H.cols / nDivh;
   for(int x = 0; x < nDivh; x++)
   {
       int Xstart = (x) * (divisionsWH);
       int Ystart = 0;
       Mat square = H(Rect(Xstart, Ystart, divisionsWH, H.cols));
       int nz = countNonZero(square);
       if (x == 0 || x == nDivh -1)
       {
           if (nz > (divisionsWH * H.cols) / 2)
           {
               finded = false;
               break; //left or right lines (|-|) of the H don't exist!
           }
       }
       else
       {
           if (nz < (divisionsWH * H.cols) / 2)
           {
               finded = false;
               break;
           }
       }
   }

   if(!finded)
   {
       rotated = true;
       Point2f Centre(InBorder.cols/2.0F, InBorder.rows/2.0F);
       Mat rot_mat = getRotationMatrix2D(Centre, 90, 1.0);
       Mat newInBorder;
       warpAffine(InBorder, newInBorder, rot_mat, InBorder.size());

       H = newInBorder(Rect(divisionsW, divisionsH, newInBorder.rows - 2*divisionsW, newInBorder.cols - 2*divisionsW));
       //imshow("rotH", H);
       nDivh = Hw/Hsw;
       divisionsWH = H.cols / nDivh;
       for(int x = 0; x < nDivh; x++)
       {
           int Xstart = (x) * (divisionsWH);
           int Ystart = 0;
           Mat square = H(Rect(Xstart, Ystart, divisionsWH, H.cols));
           int nz = countNonZero(square);
           if (x == 0 || x == nDivh -1)
           {
               if (nz > (divisionsWH * H.cols) / 2)
                   return false; //left and right lines (|-|) of the H don't exist!
           }
           else
           {
               if (nz < (divisionsWH * H.cols) / 2)
                   return false;
           }
       }
   }

   //imshow("H", H);

   return true;

}

bool HelipadDetector::GetFrontHeliCandidate(Mat &in, Mat &out, Size size, vector< Point > points) throw(cv::Exception) {

    if (points.size() != 4)
        throw cv::Exception(9001, "point.size()!=4", "MarkerDetector::warp", __FILE__, __LINE__);
    // obtain the perspective transform
    Point2f pointsRes[4], pointsIn[4];
    for (int i = 0; i < 4; i++)
        pointsIn[i] = points[i];
    pointsRes[0] = (Point2f(0, 0));
    pointsRes[1] = Point2f(size.width - 1, 0);
    pointsRes[2] = Point2f(size.width - 1, size.height - 1);
    pointsRes[3] = Point2f(0, size.height - 1);
    Mat M = getPerspectiveTransform(pointsIn, pointsRes); //Crucial step
    cv::warpPerspective(in, out, M, size, cv::INTER_NEAREST);
    return true;
}

void HelipadDetector::detectRectangles(Mat &thresImgv, vector< vector< Point > > &OutMarkerCanditates) {

    // calcualte the min_max contour sizes
    int minSize = _minSize * std::max(thresImgv.cols, thresImgv.rows) * 4;
    int maxSize = _maxSize * std::max(thresImgv.cols, thresImgv.rows) * 4;

    std::vector< cv::Vec4i > hierarchy2;
    std::vector< std::vector< cv::Point > > contours2;
    cv::Mat thres2;
    thresImgv.copyTo(thres2);
    cv::findContours(thres2, contours2, hierarchy2, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);


    vector< Point > approxCurve;

    vector< vector < Point > > CandidateHelipad; //for now, we are only interested in the 4 corners. If some more is needed a class must be created.
    /// for each contour, analyze if it is a paralelepiped likely to be the marker
    for (unsigned int i = 0; i < contours2.size(); i++) {

        // check it is a possible element by first checking is has enough points
        if (minSize < contours2[i].size() && contours2[i].size() < maxSize) {
        // approximate to a poligon
        approxPolyDP(contours2[i], approxCurve, double(contours2[i].size()) * 0.05, true); //closed poly with straight lines.
        // check that the poligon has 4 points
        if (approxCurve.size() == 4) {
                    // and is convex
                    if (isContourConvex(Mat(approxCurve))) {
                        // 						//ensure that the   distace between consecutive points is large enough
                        float minDist = 1e10;
                        for (int j = 0; j < 4; j++) {
                            float d = std::sqrt((float)(approxCurve[j].x - approxCurve[(j + 1) % 4].x) * (approxCurve[j].x - approxCurve[(j + 1) % 4].x) +
                                                (approxCurve[j].y - approxCurve[(j + 1) % 4].y) * (approxCurve[j].y - approxCurve[(j + 1) % 4].y));
                            if (d < minDist)
                                minDist = d;
                        }
                        // check that distance is not very small
                        if (minDist > 10) {
                            //drawContours(thres2, approxCurve);

                            CandidateHelipad.push_back(approxCurve);

                        }
                    }
                }

        }
    }

    //imshow("Contours", thres2); //All the candidates

    /*
    /// sort the points in anti-clockwise order
    valarray< bool > (false, MarkerCanditates.size()); // used later
    for (unsigned int i = 0; i < MarkerCanditates.size(); i++) {

        // trace a line between the first and second point.
        // if the thrid point is at the right side, then the points are anti-clockwise
        double dx1 = MarkerCanditates[i][1].x - MarkerCanditates[i][0].x;
        double dy1 = MarkerCanditates[i][1].y - MarkerCanditates[i][0].y;
        double dx2 = MarkerCanditates[i][2].x - MarkerCanditates[i][0].x;
        double dy2 = MarkerCanditates[i][2].y - MarkerCanditates[i][0].y;
        double o = (dx1 * dy2) - (dy1 * dx2);

        if (o < 0.0) { // if the third point is in the left side, then sort in anti-clockwise order
            swap(MarkerCanditates[i][1], MarkerCanditates[i][3]);
            swapped[i] = true;
            // sort the contour points
            //  	    reverse(MarkerCanditates[i].contour.begin(),MarkerCanditates[i].contour.end());//????
        }
    }

    /// remove these elements which corners are too close to each other
    // first detect candidates to be removed*/

    vector< pair< int, int > > TooNearCandidates;
    for (unsigned int i = 0; i < CandidateHelipad.size(); i++) {
        // calculate the average distance of each corner to the nearest corner of the other marker candidate
        for (unsigned int j = i + 1; j < CandidateHelipad.size(); j++) {
            valarray< float > vdist(4);
            for (int c = 0; c < 4; c++) {
                vdist[c] = sqrt((CandidateHelipad[i][c].x - CandidateHelipad[j][c].x) * (CandidateHelipad[i][c].x - CandidateHelipad[j][c].x) +
                                (CandidateHelipad[i][c].y - CandidateHelipad[j][c].y) * (CandidateHelipad[i][c].y - CandidateHelipad[j][c].y));
            }
            // if distance is too small
            if (vdist[0] < tooNearCandidatesDistance && vdist[1] < tooNearCandidatesDistance && vdist[2] < tooNearCandidatesDistance && vdist[3] < tooNearCandidatesDistance) {
                TooNearCandidates.push_back(pair< int, int >(i, j));
            }
        }
    }

    // mark for removal the element of  the pair with smaller perimeter
    valarray< bool > toRemove(false, CandidateHelipad.size());
    for (unsigned int i = 0; i < TooNearCandidates.size(); i++) {
        if (perimeter(CandidateHelipad[TooNearCandidates[i].first]) > perimeter(CandidateHelipad[TooNearCandidates[i].second]))
            toRemove[TooNearCandidates[i].second] = true;
        else
            toRemove[TooNearCandidates[i].first] = true;
    }

    // remove the invalid ones
    // finally, assign to the remaining candidates the contour
    OutMarkerCanditates.reserve(CandidateHelipad.size());
    for (size_t i = 0; i < CandidateHelipad.size(); i++) {
        if (!toRemove[i]) {
            OutMarkerCanditates.push_back(CandidateHelipad[i]);
            /*                 OutMarkerCanditates.back().contour=contours2[ MarkerCanditates[i].idx];
            if (swapped[i]) // if the corners where swapped, it is required to reverse here the points so that they are in the same order
                reverse(OutMarkerCanditates.back().contour.begin(), OutMarkerCanditates.back().contour.end()); //????*/
        }
    }
}

void HelipadDetector::GetPose(float markerSizeX, float markerSizeY, cv::Mat camMatrix, cv::Mat distCoeff, Mat &Rvec, Mat &Tvec, vector < Point >  Helipad) //, bool setYPerpendicular)
{

    if (markerSizeX <= 0)
    {
        cout << "Marker X dimension size must be grater than 0" << endl;
        return;
    }
    if (markerSizeY <= 0)
    {
        cout << "Marker Y simension size must be grater than 0" << endl;
        return;
    }
    if (camMatrix.rows == 0 || camMatrix.cols == 0)
    {
        cout << "Camera callibration matrix must be declared" << endl;
        return;
    }

    double halfSizeX = markerSizeX / 2.;
    double halfSizeY = markerSizeY / 2;
    Mat ObjPoints(8, 3, CV_32FC1);
    ObjPoints.at< float >(0, 0) = -halfSizeX;
    ObjPoints.at< float >(0, 1) = halfSizeY;
    ObjPoints.at< float >(0, 2) = 0;
    ObjPoints.at< float >(1, 0) = halfSizeX;
    ObjPoints.at< float >(1, 1) = halfSizeY;
    ObjPoints.at< float >(1, 2) = 0;
    ObjPoints.at< float >(2, 0) = halfSizeX;
    ObjPoints.at< float >(2, 1) = -halfSizeY;
    ObjPoints.at< float >(2, 2) = 0;
    ObjPoints.at< float >(3, 0) = -halfSizeX;
    ObjPoints.at< float >(3, 1) = -halfSizeY;
    ObjPoints.at< float >(3, 2) = 0;
    ObjPoints.at< float >(4, 0) = 0;
    ObjPoints.at< float >(4, 1) = halfSizeY;
    ObjPoints.at< float >(4, 2) = 0;
    ObjPoints.at< float >(5, 0) = halfSizeX;
    ObjPoints.at< float >(5, 1) = 0;
    ObjPoints.at< float >(5, 2) = 0;
    ObjPoints.at< float >(6, 0) = 0;
    ObjPoints.at< float >(6, 1) = -halfSizeY;
    ObjPoints.at< float >(6, 2) = 0;
    ObjPoints.at< float >(7, 0) = -halfSizeX;
    ObjPoints.at< float >(7, 1) = 0;
    ObjPoints.at< float >(7, 2) = 0;

    cv::Mat ImagePoints(8, 2, CV_32FC1);

    // Set image points from the marker
    for (int c = 0; c < 8; c++) {
        ImagePoints.at< float >(c, 0) = (Helipad[c].x);
        ImagePoints.at< float >(c, 1) = (Helipad[c].y);
    }

    Mat raux, taux;

    solvePnP(ObjPoints, ImagePoints, camMatrix, distCoeff, raux, taux, false);
    //solvePnPRansac(ObjPoints, ImagePoints, camMatrix, distCoeff, raux, taux);

    raux.convertTo(Rvec, CV_32F);
    taux.convertTo(Tvec, CV_32F);

    Mat rota;
    Rodrigues(Rvec, rota);

    cout << "the rotation" << endl << rota << endl;


    //Pose = set_P_matrix(RMat, Tvec);

    //cout << Pose << endl;

    /*
    vector <int>  axisX;
    axisX[0] = Pose.at<float> (0,0);
    axisX[1] = Pose.at<float> (1,0);
    axisX[2] = Pose.at<float> (2,0);

    vector <int> axisY;
    axisY[0] = Pose.at<float> (0,1);
    axisY[1] = Pose.at<float> (1,1);
    axisY[2] = Pose.at<float> (2,1);

    vector <int> axisZ;
    axisZ[0] = Pose.at<float> (0,2);
    axisZ[1] = Pose.at<float> (1,2);
    axisZ[2] = Pose.at<float> (2,2);

    vector <int> axisX2d;
    vector <int> axisY2d;
    vector <int> axisZ2d;

    projectPoints(axisX, Rvec, Tvec, CamMatrix, DistMatrix, axisX2d);
    projectPoints(axisY, Rvec, Tvec, CamMatrix, DistMatrix, axisY2d);
    projectPoints(axisZ, Rvec, Tvec, CamMatrix, DistMatrix, axisZ2d);

    Point Xaxis;
    Xaxis.x = axisX2d[0];
    Xaxis.y = axisX2d[1];
    */

    //line(imgOriginal,Helipads[k][3], Helipads[k][0] + Xaxis, CV_RGB(100, 100, 100), 2);
    // rotate the X axis so that Y is perpendicular to the marker plane
    //if (setYPerpendicular)
    //    rotateXAxis(Rvec);
    // cout<<(*this)<<endl;
}

Mat HelipadDetector::set_P_matrix(Mat R_matrix, Mat t_matrix)
{
    Mat _P_matrix;
    _P_matrix.at<double>(0,0) = R_matrix.at<double>(0,0);
    _P_matrix.at<double>(0,1) = R_matrix.at<double>(0,1);
    _P_matrix.at<double>(0,2) = R_matrix.at<double>(0,2);
    _P_matrix.at<double>(1,0) = R_matrix.at<double>(1,0);
    _P_matrix.at<double>(1,1) = R_matrix.at<double>(1,1);
    _P_matrix.at<double>(1,2) = R_matrix.at<double>(1,2);
    _P_matrix.at<double>(2,0) = R_matrix.at<double>(2,0);
    _P_matrix.at<double>(2,1) = R_matrix.at<double>(2,1);
    _P_matrix.at<double>(2,2) = R_matrix.at<double>(2,2);
    _P_matrix.at<double>(0,3) = t_matrix.at<double>(0);
    _P_matrix.at<double>(1,3) = t_matrix.at<double>(1);
    _P_matrix.at<double>(2,3) = t_matrix.at<double>(2);
    return _P_matrix;
}

int HelipadDetector::detect(Mat source, vector < vector < Point > > Helipads)
 {

    Mat imgOriginal = source;
    int nHelis = 0;

         try{

              //imshow("Original", imgOriginal); //Original image captured.

              Mat grey;

              if (imgOriginal.type() == CV_8UC3)
                  cv::cvtColor(imgOriginal, grey, CV_BGR2GRAY);
              else
                  grey = imgOriginal;

              /// Do threshold the image and detect contours
              Mat thresholded;

              switch (thresMethod) {

              case 1: //FIXED_THRES

                  cv::threshold(grey, thresholded, _thresParam1, 255, CV_THRESH_BINARY_INV);
                  break;

              case 2: // ADPT_THRES currently, this is the best method*/

                  // ensure that _thresParam1%2==1
                  if (_thresParam1 < 3)
                      _thresParam1 = 3;
                  else if (((int)_thresParam1) % 2 != 1)
                      _thresParam1 = (int)(_thresParam1 + 1);

                  cv::adaptiveThreshold(grey, thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, _thresParam1, _thresParam2);

              case 3: { //CANNY

                  // this should be the best method, and generally it is.
                  // However, some times there are small holes in the marker contour that makes
                  // the contour detector not to find it properly
                  // if there is a missing pixel
                  cv::Canny(grey, thresholded, 10, 220);
                  // I've tried a closing but it add many more points that some
                  // times makes this even worse
                  // 			  Mat aux;
                  // 			  cv::morphologyEx(thres,aux,MORPH_CLOSE,Mat());
                  // 			  out=aux;

              } break;

              }

              imshow("thres", thresholded); //Image after the threshold.

              vector< vector < Point >  > MarkerCanditates;
              detectRectangles(thresholded, MarkerCanditates);

              for (size_t i = 0; i < MarkerCanditates.size(); i++)
              {

                  Mat CannonicalMarker;
                  bool noError;

                  noError = GetFrontHeliCandidate(grey, CannonicalMarker, Size(_markerWarpSizex, _markerWarpSizey), MarkerCanditates[i] );

                  if(noError)
                  {

                      //imshow("HelipadCandidate", CannonicalMarker);
                      bool success;
                      bool rotated;
                      success = FindHelipad(CannonicalMarker, rotated);
                      if (success)
                      {
                          if(!rotated)
                          {
                              if(MarkerCanditates[i][0].x > MarkerCanditates[i][2].x)
                              {
                                  Point aux = MarkerCanditates[i][0];
                                  MarkerCanditates[i][0] = MarkerCanditates[i][2];
                                  MarkerCanditates[i][2] = aux;
                                  aux = MarkerCanditates[i][1];
                                  MarkerCanditates[i][1] = MarkerCanditates[i][3];
                                  MarkerCanditates[i][3] = aux;
                              }
                              if(MarkerCanditates[i][0].x > MarkerCanditates[i][1].x)
                              {
                                  Point aux = MarkerCanditates[i][0];
                                  MarkerCanditates[i][0] = MarkerCanditates[i][1];
                                  MarkerCanditates[i][1] = aux;
                                  aux = MarkerCanditates[i][2];
                                  MarkerCanditates[i][2] = MarkerCanditates[i][3];
                                  MarkerCanditates[i][3] = aux;
                              }
                              if(MarkerCanditates[i][0].y > MarkerCanditates[i][3].y)
                              {
                                  Point aux = MarkerCanditates[i][0];
                                  MarkerCanditates[i][0] = MarkerCanditates[i][3];
                                  MarkerCanditates[i][3] = aux;
                                  aux = MarkerCanditates[i][1];
                                  MarkerCanditates[i][1] = MarkerCanditates[i][2];
                                  MarkerCanditates[i][2] = aux;
                              }
                              drawContours(imgOriginal, MarkerCanditates[i]);
                              drawPoints(imgOriginal, MarkerCanditates[i]);
                              imshow("HelipadCandidate", CannonicalMarker);
                              Helipads.push_back(MarkerCanditates[i]);
                              nHelis ++;
                              break;
                          }
                          else
                          {
                              Point temp = MarkerCanditates[i][0];
                              MarkerCanditates[i][0] = MarkerCanditates[i][1];
                              MarkerCanditates[i][1] = MarkerCanditates[i][2];
                              MarkerCanditates[i][2] = MarkerCanditates[i][3];
                              MarkerCanditates[i][3] = temp;
                              noError = GetFrontHeliCandidate(grey, CannonicalMarker, Size(_markerWarpSizex, _markerWarpSizey), MarkerCanditates[i] );
                              if (noError)
                              {
                                  if(MarkerCanditates[i][0].x > MarkerCanditates[i][2].x)
                                  {
                                      Point aux = MarkerCanditates[i][0];
                                      MarkerCanditates[i][0] = MarkerCanditates[i][2];
                                      MarkerCanditates[i][2] = aux;
                                      aux = MarkerCanditates[i][1];
                                      MarkerCanditates[i][1] = MarkerCanditates[i][3];
                                      MarkerCanditates[i][3] = aux;
                                  }
                                  if(MarkerCanditates[i][0].x > MarkerCanditates[i][1].x)
                                  {
                                      Point aux = MarkerCanditates[i][0];
                                      MarkerCanditates[i][0] = MarkerCanditates[i][1];
                                      MarkerCanditates[i][1] = aux;
                                      aux = MarkerCanditates[i][2];
                                      MarkerCanditates[i][2] = MarkerCanditates[i][3];
                                      MarkerCanditates[i][3] = aux;
                                  }
                                  if(MarkerCanditates[i][0].y > MarkerCanditates[i][3].y)
                                  {
                                      Point aux = MarkerCanditates[i][0];
                                      MarkerCanditates[i][0] = MarkerCanditates[i][3];
                                      MarkerCanditates[i][3] = aux;
                                      aux = MarkerCanditates[i][1];
                                      MarkerCanditates[i][1] = MarkerCanditates[i][2];
                                      MarkerCanditates[i][2] = aux;
                                  }
                                  drawContours(imgOriginal, MarkerCanditates[i]);
                                  drawPoints(imgOriginal, MarkerCanditates[i]);
                                  threshold(CannonicalMarker, CannonicalMarker, 125, 255, THRESH_BINARY | THRESH_OTSU);
                                  imshow("HelipadCandidate", CannonicalMarker);
                                  Helipads.push_back(MarkerCanditates[i]);
                                  nHelis ++;
                                  break;
                              }
                          }
                      }

                  }

              }


              int Y0 = imgOriginal.rows/2;
              int X0 = imgOriginal.cols/2;
              Point Origin;
              Origin.x = X0;
              Origin.y = Y0;

              //Centre of the image
              //circle(imgOriginal, Origin, 5,  CV_RGB(200, 100, 100), -1 );

              Mat Rvec, Tvec;
              Tvec = Mat::zeros(3, 1, CV_32FC1);
              Rvec = Mat::zeros(3, 1, CV_32FC1);

              if (nHelis == 1) //Solo 1 Helipad
              {
                 for (int k = 0; k < Helipads.size(); k++)
                 {

                     //Opcional, Sacamos mas puntos del Helipad
                     //Como es un cuadrado plano, podemos deducir donde habrá mas puntos:
                     Point Pext = Helipads[k][0] + (Helipads[k][1] - Helipads[k][0])/2;
                     Helipads[k].push_back(Pext);
                     Pext = Helipads[k][1] + (Helipads[k][2] - Helipads[k][1])/2;
                     Helipads[k].push_back(Pext);
                     Pext = Helipads[k][2] + (Helipads[k][3] - Helipads[k][2])/2;
                     Helipads[k].push_back(Pext);
                     Pext = Helipads[k][3] + (Helipads[k][0] - Helipads[k][3])/2;
                     Helipads[k].push_back(Pext);
                     circle(imgOriginal, Helipads[k][4], 8, CV_RGB(250, 250, 250), -1);
                     circle(imgOriginal, Helipads[k][5], 8, CV_RGB(250, 250, 250), -1);
                     circle(imgOriginal, Helipads[k][6], 8, CV_RGB(250, 250, 250), -1);
                     circle(imgOriginal, Helipads[k][7], 8, CV_RGB(250, 250, 250), -1);


                     GetPose(markerSizeX, markerSizeY, CamMatrix, DistMatrix, Rvec, Tvec, Helipads[k]);
                     //GetPose(markerSizeX, markerSizeY, CamMatrix, Mat(), Rvec, Tvec, Helipads[k]);

                     cout << "Pixel Coordinates in the image of the helipad:" << endl;
                     cout << "corner 1: X = " << Helipads[k][0].x - X0 << " Y = " << -(Helipads[k][0].y - Y0) << endl;
                     cout << "corner 2: X = " << Helipads[k][1].x - X0 << " Y = " << -(Helipads[k][1].y - Y0) << endl;
                     cout << "corner 3: X = " << Helipads[k][2].x - X0 << " Y = " << -(Helipads[k][2].y - Y0) << endl;
                     cout << "corner 4: X = " << Helipads[k][3].x - X0 << " Y = " << -(Helipads[k][3].y - Y0) << endl;
                     cout<< "" << endl;
                     cout<< "" << endl;

                     cout << Tvec << endl;
                     Mat Rot;
                     Rodrigues(Rvec, Rot);
                     cout << Rot << endl;

                     float sizex = markerSizeX/2;
                     float sizey = markerSizeY/2;
                     Mat objectPoints(4, 3, CV_32FC1);
                     objectPoints.at< float >(0, 0) = 0;
                     objectPoints.at< float >(0, 1) = 0;
                     objectPoints.at< float >(0, 2) = 0;
                     objectPoints.at< float >(1, 0) = sizex;
                     objectPoints.at< float >(1, 1) = 0;
                     objectPoints.at< float >(1, 2) = 0;
                     objectPoints.at< float >(2, 0) = 0;
                     objectPoints.at< float >(2, 1) = sizey;
                     objectPoints.at< float >(2, 2) = 0;
                     objectPoints.at< float >(3, 0) = 0;
                     objectPoints.at< float >(3, 1) = 0;
                     objectPoints.at< float >(3, 2) = sizex;

                     vector< Point2f > imagePoints;

                     projectPoints(objectPoints, Rvec, Tvec, CamMatrix, DistMatrix, imagePoints);

                     line(imgOriginal, imagePoints[0], imagePoints[1], Scalar(0, 0, 255, 255), 3, CV_AA);
                     line(imgOriginal, imagePoints[0], imagePoints[2], Scalar(0, 255, 0, 255), 3, CV_AA);
                     line(imgOriginal, imagePoints[0], imagePoints[3], Scalar(255, 0, 0, 255), 3, CV_AA);

                     putText(imgOriginal, "x", imagePoints[1], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255, 255), 2);
                     putText(imgOriginal, "y", imagePoints[2], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0, 255), 2);
                     putText(imgOriginal, "z", imagePoints[3], FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0, 255), 2);

                 }
              }

              imshow("Source", imgOriginal);

         }
         catch(Exception e)
         {
             cout << "Excption :" << e.what() << endl;
             return 0;
         }

     return nHelis;

  }
}
