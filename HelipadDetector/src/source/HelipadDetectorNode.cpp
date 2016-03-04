//////////////////////////////////////////////////////
//  HelipadDetectorNode.cpp
//
//  Created on: March 3, 2016
//      Author: Fernando Pastor
//
//  Last modification on: March 3, 2016
//      Author: Fernando Pastor
//
//////////////////////////////////////////////////////



//I/O stream
//std::cout
#include <iostream>


//Opencv
#include <opencv2/opencv.hpp>

//ROS
#include "ros/ros.h"

//Aruco Eye
#include "HelipadDetector.h"


#include "nodes_definition.h"


using namespace std;






int main(int argc,char **argv)
{
    //Ros Init
    ros::init(argc, argv, MODULE_NAME_HELIPAD);
    ros::NodeHandle n;

    cout<<"[ROSNODE] Starting "<<ros::this_node::getName()<<endl;

    HelipadDetector MyHelipadDetector;
    MyHelipadDetector.open(n);



    try
    {
        ros::spin();
    }
    catch (std::exception &ex)
    {
        std::cout<<"[ROSNODE] Exception :"<<ex.what()<<std::endl;
    }

    return 1;
}
