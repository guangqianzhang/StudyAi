#include "ros/ros.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
// #include <tracker01/tracker.hpp>

using namespace std;
using namespace cv;
namespace detasets
{
    const string datasets_dir = "";
    const string video1 = datasets_dir + "";
    int video1_start_frame = 100; // qishizhen

    const string video = video1;
    int start_frame = video1_start_frame;
}
namespace global
{
    bool paused = true;
    Mat displayImg;
    bool selectObject = false;
    bool isRoiReady = 0;
    Point origin;
    Rect selectedRoi;

    static void onMouse(int event, int x, int y, int, void *)
    {
        if (selectObject)
        {
            selectedRoi.x = MIN(x, origin.x);
            selectedRoi.y = MIN(x, origin.y);
            selectedRoi.width = std::abs(x - origin.x);
            selectedRoi.height = std::abs(y - origin.y);
            selectedRoi &= Rect(0, 0, displayImg.cols, displayImg.rows); //不能越界
            rectangle(displayImg, selectedRoi, Scalar(0, 0, 255), 1);
        }
        switch (event)
        {
        case CV_EVENT_LBUTTONDOWN:
            origin = Point(x, y);
            selectedRoi = Rect(x, y, 0, 0);
            selectObject = true;
            isRoiReady = false;
            break;
        case CV_EVENT_LBUTTONUP:
            selectObject = false;
            if (selectedRoi.width > 0 && selectedRoi.height > 0)
            {
                isRoiReady = true;
            }
            ROS_INFO("selected well");

            break;
        //暂停/开始
        case CV_EVENT_RBUTTONDOWN:
            paused = !paused;
            break;
        default:
            break;
        }
    }
}
int main(int argc, char *argv[])
{
    ros::init(argc,argv,"match");
    ROS_INFO("hello");
    VideoCapture cap(0);
    const string image_file="/home/zgq/Pictures/zgq/5.13.20.jpeg";

    Mat srcImg=imread(image_file,IMREAD_GRAYSCALE);
    imshow("winName",srcImg);
    Mat displayImg_;
    srcImg.copyTo(displayImg_);

    global::isRoiReady=false;
    global::selectObject=false;
    global::displayImg=displayImg_;

    const string winName="Result Img";
    namedWindow(winName,WINDOW_AUTOSIZE);
    setMouseCallback(winName,global::onMouse,0);

    for (;;)
    {
        cap>>displayImg_;
         imshow(winName,displayImg_);
    }
}