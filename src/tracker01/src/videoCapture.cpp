#include "ros/ros.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <tracker01/tracker.hpp>
#include <tracker01/SingleTemplateTracker.hpp>
#include <tracker01/MultiTemplateTracker.hpp>
#include <tracker01/datasets.h>

using namespace std;
using namespace cv;

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
    ros::init(argc, argv, "tracker");
    ros::NodeHandle nh("~");
    ROS_INFO("hallo");
    int looprate_, cap_num_;
    nh.param("looprate", looprate_, 30);
    nh.param("cap_num", cap_num_, 2);
    ros::Rate looprate(looprate_);
    //指定数据
    mycv::DataSet dataset = mycv::dataset21;

    VideoCapture capture;
    capture.open(dataset.video_name);
    CV_Assert(capture.isOpened());
    const int FrameCount = (int)capture.get(VideoCaptureProperties::CAP_PROP_FRAME_COUNT);
    const int FrameWidth = (int)capture.get(VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
    const int FrameHeght = (int)capture.get(VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
    const Rect FrameArea(0, 0, FrameWidth, FrameHeght);
    cout << "FrameCount" << FrameCount << endl;

    int frameIndex = dataset.start_frame;
    Mat currentFrame;
    Mat workFrame;
    capture.set(VideoCaptureProperties::CAP_PROP_POS_FRAMES, double(frameIndex));

    const string winName = "Trackering";
    namedWindow(winName, 1);

    setMouseCallback(winName, global::onMouse, 0);
    // looprate.sleep();

    capture >> currentFrame;
    CV_Assert(!currentFrame.empty());
    ROS_INFO("now frame:%d", frameIndex);
    frameIndex++;
    //在起始帧上选择区域
    while (!global::isRoiReady)
    {
        currentFrame.copyTo(global::displayImg);
        //按下鼠标左键和抬起之间 selectObjrct=true
        if (global::selectObject && global::selectedRoi.width > 0 && global::selectedRoi.height > 0)
        {
            Mat roi_img(global::displayImg, global::selectedRoi);
            cv::bitwise_not(roi_img, roi_img); //反转显示
        }
        imshow(winName, global::displayImg);
        if (waitKey(30) >= 0)
        {
            break;
        }
    }
    //如果lockroi==true 鼠标选择无效
    if (dataset.lock_roi)
        global::selectedRoi = dataset.start_roi;

    cout << "声明跟踪对象实例化，初始化目标跟踪器。。。" << endl;
    /*     mycv::STTracker::Params params=mycv::STTracker::Params();
    params.alpha=0.7; 
    Ptr<mycv::Tracker> tracker = new mycv::SingleTemplateTracker(params); //父类下城，调用字类
    */
    mycv::MTTracker::Params mtparams=mycv::MTTracker::Params();
    mtparams.expandWidth=80;
    mtparams.sigma=Point2i(0.5,0.5);//越大越均匀
    mtparams.numPoints=800;
    mtparams.alpha=0.7;
    Ptr<mycv::Tracker> tracker = new mycv::MTTracker(mtparams); //父类下城，调用字类
    cvtColor(currentFrame, workFrame, CV_BGR2GRAY);
    tracker->init(workFrame, global::selectedRoi);

    cout << "单击鼠标右键开始跟踪" << endl;
    for (;;)
    {
        cout << "into for " << endl;
        if (!global::paused)
        {
            cout << "not pause!" << endl;
            capture >> currentFrame;
            CV_Assert(!currentFrame.empty());
            ROS_INFO("now frame:%d", frameIndex);

            frameIndex++;

            currentFrame.copyTo(global::displayImg);
            cvtColor(currentFrame, workFrame, CV_BGR2GRAY);
            //开始跟踪
            Rect CurrentBoundingBox; //传出参数
            tracker->track(workFrame, CurrentBoundingBox);
            //显示当前跟踪结果
            rectangle(global::displayImg, CurrentBoundingBox, Scalar(0, 0, 255), 2);
            imshow(winName, global::displayImg);
            if (waitKey(30) >= 0)
            {
                break;
            }
            //更新模型
            Rect searchBox;
            tracker->update(searchBox);
        }
        else
        {
            cout << "pause" << endl;
            imshow(winName, global::displayImg);
            if (waitKey(300) == 27)
                break;
        }
    }
    capture.release();
    return 0;
}
