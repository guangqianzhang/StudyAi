#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
namespace mycv
{
    struct DataSet
    {
        string video_name;
        int start_frame;
        Rect2i start_roi;
        bool lock_roi;
        DataSet(const string _videofile, int _start_frame, Rect2i _start_roi, bool _lock_roi)
        {
            video_name = _videofile;
            start_frame = _start_frame;
            start_roi = _start_roi;
            lock_roi = _lock_roi;
        }
    };
    const string datasets_dir="/home/zgq";
    const string video1=datasets_dir+"/Pictures/realcountry.mp4";
    const string video2=datasets_dir+"/Documents/Tracker/yolov4-deepsort/data/video/test.mp4";

    DataSet dataset11(video1,30,Rect2i(),false);
    DataSet dataset111(video1,30,Rect2i(155,139,37,80),true);
    DataSet dataset112(video1,30,Rect2i(227,152,21,47),true);
    DataSet dataset12(video1,80,Rect2i(),false);

    DataSet dataset21(video2,000,Rect2i(),false);
    DataSet dataset22(video2,400,Rect2i(),false);
    DataSet dataset221(video2,400,Rect2i(547,280,24,22),true);
    DataSet dataset23(video2,430,Rect2i(),false);


}
