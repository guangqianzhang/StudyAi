
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
// #include <tracker01/tracker.hpp>

using namespace std;
using namespace cv;
#define SODIFF 0
#define SADIFF 1
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

float MatchTemplate(const Mat &src, const Mat &temp, Rect2i &match_location, int match_method, Vec2i &xy_step, Vec2i &xy_stride)
{
    CV_Assert((src.type() == CV_8UC1) && (temp.type() == CV_8UC1));
    //图像 模板尺寸
    int src_width = src.cols;
    int src_height = src.rows;
    int temp_clos = temp.cols;
    int temp_rows = temp.rows;
    int y_end = src_height - temp_rows + 1;
    int x_end = src_width - temp_clos + 1;

    //记录最优位置
    float match_dgree = FLT_MAX;
    int y_match = -1, x_match = -1;
    //扫描
    for (int y = 0; y < y_end; y += xy_stride[1])
    {
        for (int x = 0; x < x_end; x += xy_stride[0])
        {
            //匹配读计算
            float match_yx = 0.0f;
            //对其模板到src，累加模板内像素点差异
            for (int r = 0; r < temp_rows; r += xy_step[1])
            {
                for (int c = 0; c < temp_clos; c += xy_step[0])
                {
                    uchar src_val = src.ptr<uchar>(y + r)[x + c];
                    uchar temp_val = temp.ptr<uchar>(r)[c];
                    if (match_method == SODIFF)
                        match_yx += float(std::abs(src_val - temp_val) * std::abs(src_val - temp_val));
                    if (match_method == SADIFF)
                        match_yx += float(std::abs(src_val - temp_val));
                }
            }
            //更新
            if (match_dgree > match_yx)
            {
                match_dgree = match_yx;
                x_match = x;
                y_match = y;
            }
        }
    }
    match_location = Rect2i(x_match, y_match, temp_clos, temp_rows);
    return match_dgree;
}
float MatchTemplate(const Mat &src, const Mat &temp, Rect2i &match_location, int match_method, const vector<Point2d> &sample_points)
{
    CV_Assert((src.type() == CV_8UC1) && (temp.type() == CV_8UC1));
    //图像 模板尺寸
    int src_width = src.cols;
    int src_height = src.rows;
    int temp_clos = temp.cols;
    int temp_rows = temp.rows;
    int y_end = src_height - temp_rows + 1;
    int x_end = src_width - temp_clos + 1;

    //缩放点击
    vector<Point2i> Sample_Points(sample_points.size());
    for (size_t k = 0; k < sample_points.size(); k++)
    {
        const Point2d &ptd = sample_points[k];
        Point2i &pti = Sample_Points[k];
        pti.x = cvRound(ptd.x * temp_clos);
        pti.y = cvRound(ptd.y * temp_rows);
    }
    //记录最优位置
    float match_dgree = FLT_MAX;
    int y_match = -1, x_match = -1;
    //扫描
    for (int y = 0; y < y_end; y++)
    {
        for (int x = 0; x < x_end; x++)
        {
            //匹配读计算
            float match_yx = 0.0f;
            //按照采样点数组计算模板与原始图像匹配度
            for (size_t k = 0; k < sample_points.size(); k++)
            {
                Point2i &pt = Sample_Points[k];
                uchar src_val = src.ptr<uchar>(y + pt.y)[x + pt.x];
                uchar temp_val = temp.ptr<uchar>(pt.y)[pt.x];
                if (match_method == 0)
                {
                    match_yx += float(std::abs(src_val - temp_val) * std::abs(src_val - temp_val));
                }
                if (match_method == 1)
                {
                    match_yx += float(std::abs(src_val - temp_val));
                }
            }
            //更新
            if (match_dgree > match_yx)
            {
                match_dgree = match_yx;
                x_match = x;
                y_match = y;
            }
        }
    }
    match_location = Rect2i(x_match, y_match, temp_clos, temp_rows);
    return match_dgree;
}

//产生截断正态分布点击
void GenerateRandomSamplePoints(vector<Point2d> &sample_points, int num_points = 1000, Point2d sigma = Point2d(0.3, 0.3))
{
    RNG rng = theRNG();
    Rect2d sample_area(0.0, 0.0, 1.0, 1.0);
    for (int k = 0; k < num_points;)
    {
        Point2d pt;
        pt.x = sample_area.width / 2 + rng.gaussian(sigma.x);
        pt.y = sample_area.height / 2 + rng.gaussian(sigma.y);
        if (sample_area.contains(pt))
        {
            sample_points.push_back(pt);
            k++;
        }
    }
}
//指定倍率缩放矩形区域
void ResizeRect(const Rect &srcRect, Rect &dstRect, double fx, double fy)
{
    double center_x = (srcRect.x + srcRect.width / 2.0) * fx;
    double center_y = (srcRect.y + srcRect.height / 2.0) * fy;
    double dstWidth = srcRect.width * fx;
    double dstHight = srcRect.height * fy;
    int top_left_x = cvRound(center_x - dstWidth / 2.0);
    int top_left_y = cvRound(center_y - dstHight / 2.0);
    dstRect = Rect(top_left_x, top_left_y, cvRound(dstWidth), cvRound(dstHight));
}
//产生多尺度目标模板
void GenerateMultiScaleTargetTempletes(const Mat &origin_target, vector<Mat> &multiscale_target)
{
    vector<double> resize_scales = {1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5};
    multiscale_target.resize(resize_scales.size(), Mat());
    for (size_t scidx = 0; scidx < resize_scales.size(); scidx++)
    {
        cv::resize(origin_target, multiscale_target[scidx], Size(),
                   resize_scales[scidx], resize_scales[scidx], InterpolationFlags::INTER_AREA);
    }
    return;
}
void ShowMultiScaleTemplate(const vector<Mat> &multiscale_target)
{
    int total_cols = 0, total_rows = 0;
    vector<Rect2i> target_rois(multiscale_target.size());
    for (size_t k = 0; k < multiscale_target.size(); k++)
    {
        target_rois[k] = Rect2i(total_cols, 0, multiscale_target[k].cols, multiscale_target[k].rows);
        total_cols += multiscale_target[k].cols;
        total_rows = max(multiscale_target[k].rows, total_rows);
    }
    Mat targetsImg = Mat::zeros(total_rows, total_cols, CV_8UC1);
    for (size_t k = 0; k < multiscale_target.size(); k++)
    {
        multiscale_target[k].copyTo(targetsImg(target_rois[k]));
    }
    imshow("Targets Image", targetsImg);
    waitKey(100);
}
//使用多尺度模板匹配
float MatchMultiScaleTemplates(const Mat& src,const vector<Mat>& multiscale_templs,Rect2i& best_match_location,
int match_method,const vector<Point2d>& sample_points,int match_strategy=1){
    CV_Assert(match_strategy==0||match_strategy==1);

    float bestMatchDgree=FLT_MAX;
    Rect bestMatchLocation;
    Rect matchLocation;
    float matchDgree;

    for(size_t scaleIdx=0;scaleIdx<multiscale_templs.size();scaleIdx++){
        const Mat& templ=multiscale_templs[scaleIdx];
        if(match_strategy==0){
            Vec2i xy_step(1,1);
            Vec2i xy_stride(2,2);
            matchDgree=MatchTemplate(src,templ,matchLocation,match_method,xy_step,xy_stride);
        }
        if(match_strategy==1){
            matchDgree=MatchTemplate(src,templ,matchLocation,match_method,sample_points);
        }
        //记录最佳匹配
        if(matchDgree<bestMatchDgree){
            bestMatchDgree=matchDgree;
            bestMatchLocation=matchLocation;
        }
    }//endof scaleIdx
    best_match_location=bestMatchLocation;
    return bestMatchDgree;
}

int main(int argc, char **argv)
{
    // 打开摄像头
    // VideoCapture capture(0);

    // 打开文件
    VideoCapture capture;
    capture.open("/home/zgq/Pictures/realcountry.mp4");
    if (!capture.isOpened())
    {
        printf("could not read this video file...\n");
        return -1;
    }
    Size S = Size((int)capture.get(CAP_PROP_FRAME_WIDTH),
                  (int)capture.get(CAP_PROP_FRAME_HEIGHT));
    int fps = capture.get(CAP_PROP_FPS);
    printf("current fps : %d \n", fps);
    // VideoWriter writer("C:/Users/Dell/Desktop/picture/test.mp4", CAP_OPENCV_MJPEG, fps, S, true);

    Mat frame;
    Mat displayImg_;
    Mat workFrame;
    capture.read(frame);
    // frame = imread("/home/zgq/Pictures/zgq/5.13.20.jpeg", IMREAD_GRAYSCALE);
    frame.copyTo(displayImg_);
    cvtColor(frame, workFrame, CV_BGR2GRAY);
    global::isRoiReady = false;
    global::selectObject = false;
    global::displayImg = displayImg_;
    // cvtColor(displayImg_, displayImg_, CV_BGR2GRAY);
    const string winName = "camera-demo";
    namedWindow(winName, WINDOW_AUTOSIZE);
    setMouseCallback(winName, global::onMouse, 0);

    int npoints = 1000;
    Point2d sigma(0.2, 0.2);
    vector<Point2d> sample_points;
    GenerateRandomSamplePoints(sample_points, npoints, sigma);
    CV_Assert(sample_points.size() == npoints);

    //循环显示图像，等待ROI
    for (;;)
    {
        imshow(winName, displayImg_);
        // cvtColor(displayImg_, workFrame, CV_BGR2GRAY);
        if (global::isRoiReady)
        {

            //显示模板
            global::isRoiReady = false;
            Rect roiRect = global::selectedRoi;
            Mat roiImg = workFrame(roiRect).clone();
            imshow("ROI Image", roiImg);
            //多尺度目标模板
            vector<Mat> multiscale_targets;
            GenerateMultiScaleTargetTempletes(roiImg, multiscale_targets);
            ShowMultiScaleTemplate(multiscale_targets);
            //噪声
            Mat noiseImg(workFrame.size(), workFrame.type());
            cv::randn(noiseImg, Scalar(0), Scalar(30));
            Mat workImg = noiseImg + workFrame;
            workImg.copyTo(displayImg_);
            rectangle(workImg, global::selectedRoi, Scalar::all(0), 4);
            imshow(winName, displayImg_);
            //缩放原始图像 再进行匹配
            vector<double> resize_ratios = {1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5};
            for (size_t k = 0; k < resize_ratios.size(); k++)
            {
                Mat resizedWorkImag, resizedDisplayImg;
                cv::resize(workFrame, resizedWorkImag, Size(), resize_ratios[k], resize_ratios[k], 1);
                cv::resize(displayImg_, resizedDisplayImg, Size(), resize_ratios[k], resize_ratios[k], 1);
                Rect resizedSelection;
                ResizeRect(global::selectedRoi, resizedSelection, resize_ratios[k], resize_ratios[k]);
                rectangle(resizedDisplayImg, resizedSelection, Scalar::all(0), 4);

                imshow(winName, resizedDisplayImg);
                waitKey(25);

                //模板在噪声中匹配
                Rect2i match_location;
                int match_method = 1;
                int match_strategy = 1;
                //多尺度模板匹配
                MatchMultiScaleTemplates(resizedWorkImag,multiscale_targets,match_location,
                match_method,sample_points,match_strategy);
/*                  if (match_strategy == 0)
                {
                    Vec2i xv_step(2, 2), xy_stride(2, 2);

                    float match_dgree = MatchTemplate(resizedDisplayImg, roiImg, match_location, match_method, xv_step, xy_stride);
                }
                // cout << "匹配度" << match_dgree << endl;
                // cout << "match ROI" << matchRoi << endl;
                if (match_strategy == 1)
                {

                    vector<Point2i> Sample_Points;
                    float match_dgree = MatchTemplate(resizedDisplayImg, roiImg, match_location, match_method, sample_points);
                    //在roi上显示采样点
                    for (size_t k = 0; k < Sample_Points.size(); k++)
                    {
                        circle(roiImg, Sample_Points[k], 1, Scalar(0, 100, 255));
                    }
                }
                imshow("ROI Image", roiImg);  */
                //显示匹配结果
                Rect matchRoi(match_location);
                rectangle(resizedDisplayImg, matchRoi, Scalar::all(255), 2);

                imshow(winName, resizedDisplayImg);
            }
        }
        char c = waitKey(50);
        if (c == 27)
        {
            break;
        }
    }
    /*     while (capture.read(frame))
        {

            imshow("camera-demo", frame);
            // writer.write(frame);

            char c = waitKey(50);
            if (c == 27)
            {
                break;
            }
        } */
    capture.release();
    // writer.release();
    waitKey(0);
    return 0;
}
