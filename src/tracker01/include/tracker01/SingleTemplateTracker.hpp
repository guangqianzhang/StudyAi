#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "tracker.hpp"

namespace mycv
{
    
    class SingleTemplateTracker : public Tracker
    {
    public:
    enum MatchMethod{SQDIFF=0,SADIFF=1};
    enum MatchStrategy{UNIFORM=0,NORMAL=1};
    struct Params{

        
        int expandWidth;              //局部扩展
        MatchStrategy matchStrategy ;       //匹配策略 1 随机；0 局部
        MatchMethod matchMethod;//匹配方法
        double alpha;//模板更新速度
        int numPoints;//随机采样点数
        Point2i sigma;//正态分布标准差
        Vec2i xyStep;//模板内采样不长
        Vec2i xyStride;//模板在图像内滑动不长
        Params(){
            expandWidth=50;
            matchMethod=MatchMethod::SADIFF;
            matchStrategy=MatchStrategy::NORMAL;
            alpha=0.7;
            numPoints=500;
            sigma=Point2d(0.5,0.5);
            xyStep=Vec2i(2,2);
            xyStride=Vec2i(1,1);
        }

    };
        Mat TargetTemplate;//目标模板
        Rect CurrentBoundingBox;      //当前帧找到的目标框
        Mat CurrentTargetPatch;       //当前帧找到的图像块
        Rect FrameArea;               //视频帧矩形
        Rect NextSearchArea;          //下一帧搜索范围
        vector<Point2d> SamplePoints; //标准正态分布采样点
        Params params;
    private:
        float MatchTemplate(const Mat &src, const Mat &temp, Point2i &match_location,
                            MatchMethod match_method, Vec2i &xy_step, Vec2i &xy_stride)
        {
            CV_Assert((src.type() == CV_8UC1) && (temp.type() == CV_8UC1));
            //图像 模板尺寸
            int src_width = src.cols;
            int src_height = src.rows;
            int tempe_clos = temp.cols;
            int templ_rows = temp.rows;
            int y_end = src_height - templ_rows + 1;
            int x_end = src_width - tempe_clos + 1;

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
                    for (int r = 0; r < templ_rows; r += xy_step[1])
                    {
                        for (int c = 0; c < tempe_clos; c += xy_step[0])
                        {
                            uchar src_val = src.ptr<uchar>(y + r)[x + c];
                            uchar temp_val = temp.ptr<uchar>(r)[c];
                            if (match_method == MatchMethod::SQDIFF) // SODIFF
                                match_yx += float(std::abs(src_val - temp_val) * std::abs(src_val - temp_val));
                            if (match_method == MatchMethod::SADIFF) // SADIFF
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
            match_location = Point2i(x_match, y_match);
            return match_dgree;
        }
        float MatchTemplate(const Mat &src, const Mat &temp, Point2i &match_location,
                            MatchMethod match_method, const vector<Point2d> &sample_points)
        {

            CV_Assert((src.type() == CV_8UC1) && (temp.type() == CV_8UC1));
            //图像 模板尺寸
            int src_width = src.cols;
            int src_height = src.rows;
            int tempe_clos = temp.cols;
            int templ_rows = temp.rows;
            int y_end = src_height - templ_rows + 1;
            int x_end = src_width - tempe_clos + 1;

            //缩放
            vector<Point2i> Sample_Points(sample_points.size());
#pragma omp parallel for
            for (size_t k = 0; k < sample_points.size(); k++)
            {
                const Point2d &ptd = sample_points[k];
                Point2i &pti = Sample_Points[k];
                pti.x = cvRound(ptd.x * tempe_clos);
                pti.y = cvRound(ptd.y * templ_rows);
            }
            //记录最佳匹配
            float match_dgree = FLT_MAX;
            int y_match = -1, x_match = -1;
#pragma omp parallel for
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
                        if (match_method == MatchMethod::SQDIFF)
                        {
                            match_yx += float(std::abs(src_val - temp_val) * std::abs(src_val - temp_val));
                        }
                        if (match_method == MatchMethod::SADIFF)
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
            match_location = Point2i(x_match, y_match);
            return match_dgree;
        }
        //估计下一帧范围
        void EstimateSearchArea(const Rect &target_location, Rect &search_area, int expend_x, int expend_y)
        {
            float center_x = target_location.x + 0.5f * target_location.width;
            float center_y = target_location.y + 0.5f * target_location.height;
            search_area.width = target_location.width + expend_x;
            search_area.height = target_location.height + expend_y;
            search_area.x = int(center_x - 0.5f * search_area.width);
            search_area.y = int(center_y - 0.5f * search_area.height);
            search_area &= this->FrameArea;
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

    public:
        SingleTemplateTracker(Params _params);
        virtual ~SingleTemplateTracker();
        bool init(const Mat &initFrame, const Rect &initBoundingBox)
        {
            cout << "Go SingleTemplateTracker::init!!" << endl;
            this->FrameArea = Rect(0, 0, initFrame.cols, initFrame.rows);
            cout << "initBoundingBox" << initBoundingBox.size() << endl;
            this->TargetTemplate = initFrame(initBoundingBox).clone();
            cout << "TargetTemplate" << TargetTemplate.size() << endl;

            //下一帧搜索范围
            this->EstimateSearchArea(initBoundingBox, this->NextSearchArea, this->params.expandWidth, this->params.expandWidth);
            //初始化随机采样
          

            this->GenerateRandomSamplePoints(this->SamplePoints, this->params.numPoints, this->params.sigma);

            return false;
        }

        bool track(const Mat &currentFrame, Rect &currentBoundingBox)
        {
            cout << "Go SingleTemplateTracker::track!!" << endl;
            // matching tempale 全帧匹配
            Point2i match_location(-1, -1);
            //搜索目标
            
            if (this->params.matchStrategy == MatchStrategy::UNIFORM)
            {

                this->MatchTemplate(currentFrame(this->NextSearchArea), this->TargetTemplate, match_location, 
                this->params.matchMethod, this->params.xyStep, this->params.xyStride);
            }
            if (this->params.matchStrategy == MatchStrategy::NORMAL)
            {
                this->MatchTemplate(currentFrame(this->NextSearchArea), this->TargetTemplate,
                                    match_location,this->params.matchMethod, this->SamplePoints);
            }
            //调整匹配点位置
            match_location.x += this->NextSearchArea.x;
            match_location.y += this->NextSearchArea.y;
            //计算当前位置目标框
            this->CurrentBoundingBox = Rect(match_location.x, match_location.y, this->TargetTemplate.cols, this->TargetTemplate.rows);

            //抓取当前帧的目标图像块
            cout << "CurrentBoundingBox" << CurrentBoundingBox.size() << endl;
            this->CurrentTargetPatch = currentFrame(this->CurrentBoundingBox).clone();

            currentBoundingBox = this->CurrentBoundingBox;
            return false;
        }
        bool update()
        {
            cout << "Go SingleTemplateTracker::update!!" << endl;
            //更新目标表面特征模型 t(k+1)=aT(k)+bT(k-1)
           
            cout << TargetTemplate.size() << endl;
            cout << CurrentTargetPatch.size() << endl;

            cv::addWeighted(this->TargetTemplate, this->params.alpha, this->CurrentTargetPatch, 1.0 - this->params.alpha, 0.0, this->TargetTemplate);

            //更新下一帧上的搜索范围
            this->EstimateSearchArea(this->CurrentBoundingBox, this->NextSearchArea, this->params.expandWidth, this->params.expandWidth);
            return false;
        }
    };

    SingleTemplateTracker::SingleTemplateTracker(Params _params)
    {
        cout << "Go SingleTemplateTracker::SingleTemplateTracker!!" << endl;
        this->params=_params;
    }

    SingleTemplateTracker::~SingleTemplateTracker()
    {
        cout << "Go SingleTemplateTracker::~SingleTemplateTracker!!" << endl;
    }
    typedef SingleTemplateTracker STTracker;
}