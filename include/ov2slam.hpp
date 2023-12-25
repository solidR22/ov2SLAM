/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/
#pragma once


#include <vector>
#include <queue>
#include <mutex>
#include <string>

#include "slam_params.hpp"
#include "ros_visualizer.hpp"

#include "logger.hpp"

#include "camera_calibration.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"

#include "frame.hpp"
#include "map_manager.hpp"
#include "visual_front_end.hpp"
#include "mapper.hpp"
#include "estimator.hpp"

class SlamManager {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SlamManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<RosVisualizer> pviz);

    void run();

    bool getNewImage(cv::Mat &iml, cv::Mat &imr, double &time);

    // SLAM系统接口，输入双目图像
    void addNewStereoImages(const double time, cv::Mat &im0, cv::Mat &im1); 
    void addNewMonoImage(const double time, cv::Mat &im0);

    // 读取配置文件，初始化相机参数
    void setupCalibration();
    // 计算立体矫正参数
    void setupStereoCalibration();
    
    void reset();

    void writeResults();

    void writeFullTrajectoryLC();

    void visualizeAtFrameRate(const double time);
    void visualizeFrame(const cv::Mat &imleft, const double time);
    void visualizeVOTraj(const double time);

    void visualizeAtKFsRate(const double time);
    void visualizeCovisibleKFs(const double time);
    void visualizeFullKFsTraj(const double time);
    
    void visualizeFinalKFsTraj();

    int frame_id_ = -1;                             // 帧ID
    bool bnew_img_available_ = false;               // 有新图像输入

    bool bexit_required_ = false;                   // 是否退出的标志
    bool bis_on_ = false;
    
    bool bframe_viz_ison_ = false;
    bool bkf_viz_ison_ = false;

    std::shared_ptr<SlamParams> pslamstate_;        // 配置文件参数
    std::shared_ptr<RosVisualizer> prosviz_;        // ROS发布

    std::shared_ptr<CameraCalibration> pcalib_model_left_;  // 相机模型左
    std::shared_ptr<CameraCalibration> pcalib_model_right_; // 相机模型右

    std::shared_ptr<Frame> pcurframe_;                      // 当前帧

    std::shared_ptr<MapManager> pmap_;

    std::unique_ptr<VisualFrontEnd> pvisualfrontend_;
    std::unique_ptr<Mapper> pmapper_;

    std::shared_ptr<FeatureExtractor> pfeatextract_;        // 特征点提取
    std::shared_ptr<FeatureTracker> ptracker_;              // 追踪

    std::queue<cv::Mat> qimg_left_, qimg_right_;            // 存放图像的队列
    std::queue<double> qimg_time_;                          // 存放图像时间的队列

    std::mutex img_mutex_;
};