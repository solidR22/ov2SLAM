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


#include <iostream>
#include <string>
#include <chrono>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <sophus/se3.hpp>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>

#include "profiler.hpp"

class SlamParams {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SlamParams() {}
    
    SlamParams(const cv::FileStorage &fsSettings);

    void reset();

    //=====================================================
    // Variables relative to the current state of the SLAM
    //=====================================================

    bool blocalba_is_on_ = false;      // 局部BA是否正在进行
    bool blc_is_on_ = false;           // 正在闭环
    bool bvision_init_ = false;        // 是否初始化
    bool breset_req_ = false;
    bool bforce_realtime_ = false;

    //=====================================================
    // Variables relative to the setup used for the SLAM
    //=====================================================

    std::string save_path;

    // Calibration parameters (TODO: Get Ready to store all of these in a vector to handle N camera)
    std::string cam_left_topic_, cam_right_topic_;
    std::string cam_left_model_, cam_right_model_;

    double fxl_, fyl_, cxl_, cyl_;
    double k1l_, k2l_, p1l_, p2l_;

    double fxr_, fyr_, cxr_, cyr_;
    double k1r_, k2r_, p1r_, p2r_;

    double img_left_w_, img_left_h_;
    double img_right_w_, img_right_h_;

    // Extrinsic parameters
    Sophus::SE3d T_left_right_;

    // SLAM settings
    bool debug_, log_timings_;

    bool mono_, stereo_;          // 单目还是双目模式

    bool slam_mode_;

    bool buse_loop_closer_;       // 是否开启回环
    int lckfid_ = -1;             // 闭环线程中正在处理的关键帧ID

    float finit_parallax_;        // 用于判定关键帧的视差 
    
    bool bdo_stereo_rect_;        // 是否进行立体矫正，文件中为0
    double alpha_;

    bool bdo_undist_;             // 是否对图像去畸变

    // Keypoints Extraction
    bool use_fast_, use_shi_tomasi_, use_brief_;
    bool use_singlescale_detector_;
    
    int nbmaxkps_;                     // 每一帧特征点的最大数量
    int nmaxdist_;                     //特征点的最大/小距离
    double dmaxquality_;               // ? used for gftt or singlescale
    int nfast_th_;                     // ? used for gftt or singlescale

    // Image Processing
    bool use_clahe_;                   // 使用图像预处理，文件设置为1
    float fclahe_val_;                 // 对比度限制参数，控制每个像素的对比度增强程度，较大的值会导致更强烈的对比度增强

    // KLT Parameters
    bool do_klt_, klt_use_prior_;      // 使用光流，优先使用3D点追踪
    bool btrack_keyframetoframe_;      // 计算关键帧到当前帧的光流，文件设置为false
    int nklt_win_size_, nklt_pyr_lvl_; // 从0开始的最大金字塔等级编号，文件设置为3
    cv::Size klt_win_size_;            // 光流算法的窗口大小，文件设置为9

    float fmax_fbklt_dist_;            // ?最大光流距离
    int nmax_iter_;                    // 最大迭代次数
    float fmax_px_precision_;          // 迭代最大相对变化阈值

    int nklt_err_;                     // ?光流误差

    // Matching th.
    bool bdo_track_localmap_;          // 跟踪局部地图，文件设置为1
    
    float fmax_desc_dist_;             // ?局部地图跟踪的
    float fmax_proj_pxdist_;           // 局部地图跟踪的最大特征点像素距离误差

    // Error thresholds
    bool doepipolar_; // 极线去除外点，开
    bool dop3p_;      // 进行p3p,关
    bool bdo_random; // RANDOMIZE RANSAC?
    float fransac_err_;
    int nransac_iter_;
    float fepi_th_;

    float fmax_reproj_err_;  // 立体匹配时的空间点投影到像素平面的误差
    bool buse_inv_depth_;

    // Bundle Adjustment Parameters
    // (mostly related to Ceres options)
    float robust_mono_th_;
    float robust_stereo_th_;

    bool use_sparse_schur_; // If False, Dense Schur used
    bool use_dogleg_; // If False, Lev.-Marq. used
    bool use_subspace_dogleg_; // If False, Powell's trad. Dogleg used
    bool use_nonmonotic_step_;

    // Estimator parameters
    bool apply_l2_after_robust_; // If true, a L2 optim is applied to refine the results from robust cost function

    int nmin_covscore_; // Number of common observations req. for opt. a KF in localBA

    // Map Filtering parameters
    float fkf_filtering_ratio_;   // 优化线程，该关键帧观测到的"95%"的3D点已经被至少其他4个关键帧观测到

    // Final BA
    bool do_full_ba_;
};
