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

#include <opencv2/highgui.hpp>

#include "multi_view_geometry.hpp"

#include "map_manager.hpp"

MapManager::MapManager(std::shared_ptr<SlamParams> pstate, std::shared_ptr<Frame> pframe, std::shared_ptr<FeatureExtractor> pfeatextract, std::shared_ptr<FeatureTracker> ptracker)
    : nlmid_(0), nkfid_(0), nblms_(0), nbkfs_(0), pslamstate_(pstate), pfeatextract_(pfeatextract), ptracker_(ptracker), pcurframe_(pframe)
{
    pcloud_.reset( new pcl::PointCloud<pcl::PointXYZRGB>() );
    pcloud_->points.reserve(1e5);
}


// This function turn the current frame into a Keyframe.
// Keypoints extraction is performed and the related MPs and
// the new KF are added to the map.
// im为预处理后的，imraw为原始图
void MapManager::createKeyframe(const cv::Mat &im, const cv::Mat &imraw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.FE_createKeyframe");

    // Prepare Frame to become a KF
    // (Update observations between MPs / KFs)
    prepareFrame();

    // Detect in im and describe in imraw
    extractKeypoints(im, imraw);

    // Add KF to the map
    addKeyframe();

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.FE_createKeyframe");
}

// Prepare Frame to become a KF
// (Update observations between MPs / KFs)
void MapManager::prepareFrame()
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_prepareFrame");

    // Update new KF id
    pcurframe_->kfid_ = nkfid_;

    // Filter if too many kps，当前帧的特征点数量大于最大数量
    if( (int)pcurframe_->nbkps_ > pslamstate_->nbmaxkps_ ) {
        // 遍历每一个网格中的特征点，取出每一个网格
        for( const auto &vkpids : pcurframe_->vgridkps_ ) {
            // 该网格中的特征点数量大于2
            if( vkpids.size() > 2 ) {
                int lmid2remove = -1; // 记录观测次数最少的特征点对应的空间点ID
                // 初始化为最大值，记录每个网格内被观测次数最少的特征点
                size_t minnbobs = std::numeric_limits<size_t>::max();
                // 取出网格中的每一个特征点ID
                for( const auto &lmid : vkpids ) {
                    // 取出对应的地图点
                    auto plmit = map_plms_.find(lmid);
                    if( plmit != map_plms_.end() ) {
                        // 得到能观测到这个特征点的关键帧数量
                        size_t nbobs = plmit->second->getKfObsSet().size();
                        if( nbobs < minnbobs ) {
                            lmid2remove = lmid;
                            minnbobs = nbobs;
                        }
                    } else {
                        // 地图中没有找到这个特征点，直接从当前帧中删除
                        removeObsFromCurFrameById(lmid);
                        break;
                    }
                }
                if( lmid2remove >= 0 ) {
                    // 删除观测最少的点
                    removeObsFromCurFrameById(lmid2remove);
                }
            }
        }
    }
    // 遍历当前帧的特征点，如果没有对应的地图点，直接从当前帧删除
    for( const auto &kp : pcurframe_->getKeypoints() ) {

        // Get the related MapPoint
        auto plmit = map_plms_.find(kp.lmid_);
        
        if( plmit == map_plms_.end() ) {
            removeObsFromCurFrameById(kp.lmid_);
            continue;
        }

        // Relate new KF id to the MP
        plmit->second->addKfObs(nkfid_);
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_prepareFrame");
}
// 更新共视信息，在Frame中插入共视关键帧和局部地图点信息（是共视关键帧的点，但不是当前帧的点）
void MapManager::updateFrameCovisibility(Frame &frame)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_updateFrameCovisilbity");

    // Update the MPs and the covisilbe graph between KFs
    std::map<int,int> map_covkfs;  // 当前关键帧的共视关键帧<共视关键帧ID, 共视特征点个数>
    std::unordered_set<int> set_local_mapids;  // 是共视帧的3D点，但不是当前帧的特征点
    // 取出这个当前关键帧的所有特征点
    for( const auto &kp : frame.getKeypoints() ) {

        // Get the related MP
        auto plmit = map_plms_.find(kp.lmid_);
        
        if( plmit == map_plms_.end() ) {
            removeMapPointObs(kp.lmid_, frame.kfid_);
            removeObsFromCurFrameById(kp.lmid_);
            continue;
        }

        // Get the set of KFs observing this KF to update 
        // covisible KFs
        // 返回这个地图点的共视关键帧ID
        for( const auto &kfid : plmit->second->getKfObsSet() ) 
        {
            if( kfid != frame.kfid_ ) 
            {
                auto it = map_covkfs.find(kfid);
                if( it != map_covkfs.end() ) {
                    it->second += 1;
                } else {
                    map_covkfs.emplace(kfid, 1);
                }
            }
        }
    }

    // Update covisibility for covisible KFs
    std::set<int> set_badkfids; // 不好的共视关键帧
    // 遍历共视关键帧
    for( const auto &kfid_cov : map_covkfs ) 
    {
        int kfid = kfid_cov.first;
        int covscore = kfid_cov.second;
        // 取出共视关键帧
        auto pkfit = map_pkfs_.find(kfid);
        if( pkfit != map_pkfs_.end() ) 
        {
            // Will emplace or update covisiblity
            pkfit->second->map_covkfs_[frame.kfid_] = covscore;

            // Set the unobserved local map for future tracking
            // 取出共视关键帧的所有的3D点
            for( const auto &kp : pkfit->second->getKeypoints3d() ) {
                // 如果不是当前帧的特征点
                if( !frame.isObservingKp(kp.lmid_) ) {
                    set_local_mapids.insert(kp.lmid_);
                }
            }
        } else { // 不存在这个关键帧
            set_badkfids.insert(kfid);
        }
    }
    // 去除不好的共视关键帧
    for( const auto &kfid : set_badkfids ) {
        map_covkfs.erase(kfid);
    }
    
    // Update the set of covisible KFs
    frame.map_covkfs_.swap(map_covkfs);

    // Update local map of unobserved MPs
    if( set_local_mapids.size() > 0.5 * frame.set_local_mapids_.size() ) {
        frame.set_local_mapids_.swap(set_local_mapids);
    } else {
        frame.set_local_mapids_.insert(set_local_mapids.begin(), set_local_mapids.end());
    }
    
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_updateFrameCovisilbity");
}

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create MPs
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ ) {
        // Add keypoint to current frame
        frame.addKeypoint(vpts.at(i), nlmid_);

        // Create landmark with same id
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
        addMapPoint(col);
    }
}

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, 
    const std::vector<int> &vscales, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        // Add keypoint to current frame
        frame.addKeypoint(vpts.at(i), nlmid_, vscales.at(i));

        // Create landmark with same id
        cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
        addMapPoint(col);
    }
}

void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_, vdescs.at(i));

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x); // 像素值
            addMapPoint(vdescs.at(i), col);
        } 
        else {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_);

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}


void MapManager::addKeypointsToFrame(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const std::vector<int> &vscales, const std::vector<float> &vangles, 
                        const std::vector<cv::Mat> &vdescs, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    
    // Add keypoints + create landmarks
    size_t nbpts = vpts.size();
    for( size_t i = 0 ; i < nbpts ; i++ )
    {
        if( !vdescs.at(i).empty() ) {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_, vdescs.at(i), vscales.at(i), vangles.at(i));

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(vdescs.at(i), col);
        } 
        else {
            // Add keypoint to current frame
            frame.addKeypoint(vpts.at(i), nlmid_);

            // Create landmark with same id
            cv::Scalar col = im.at<uchar>(vpts.at(i).y,vpts.at(i).x);
            addMapPoint(col);
        }
    }
}

// Extract new kps into provided image and update cur. Frame
void MapManager::extractKeypoints(const cv::Mat &im, const cv::Mat &imraw)
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("2.FE_CF_extractKeypoints");
    // 获取当前帧的特征点
    std::vector<Keypoint> vkps = pcurframe_->getKeypoints();

    std::vector<cv::Point2f> vpts; // 存放当前帧特征点的像素坐标
    std::vector<int> vscales;
    std::vector<float> vangles;

    for( auto &kp : vkps ) {
        vpts.push_back(kp.px_);
    }

    if( pslamstate_->use_brief_ ) {
        describeKeypoints(imraw, vkps, vpts);
    }

    int nb2detect = pslamstate_->nbmaxkps_ - pcurframe_->noccupcells_;

    if( nb2detect > 0 ) {
        // Detect kps in the provided images
        // using the cur kps and img roi to set a mask
        std::vector<cv::Point2f> vnewpts;

        if( pslamstate_->use_shi_tomasi_ ) {
            vnewpts = pfeatextract_->detectGFTT(im, vpts, pcurframe_->pcalib_leftcam_->roi_mask_, nb2detect);
        } 
        else if( pslamstate_->use_fast_ ) {
            vnewpts = pfeatextract_->detectGridFAST(im, pslamstate_->nmaxdist_, vpts, pcurframe_->pcalib_leftcam_->roi_rect_);
        } 
        else if ( pslamstate_->use_singlescale_detector_ ) {
            vnewpts = pfeatextract_->detectSingleScale(im, pslamstate_->nmaxdist_, vpts, pcurframe_->pcalib_leftcam_->roi_rect_);
        } else {
            std::cerr << "\n Choose a detector between : gftt / FAST / SingleScale detector!";
            exit(-1);
        }

        if( !vnewpts.empty() ) {
            if( pslamstate_->use_brief_ ) {
                std::vector<cv::Mat> vdescs;
                vdescs = pfeatextract_->describeBRIEF(imraw, vnewpts);
                addKeypointsToFrame(im, vnewpts, vdescs, *pcurframe_);
            } 
            else if( pslamstate_->use_shi_tomasi_ || pslamstate_->use_fast_ 
                || pslamstate_->use_singlescale_detector_ ) 
            {
                addKeypointsToFrame(im, vnewpts, *pcurframe_);
            }
        }
    }

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "2.FE_CF_extractKeypoints");
}


// Describe cur frame kps in cur image
void MapManager::describeKeypoints(const cv::Mat &im, const std::vector<Keypoint> &vkps, const std::vector<cv::Point2f> &vpts, const std::vector<int> *pvscales, std::vector<float> *pvangles)
{
    size_t nbkps = vkps.size();
    std::vector<cv::Mat> vdescs;

    if( pslamstate_->use_brief_ ) {
        vdescs = pfeatextract_->describeBRIEF(im, vpts);
    }

    assert( vkps.size() == vdescs.size() );

    for( size_t i = 0 ; i < nbkps ; i++ ) {
        if( !vdescs.at(i).empty() ) {
            pcurframe_->updateKeypointDesc(vkps.at(i).lmid_, vdescs.at(i));
            map_plms_.at(vkps.at(i).lmid_)->addDesc(pcurframe_->kfid_, vdescs.at(i));
        }
    }
}


// This function is responsible for performing stereo matching operations
// for the means of triangulation
// 输出Frame和左右图像的金字塔
void MapManager::stereoMatching(Frame &frame, const std::vector<cv::Mat> &vleftpyr, const std::vector<cv::Mat> &vrightpyr) 
{
    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::Start("1.KF_stereoMatching");

    // Find stereo correspondances with left kps，取出左图的特征点，与特征点的数量
    auto vleftkps = frame.getKeypoints();
    size_t nbkps = vleftkps.size();

    // ZNCC Parameters
    size_t nmaxpyrlvl = pslamstate_->nklt_pyr_lvl_*2;
    int winsize = 7;
    // ? = 2^3 = 8
    float uppyrcoef = std::pow(2,pslamstate_->nklt_pyr_lvl_);
    float downpyrcoef = 1. / uppyrcoef;
    
    std::vector<int> v3dkpids, vkpids, voutkpids, vpriorids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;

    // First we're gonna track 3d kps on only 2 levels
    v3dkpids.reserve(frame.nb3dkps_);    // 右图3D点的ID
    v3dkps.reserve(frame.nb3dkps_);      // 右图3D点对应的左图的像素坐标
    v3dpriors.reserve(frame.nb3dkps_);   // 右图3D点的投影坐标（初始化坐标）

    // Then we'll track 2d kps on full pyramid levels
    vkpids.reserve(nbkps);
    vkps.reserve(nbkps);
    vpriors.reserve(nbkps);

    // 遍历左图的特征点，如果是3D点直接投影到右图，如果不是，找到附近的地图点，有3D点再取附近3D点的平均深度投影到右图
    for( size_t i = 0 ; i < nbkps ; i++ )
    {
        // 取出左图的KeyPoint
        auto &kp = vleftkps.at(i);

        // 初始化右图的特征点坐标为左图的特征点坐标
        cv::Point2f priorpt = kp.px_;

        // If 3D, check if we can find a prior in right image
        if( kp.is3d_ ) {
            auto plm = getMapPoint(kp.lmid_);
            if( plm != nullptr ) {
                cv::Point2f projpt = frame.projWorldToRightImageDist(plm->getPoint());
                if( frame.isInRightImage(projpt) ) {
                    v3dkps.push_back(kp.px_);
                    v3dpriors.push_back(projpt);
                    v3dkpids.push_back(kp.lmid_);
                    continue;
                } 
            } else {
                removeMapPointObs(kp.lmid_, frame.kfid_);
                continue;
            }
        } 
        // 如果不是3D点才会继续往下处理
        // If stereo rect images, prior from SAD，文件设置没有立体矫正
        if( pslamstate_->bdo_stereo_rect_ ) {

            float xprior = -1.;
            float l1err;

            cv::Point2f pyrleftpt = kp.px_ * downpyrcoef;

            ptracker_->getLineMinSAD(vleftpyr.at(nmaxpyrlvl), vrightpyr.at(nmaxpyrlvl), pyrleftpt, winsize, xprior, l1err, true);

            xprior *= uppyrcoef;

            if( xprior >= 0 && xprior <= kp.px_.x ) {
                priorpt.x = xprior;
            }

        }
        else { // Generate prior from 3d neighbors
            const size_t nbmin3dcokps = 1;
            // 找到附近（左上）网格中的特征点
            auto vnearkps = frame.getSurroundingKeypoints(kp);
            if( vnearkps.size() >= nbmin3dcokps ) // 特征点大于0
            {
                std::vector<Keypoint> vnear3dkps;    // 存储附近的3D特征点
                vnear3dkps.reserve(vnearkps.size());
                for( const auto &cokp : vnearkps ) {
                    if( cokp.is3d_ ) {
                        vnear3dkps.push_back(cokp);
                    }
                }

                if( vnear3dkps.size() >= nbmin3dcokps ) { // 3D特征点大于0
                
                    size_t nb3dkp = 0;   // 对应的地图点的数量
                    double mean_z = 0.;
                    double weights = 0.;

                    for( const auto &cokp : vnear3dkps ) {  // 遍历有3D信息的特征点
                        auto plm = getMapPoint(cokp.lmid_); // 得到地图点
                        if( plm != nullptr ) {
                            nb3dkp++;
                            double coef = 1. / cv::norm(cokp.unpx_ - kp.unpx_); // 1/(左图特征点像素坐标-左图附近特征点像素坐标)
                            weights += coef;
                            mean_z += coef * frame.projWorldToCam(plm->getPoint()).z(); // 空间点的三维坐标转到相机坐标系下的z值
                        }
                    }

                    if( nb3dkp >= nbmin3dcokps ) {   // 对应的地图点大于0
                        mean_z /= weights;           // 空间点的三维坐标转到相机坐标系下的平均z值
                        Eigen::Vector3d predcampt = mean_z * ( kp.bv_ / kp.bv_.z() ); // 将左图的特征点的深度设为平均深度

                        cv::Point2f projpt = frame.projCamToRightImageDist(predcampt); // 使用平均深度的点投影到右图，计算像素坐标

                        if( frame.isInRightImage(projpt) ) 
                        {
                            v3dkps.push_back(kp.px_);
                            v3dpriors.push_back(projpt);
                            v3dkpids.push_back(kp.lmid_);
                            continue;
                        }
                    }
                }
            }
        }

        vkpids.push_back(kp.lmid_);
        vkps.push_back(kp.px_);
        vpriors.push_back(priorpt);
    }

    // Storing good tracks   
    std::vector<cv::Point2f> vgoodrkps; // 追踪上的特征点像素坐标，右图
    std::vector<int> vgoodids;          // 追踪上的特征点ID
    vgoodrkps.reserve(nbkps);
    vgoodids.reserve(nbkps);

    // 1st track 3d kps if using prior
    if( !v3dpriors.empty() ) 
    {
        size_t nbpyrlvl = 1;
        int nwinsize = pslamstate_->nklt_win_size_; // What about a smaller window here?

        if( vleftpyr.size() < 2*(nbpyrlvl+1) ) {
            nbpyrlvl = vleftpyr.size() / 2 - 1;
        }

        // Good / bad kps vector
        std::vector<bool> vkpstatus;
        // 光流法找特征点
        ptracker_->fbKltTracking(
                    vleftpyr, 
                    vrightpyr, 
                    nwinsize, 
                    nbpyrlvl, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    v3dkps, 
                    v3dpriors,   // 优化后的特征点
                    vkpstatus);

        size_t nbgood = 0;
        size_t nb3dkps = v3dkps.size();
        
        for(size_t i = 0 ; i < nb3dkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(v3dpriors.at(i));
                vgoodids.push_back(v3dkpids.at(i));
                nbgood++;
            } else {
                // If tracking failed, gonna try on full pyramid size
                // without prior for 2d kps
                vkpids.push_back(v3dkpids.at(i));
                vkps.push_back(v3dkps.at(i));
                vpriors.push_back(v3dpriors.at(i));
            }
        }

        if( pslamstate_->debug_ ) 
            std::cout << "\n >>> Stereo KLT Tracking on priors : " << nbgood 
                << " out of " << nb3dkps << " kps tracked!\n";
    }

    // 2nd track other kps if any
    if( !vkps.empty() ) 
    {
        // Good / bad kps vector
        std::vector<bool> vkpstatus;

        ptracker_->fbKltTracking(
                    vleftpyr, 
                    vrightpyr, 
                    pslamstate_->nklt_win_size_, 
                    pslamstate_->nklt_pyr_lvl_, 
                    pslamstate_->nklt_err_, 
                    pslamstate_->fmax_fbklt_dist_, 
                    vkps, 
                    vpriors, 
                    vkpstatus);

        size_t nbgood = 0;
        size_t nb2dkps = vkps.size();

        for(size_t i = 0 ; i < nb2dkps  ; i++ ) 
        {
            if( vkpstatus.at(i) ) {
                vgoodrkps.push_back(vpriors.at(i));
                vgoodids.push_back(vkpids.at(i));
                nbgood++;
            }
        }

        if( pslamstate_->debug_ )
            std::cout << "\n >>> Stereo KLT Tracking w. no priors : " << nbgood
                << " out of " << nb2dkps << " kps tracked!\n";
    }

    nbkps = vgoodids.size();
    size_t nbgood = 0;

    float epi_err = 0.;
    // 遍历特征点，将检测通过的特征点，插入到帧的信息
    for( size_t i = 0; i < nbkps ; i++ ) 
    {
        // 取出匹配上的左右图的去畸变的特征点
        cv::Point2f lunpx = frame.getKeypointById(vgoodids.at(i)).unpx_;
        cv::Point2f runpx = frame.pcalib_rightcam_->undistortImagePoint(vgoodrkps.at(i));

        // Check epipolar consistency (same row for rectified images)
        if( pslamstate_->bdo_stereo_rect_ ) {
            epi_err = fabs(lunpx.y - runpx.y);
            // Correct right kp to be on the same row
            vgoodrkps.at(i).y = lunpx.y;
        }
        else {
            // ?计算误差
            epi_err = MultiViewGeometry::computeSampsonDistance(frame.Frl_, lunpx, runpx);
        }
        
        if( epi_err <= 2. ) 
        {
            frame.updateKeypointStereo(vgoodids.at(i), vgoodrkps.at(i));
            nbgood++;
        }
    }

    if( pslamstate_->debug_ )
        std::cout << "\n \t>>> Nb of stereo tracks: " << nbgood
            << " out of " << nbkps << "\n";

    if( pslamstate_->debug_ || pslamstate_->log_timings_ )
        Profiler::StopAndDisplay(pslamstate_->debug_, "1.KF_stereoMatching");
}


Eigen::Vector3d MapManager::computeTriangulation(const Sophus::SE3d &T, const Eigen::Vector3d &bvl, const Eigen::Vector3d &bvr)
{
    // OpenGV Triangulate
    return MultiViewGeometry::triangulate(T, bvl, bvr);
}

// This function copies cur. Frame to add it to the KF map
void MapManager::addKeyframe()
{
    // Create a copy of Cur. Frame shared_ptr for creating an 
    // independant KF to add to the map
    std::shared_ptr<Frame> pkf = std::allocate_shared<Frame>(Eigen::aligned_allocator<Frame>(), *pcurframe_);

    std::lock_guard<std::mutex> lock(kf_mutex_);

    // Add KF to the unordered map and update id/nb
    map_pkfs_.emplace(nkfid_, pkf);
    nbkfs_++;
    nkfid_++;
}

// This function adds a new MP to the map
void MapManager::addMapPoint(const cv::Scalar &color)
{
    // Create a new MP with a unique lmid and a KF id obs
    std::shared_ptr<MapPoint> plm = std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), nlmid_, nkfid_, color);

    // Add new MP to the map and update id/nb
    map_plms_.emplace(nlmid_, plm);
    nlmid_++;
    nblms_++;

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if( plm->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plm->color_[0] 
                                    , plm->color_[0]
                                    , plm->color_[0]
                                    );
    }
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.push_back(colored_pt);
}


// This function adds a new MP to the map with desc
void MapManager::addMapPoint(const cv::Mat &desc, const cv::Scalar &color)
{
    // Create a new MP with a unique lmid and a KF id obs
    std::shared_ptr<MapPoint> plm = std::allocate_shared<MapPoint>(Eigen::aligned_allocator<MapPoint>(), nlmid_, nkfid_, desc, color);

    // Add new MP to the map and update id/nb
    map_plms_.emplace(nlmid_, plm);
    nlmid_++;
    nblms_++;

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if( plm->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plm->color_[0] 
                                    , plm->color_[0]
                                    , plm->color_[0]
                                    );
    }
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.push_back(colored_pt);
}

// Returns a shared_ptr of the req. KF
std::shared_ptr<Frame> MapManager::getKeyframe(const int kfid) const
{
    std::lock_guard<std::mutex> lock(kf_mutex_);

    auto it = map_pkfs_.find(kfid);
    if( it == map_pkfs_.end() ) {
        return nullptr;
    }
    return it->second;
}

// Returns a shared_ptr of the req. MP
std::shared_ptr<MapPoint> MapManager::getMapPoint(const int lmid) const
{
    std::lock_guard<std::mutex> lock(lm_mutex_);

    auto it = map_plms_.find(lmid);
    if( it == map_plms_.end() ) {
        return nullptr;
    }
    return it->second;
}

// Update a MP world pos. (vkps.at(kpidx).lmid_, wpt, 1./left_pt.z())
void MapManager::updateMapPoint(const int lmid, const Eigen::Vector3d &wpt, const double kfanch_invdepth)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);
    // 找到这个地图点
    auto plmit = map_plms_.find(lmid);

    if( plmit == map_plms_.end() ) {
        return;
    }

    if( plmit->second == nullptr ) {
        return;
    }

    // 目前还没有在地图点中被设置成3D点
    if( !plmit->second->is3d_ ) {
        // 遍历能看到这个地图点的所有关键帧
        for( const auto &kfid : plmit->second->getKfObsSet() ) {
            auto pkfit = map_pkfs_.find(kfid);
            // 这个关键帧存在
            if( pkfit != map_pkfs_.end() ) {
                // 将这个帧中的这个ID的特征点设为3D点
                pkfit->second->turnKeypoint3d(lmid);
            } else {
                // 这个帧不存在，从地图点中删除这个帧的信息
                plmit->second->removeKfObs(kfid);
            }
        }
        // 这个地图点能在当前帧被观测到
        if( plmit->second->isobs_ ) {
            pcurframe_->turnKeypoint3d(lmid);
        }
    }

    // Update MP world pos.
    if( kfanch_invdepth >= 0. ) {
        plmit->second->setPoint(wpt, kfanch_invdepth);
    } else {
        plmit->second->setPoint(wpt);
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    if(plmit->second->isobs_ ) {
        colored_pt = pcl::PointXYZRGB(255, 0, 0);
    } else {
        colored_pt = pcl::PointXYZRGB(plmit->second->color_[0] 
                                    , plmit->second->color_[0]
                                    , plmit->second->color_[0]
                                    );
    }
    colored_pt.x = wpt.x();
    colored_pt.y = wpt.y();
    colored_pt.z = wpt.z();
    pcloud_->points.at(lmid) = colored_pt;
}

// Add a new KF obs to provided MP (lmid)
void MapManager::addMapPointKfObs(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    auto pkfit = map_pkfs_.find(kfid);
    auto plmit = map_plms_.find(lmid);

    if( pkfit == map_pkfs_.end() ) {
        return;
    }

    if( plmit == map_plms_.end() ) {
        return;
    }

    plmit->second->addKfObs(kfid);

    for( const auto &cokfid : plmit->second->getKfObsSet() ) {
        if( cokfid != kfid ) {
            auto pcokfit =  map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                pcokfit->second->addCovisibleKf(kfid);
                pkfit->second->addCovisibleKf(cokfid);
            } else {
                plmit->second->removeKfObs(cokfid);
            }
        }
    }
}

// Merge two MapPoints，当前帧的特征点ID，地图点ID
void MapManager::mergeMapPoints(const int prevlmid, const int newlmid)
{
    // 1. Get Kf obs + descs from prev MP
    // 2. Remove prev MP
    // 3. Update new MP and related KF / cur Frame

    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get prev MP to merge into new MP

    auto pprevlmit = map_plms_.find(prevlmid);
    auto pnewlmit = map_plms_.find(newlmid);

    if( pprevlmit == map_plms_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as prevlm is null\n";
        return;
    } else if( pnewlmit == map_plms_.end() ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is null\n";
        return;
    } else if ( !pnewlmit->second->is3d_ ) {
        if( pslamstate_->debug_ )
            std::cout << "\nMergeMapPoints skipping as newlm is not 3d\n";
        return;
    }

    // 1. Get Kf obs + descs from prev MP
    std::set<int> setnewkfids = pnewlmit->second->getKfObsSet();
    std::set<int> setprevkfids = pprevlmit->second->getKfObsSet();
    std::unordered_map<int, cv::Mat> map_prev_kf_desc_ = pprevlmit->second->map_kf_desc_;

    // 3. Update new MP and related KF / cur Frame
    for( const auto &pkfid : setprevkfids ) 
    {
        // Get prev KF and update keypoint
        auto pkfit =  map_pkfs_.find(pkfid);
        if( pkfit != map_pkfs_.end() ) {
            if( pkfit->second->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
            {
                pnewlmit->second->addKfObs(pkfid);
                for( const auto &nkfid : setnewkfids ) {
                    auto pcokfit = map_pkfs_.find(nkfid);
                    if( pcokfit != map_pkfs_.end() ) {
                        pkfit->second->addCovisibleKf(nkfid);
                        pcokfit->second->addCovisibleKf(pkfid);
                    }
                }
            }
        }
    }

    for( const auto &kfid_desc : map_prev_kf_desc_ ) {
        pnewlmit->second->addDesc(kfid_desc.first, kfid_desc.second);
    }

    // Turn new MP observed by cur Frame if prev MP
    // was + update cur Frame's kp ref to new MP
    if( pcurframe_->isObservingKp(prevlmid) ) 
    {
        if( pcurframe_->updateKeypointId(prevlmid, newlmid, pnewlmit->second->is3d_) )
        {
            setMapPointObs(newlmid);
        }
    }

    if( pprevlmit->second->is3d_ ) {
        nblms_--; 
    }

    // Erase MP and update nb MPs
    map_plms_.erase( pprevlmit );
    
    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points[prevlmid] = colored_pt;
}

// Remove a KF from the map
void MapManager::removeKeyframe(const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get KF to remove
    auto pkfit = map_pkfs_.find(kfid);
    // Skip if KF does not exist
    if( pkfit == map_pkfs_.end() ) {
        return;
    }

    // Remove the KF obs from all observed MP
    for( const auto &kp : pkfit->second->getKeypoints() ) {
        // Get MP and remove KF obs
        auto plmit = map_plms_.find(kp.lmid_);
        if( plmit == map_plms_.end() ) {
            continue;
        }
        plmit->second->removeKfObs(kfid);
    }
    for( const auto &kfid_cov : pkfit->second->getCovisibleKfMap() ) {
        auto pcokfit = map_pkfs_.find(kfid_cov.first);
        if( pcokfit != map_pkfs_.end() ) {
            pcokfit->second->removeCovisibleKf(kfid);
        }
    }

    // Remove KF and update nb KFs
    map_pkfs_.erase( pkfit );
    nbkfs_--;

    if( pslamstate_->debug_ )
        std::cout << "\n \t >>> removeKeyframe() --> Removed KF #" << kfid;
}

// Remove a MP from the map
void MapManager::removeMapPoint(const int lmid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Get related MP
    auto plmit = map_plms_.find(lmid);
    // Skip if MP does not exist
    if( plmit != map_plms_.end() ) {
        // Remove all observations from KFs
        for( const auto &kfid : plmit->second->getKfObsSet() ) 
        {
            auto pkfit = map_pkfs_.find(kfid);
            if( pkfit == map_pkfs_.end() ) {
                continue;
            }
            pkfit->second->removeKeypointById(lmid);

            for( const auto &cokfid : plmit->second->getKfObsSet() ) {
                if( cokfid != kfid ) {
                    pkfit->second->decreaseCovisibleKf(cokfid);
                }
            }
        }

        // If obs in cur Frame, remove cur obs
        if( plmit->second->isobs_ ) {
            pcurframe_->removeKeypointById(lmid);
        }

        if( plmit->second->is3d_ ) {
            nblms_--; 
        }

        // Erase MP and update nb MPs
        map_plms_.erase( plmit );
    }

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;
    colored_pt = pcl::PointXYZRGB(0, 0, 0);
    colored_pt.x = 0.;
    colored_pt.y = 0.;
    colored_pt.z = 0.;
    pcloud_->points.at(lmid) = colored_pt;
}

// Remove a KF obs from a MP
void MapManager::removeMapPointObs(const int lmid, const int kfid)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    // Remove MP obs from KF
    auto pkfit = map_pkfs_.find(kfid);
    if( pkfit != map_pkfs_.end() ) {
        pkfit->second->removeKeypointById(lmid);
    }

    // Remove KF obs from MP
    auto plmit = map_plms_.find(lmid);

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        return;
    }
    plmit->second->removeKfObs(kfid);

    if( pkfit != map_pkfs_.end() ) {
        for( const auto &cokfid : plmit->second->getKfObsSet() ) {
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                pkfit->second->decreaseCovisibleKf(cokfid);
                pcokfit->second->decreaseCovisibleKf(kfid);
            }
        }
    }
}

void MapManager::removeMapPointObs(MapPoint &lm, Frame &frame)
{
    std::lock_guard<std::mutex> lock(lm_mutex_);
    std::lock_guard<std::mutex> lockkf(kf_mutex_);

    frame.removeKeypointById(lm.lmid_);
    lm.removeKfObs(frame.kfid_);

    for( const auto &cokfid : lm.getKfObsSet() ) {
        if( cokfid != frame.kfid_ ) {
            auto pcokfit = map_pkfs_.find(cokfid);
            if( pcokfit != map_pkfs_.end() ) {
                frame.decreaseCovisibleKf(cokfid);
                pcokfit->second->decreaseCovisibleKf(frame.kfid_);
            }
        }
    }
}

// Remove a MP obs from cur Frame
void MapManager::removeObsFromCurFrameById(const int lmid)
{
    // Remove cur obs
    pcurframe_->removeKeypointById(lmid);
    
    // Set MP as not obs
    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return;
    }

    plmit->second->isobs_ = false;

    // Update MP color，设置观测不到的点的颜色为浅灰色
    colored_pt = pcl::PointXYZRGB(plmit->second->color_[0] 
                                , plmit->second->color_[0]
                                , plmit->second->color_[0]
                                );
                                
    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;
}

bool MapManager::setMapPointObs(const int lmid) 
{
    if( lmid >= (int)pcloud_->points.size() ) {
        return false;
    }

    auto plmit = map_plms_.find(lmid);

    // Visualization related part for pointcloud obs
    pcl::PointXYZRGB colored_pt;

    // Skip if MP does not exist
    if( plmit == map_plms_.end() ) {
        // Set the MP at origin
        pcloud_->points.at(lmid) = colored_pt;
        return false;
    }

    plmit->second->isobs_ = true;

    // Update MP color
    colored_pt = pcl::PointXYZRGB(200, 0, 0);
    colored_pt.x = pcloud_->points.at(lmid).x;
    colored_pt.y = pcloud_->points.at(lmid).y;
    colored_pt.z = pcloud_->points.at(lmid).z;
    pcloud_->points.at(lmid) = colored_pt;

    return true;
}

// Reset MapManager
void MapManager::reset()
{
    nlmid_ = 0;
    nkfid_ = 0;
    nblms_ = 0;
    nbkfs_ = 0;

    map_pkfs_.clear();
    map_plms_.clear();

    pcloud_->points.clear();
}
