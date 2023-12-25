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

#include "feature_extractor.hpp"

#include <algorithm>
#include <iterator>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp> 

#ifdef OPENCV_CONTRIB
#include <opencv2/xfeatures2d.hpp>
#endif
#include <opencv2/video/tracking.hpp>

#ifdef OPENCV_LAMBDA_MISSING

namespace cv {

  class ParallelLoopBodyLambdaWrapper : public ParallelLoopBody
  {
    private:
      std::function<void(const Range&)> m_functor;
    public:
      ParallelLoopBodyLambdaWrapper(std::function<void(const Range&)> functor) :
        m_functor(functor)
        { }    
      virtual void operator() (const cv::Range& range) const
        {
          m_functor(range);
        }
  };
  
  inline void parallel_for_(const cv::Range& range, std::function<void(const cv::Range&)> functor, double nstripes=-1.)
    {
      parallel_for_(range, ParallelLoopBodyLambdaWrapper(functor), nstripes);
    }
}

#endif


cv::Ptr<cv::GFTTDetector> pgftt_;
cv::Ptr<cv::FastFeatureDetector> pfast_;
#ifdef OPENCV_CONTRIB
cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> pbrief_;
#else
cv::Ptr<cv::DescriptorExtractor> pbrief_;
#endif

bool compare_response(cv::KeyPoint first, cv::KeyPoint second) {
    return first.response > second.response;
}

FeatureExtractor::FeatureExtractor(size_t nmaxpts, size_t nmaxdist, double dmaxquality, int nfast_th)
    : nmaxpts_(nmaxpts), nmaxdist_(nmaxdist), dmaxquality_(dmaxquality), nfast_th_(nfast_th)
{
    nmindist_ = nmaxdist / 2.;
    dminquality_ = dmaxquality / 2.;

    std::cout << "\n*********************************\n";
    std::cout << "\nFeature Extractor is constructed!\n";
    std::cout << "\n>>> Maximum nb of kps : " << nmaxpts_;
    std::cout << "\n>>> Maximum kps dist : " << nmaxdist_;
    std::cout << "\n>>> Minimum kps dist : " << nmindist_;
    std::cout << "\n>>> Maximum kps qual : " << dmaxquality_;
    std::cout << "\n>>> Minimum kps qual : " << dminquality_;
    std::cout << "\n*********************************\n";
}


/**
 * \brief Detect GFTT features (Harris corners with Shi-Tomasi method) using OpenCV.
 *
 * \param[in] im  Image to process (cv::Mat).
 * \param[in] vcurkps Vector of px positions to filter out detections.
 * \param[in] roi  Initial ROI mask (255 ok / 0 reject) to filter out detections (cv::Mat<CV_8UC1>).
 * \return Vector of deteted kps px pos.
 */
std::vector<cv::Point2f> FeatureExtractor::detectGFTT(const cv::Mat &im, const std::vector<cv::Point2f> &vcurkps, const cv::Mat &roi, int nbmax) const
{
    // 1. Check how many kps we need to detect
    size_t nb2detect = nmaxpts_ - vcurkps.size();
    if( vcurkps.size() >= nmaxpts_ ) {
        return std::vector<cv::Point2f>();
    }

    if( nbmax != -1 ) {
        nb2detect = nbmax;
    }

    // 1.1 Init the mask
    cv::Mat mask;
    if( !roi.empty() ) {
        mask = roi.clone();
    }
    setMask(im, vcurkps, nmaxdist_, mask);

    // 1.2 Extract kps
    std::vector<cv::KeyPoint> vnewkps;
    std::vector<cv::Point2f> vnewpts;
    vnewkps.reserve(nb2detect);
    vnewpts.reserve(nb2detect);

    // std::cout << "\n*****************************\n";
    // std::cout << "\t\t GFTT \n";
    // std::cout << "\n> Nb max pts : " << nmaxpts_;
    // std::cout << "\n> Nb 2 detect : " << nb2detect;
    // std::cout << "\n> Quality : " << dminquality_;
    // std::cout << "\n> Dist : " << nmaxdist_;

    // Init. detector if not done yet
    if( pgftt_ == nullptr ) {
        pgftt_ = cv::GFTTDetector::create(nmaxpts_, dminquality_, nmaxdist_);
    }
    
    pgftt_->setQualityLevel(dminquality_);
    pgftt_->setMinDistance(nmaxdist_);
    pgftt_->setMaxFeatures(nb2detect);
    pgftt_->detect(im, vnewkps, mask);

    // cv::KeyPointsFilter::runByPixelsMask(vnewkps,mask);
    cv::KeyPoint::convert(vnewkps, vnewpts);
    vnewkps.clear();

    // Check number of detections
    size_t nbdetected = vnewpts.size();
    // std::cout << "\n \t>>> Found : " << nbdetected;

    // Compute Corners with Sub-Pixel Accuracy
    if( nbdetected > 0 )
    {
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vnewpts, winSize, zeroZone, criteria);
    }

    // If enough, return kps
    if( nbdetected >= 0.66 * nb2detect || nb2detect < 20 ) {
        return vnewpts;
    }

    // Else, detect more
    nb2detect -= nbdetected;
    std::vector<cv::Point2f> vmorepts;
    vmorepts.reserve(nb2detect);

    // Update mask to force detection around 
    // not observed areas
    mask.release();
    if( !roi.empty() ) {
        mask = roi.clone();
    }
    setMask(im, vcurkps, nmindist_, mask);
    setMask(im, vnewpts, nmindist_, mask);

    // Detect additional kps
    // std::cout << "\n \t>>>  Searching more : " << nb2detect;

    pgftt_->setQualityLevel(dmaxquality_);
    pgftt_->setMinDistance(nmindist_);
    pgftt_->setMaxFeatures(nb2detect);
    pgftt_->detect(im, vnewkps, mask);

    cv::KeyPoint::convert(vnewkps, vmorepts);
    vnewkps.clear();

    nbdetected = vmorepts.size();
    // std::cout << "\n \t>>>  Additionally found : " << nbdetected;

    // Compute Corners with Sub-Pixel Accuracy
    if( nbdetected > 0 )
    {
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vmorepts, winSize, zeroZone, criteria);
    }

    // Insert new detections
    vnewpts.insert(vnewpts.end(), 
                    std::make_move_iterator(vmorepts.begin()),
                    std::make_move_iterator(vmorepts.end())
                );

    // std::cout << "\n \t>>>  Total found : " << vnewpts.size();

    // Return vector of detected kps px pos.
    return vnewpts;
}


std::vector<cv::Mat> FeatureExtractor::describeBRIEF(const cv::Mat &im, const std::vector<cv::Point2f> &vpts) const
{
    if( vpts.empty() ) {
        // std::cout << "\nNo kps provided to function describeBRIEF() \n";
        return std::vector<cv::Mat>();
    }

    std::vector<cv::KeyPoint> vkps;
    size_t nbkps = vpts.size();
    vkps.reserve(nbkps);
    std::vector<cv::Mat> vdescs;
    vdescs.reserve(nbkps);

    cv::KeyPoint::convert(vpts, vkps);

    cv::Mat descs;

    if( pbrief_ == nullptr ) {
        #ifdef OPENCV_CONTRIB
        pbrief_ = cv::xfeatures2d::BriefDescriptorExtractor::create();
        #else
        pbrief_  = cv::ORB::create(500, 1., 0);
        std::cout << "\n\n=======================================================================\n";
        std::cout << " BRIEF CANNOT BE USED ACCORDING TO CMAKELISTS (Opencv Contrib not enabled) \n";
        std::cout << " ORB WILL BE USED INSTEAD!  (BUT NO ROTATION  OR SCALE INVARIANCE ENABLED) \n";
        std::cout << "\n\n=======================================================================\n\n";
        #endif
    }

    // std::cout << "\nCOmputing desc for #" << vkps.size() << " kps\n";

    pbrief_->compute(im, vkps, descs);

    // std::cout << "\nDesc computed for #" << vkps.size() << " kps\n";

    if( vkps.empty() ) {
        return std::vector<cv::Mat>(nbkps, cv::Mat());
    }

    size_t k = 0;
    for( size_t i = 0 ; i < nbkps ; i++ ) 
    {
        if( k < vkps.size() ) {
            if( vkps[k].pt == vpts[i] ) {
                // vdescs.push_back(descs.row(k).clone());
                vdescs.push_back(descs.row(k));
                k++;
            }
            else {
                vdescs.push_back(cv::Mat());
            }
        } else {
            vdescs.push_back(cv::Mat());
        }
    }

    assert( vdescs.size() == vpts.size() );

    // std::cout << "\n \t >>> describeBRIEF : " << vkps.size() << " kps described!\n";

    return vdescs;
}

/**
1.函数开始检查输入图像 im 是否为空，如果为空，则返回一个空的 std::vector<cv::Point2f>。
2.计算图像的行数 nrows 和列数 ncols。
3.根据输入的 ncellsize 计算每个细胞（cell）的半边长度 nhalfcell。
4.计算图像中细胞的行数 nhcells 和列数 nwcells，以及总的细胞数 nbcells。
5.创建一个 std::vector<cv::Point2f>，用于存储检测到的特征点，并预留空间。
6.创建一个布尔类型的二维向量 voccupcells，用于表示细胞是否被占用。
7.创建一个和输入图像相同大小的掩码 mask，并设置所有元素为1。
8.对于输入的当前关键点 vcurkps，将它们所在的细胞标记为已占用，并在掩码中对应位置画圆以避免在该区域检测新的特征点。
9.初始化变量 nboccup 为0，表示占用的细胞数。
10.使用并行计算（parallel_for_）遍历每个细胞，对于未占用的细胞，使用高斯滤波和角点检测计算特征响应图，并在细胞内选择最大特征响应的点作为特征点。
11.将检测到的特征点保存在 vvdetectedpx 中，并在掩码中更新相应位置。
12.最后，根据检测到的特征点和次要检测到的特征点（vvsecdetectionspx），更新输出的特征点向量 vdetectedpx。
13.最后，根据检测到的特征点的数量和占用的细胞数量，调整参数 dmaxquality_ 的值。
14.如果检测到的特征点数量低于细胞总数的1/3，则减小 dmaxquality_，如果特征点数量超过总数的90%，则增大 dmaxquality_。
15.最后，如果检测到的特征点不为空，使用 cv::cornerSubPix 对特征点进行亚像素精度的优化。
*/
// 原始图像，网格尺寸，已经有的特征点，提取的部分
std::vector<cv::Point2f> FeatureExtractor::detectSingleScale(const cv::Mat &im, const int ncellsize, 
        const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi) 
{    
    if( im.empty() ) {
        // std::cerr << "\n No image provided to detectSingleScale() !\n";
        return std::vector<cv::Point2f>();
    }

    size_t ncols = im.cols;
    size_t nrows = im.rows;

    size_t nhalfcell = ncellsize / 4;

    size_t nhcells = nrows / ncellsize;
    size_t nwcells = ncols / ncellsize;

    size_t nbcells = nhcells * nwcells;

    // 提取出的特征点坐标
    std::vector<cv::Point2f> vdetectedpx;
    vdetectedpx.reserve(nbcells);
    // 网格内是否有特征点
    std::vector<std::vector<bool>> voccupcells(
            nhcells+1, 
            std::vector<bool>(nwcells+1, false)
            );
    
    cv::Mat mask = cv::Mat::ones(im.rows, im.cols, CV_32F);

    for( const auto &px : vcurkps ) {
        voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
        cv::circle(mask, px, nhalfcell, cv::Scalar(0.), -1);
    }

    // std::cout << "\n Single Scale detection \n";
    // std::cout << "\n nhcells : " << nhcells << " / nwcells : " << nwcells;
    // std::cout << " / nbcells : " << nhcells * nwcells;
    // std::cout << "\n cellsize : " << ncellsize;

    size_t nboccup = 0;

    std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells); // 检测到的特征点

    std::vector<std::vector<cv::Point2f>> vvsecdetectionspx(nbcells);

    auto cvrange = cv::Range(0, nbcells);

    parallel_for_(cvrange, [&](const cv::Range& range) {
        for( int i = range.start ; i < range.end ; i++ ) {

        size_t r = floor(i / nwcells);
        size_t c = i % nwcells;

        if( voccupcells[r][c] ) {
                nboccup++;
                continue;
        }

        size_t x = c*ncellsize;
        size_t y = r*ncellsize;
        
        cv::Rect hroi(x,y,ncellsize,ncellsize);

        if( x+ncellsize < ncols-1 && y+ncellsize < nrows-1 ) {
            cv::Mat hmap;
            cv::Mat filtered_im;
            cv::GaussianBlur(im(hroi), filtered_im, cv::Size(3,3), 0.);
            // 是 OpenCV 中用于计算图像中每个像素的最小特征值的函数，通常用于角点检测。
            cv::cornerMinEigenVal(filtered_im, hmap, 3, 3);

            double dminval, dmaxval;
            cv::Point minpx, maxpx; // 特征点坐标

            cv::minMaxLoc(hmap.mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
            maxpx.x += x;
            maxpx.y += y;

            if( maxpx.x < roi.x || maxpx.y < roi.y 
                || maxpx.x >= roi.x+roi.width 
                || maxpx.y >= roi.y+roi.height )
            {
                continue;
            }

            if( dmaxval >= dmaxquality_ ) {
                vvdetectedpx.at(i).push_back(maxpx);
                cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
            }

            cv::minMaxLoc(hmap.mul(mask(hroi)), &dminval, &dmaxval, &minpx, &maxpx);
            maxpx.x += x;
            maxpx.y += y;

            if( maxpx.x < roi.x || maxpx.y < roi.y 
                || maxpx.x >= roi.x+roi.width 
                || maxpx.y >= roi.y+roi.height )
            {
                continue;
            }

            if( dmaxval >= dmaxquality_ ) {
                vvsecdetectionspx.at(i).push_back(maxpx);
                cv::circle(mask, maxpx, nhalfcell, cv::Scalar(0.), -1);
            }
        }
    }
    });

    for( const auto &vpx : vvdetectedpx ) {
        if( !vpx.empty() ) {
            vdetectedpx.insert(vdetectedpx.end(), vpx.begin(), vpx.end());
        }
    }

    size_t nbkps = vdetectedpx.size();

    if( nbkps+nboccup < nbcells ) {
        size_t nbsec = nbcells - (nbkps+nboccup);
        size_t k = 0;
        for( const auto &vseckp : vvsecdetectionspx ) {
            if( !vseckp.empty() ) {
                vdetectedpx.push_back(vseckp.back());
                k++;
                if( k == nbsec ) {
                    break;
                }
            }
        }
    }

    nbkps = vdetectedpx.size();

    if( nbkps < 0.33 * (nbcells - nboccup) ) {
        dmaxquality_ /= 2.;
    } 
    else if( nbkps > 0.9 * (nbcells - nboccup) ) {
        dmaxquality_ *= 1.5;
    }

    // Compute Corners with Sub-Pixel Accuracy
    if( !vdetectedpx.empty() )
    {
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vdetectedpx, winSize, zeroZone, criteria);
    }

    // std::cout << "\n \t>>> Found : " << nbkps;

    return vdetectedpx;
}


std::vector<cv::Point2f> FeatureExtractor::detectGridFAST(const cv::Mat &im, const int ncellsize, 
        const std::vector<cv::Point2f> &vcurkps, const cv::Rect &roi)
{    
    if( im.empty() ) {
        // std::cerr << "\n No image provided to detectGridFAST() !\n";
        return std::vector<cv::Point2f>();
    }

    size_t ncols = im.cols;
    size_t nrows = im.rows;

    size_t nhalfcell = ncellsize / 4;

    size_t nhcells = nrows / ncellsize;
    size_t nwcells = ncols / ncellsize;

    size_t nbcells = nhcells * nwcells;

    std::vector<cv::Point2f> vdetectedpx;
    vdetectedpx.reserve(nbcells);

    std::vector<std::vector<bool>> voccupcells(
            nhcells+1, 
            std::vector<bool>(nwcells+1, false)
            );

    cv::Mat mask = cv::Mat::ones(im.rows, im.cols, CV_32F);

    for( const auto &px : vcurkps ) {
        voccupcells[px.y / ncellsize][px.x / ncellsize] = true;
        cv::circle(mask, px, nhalfcell, cv::Scalar(0), -1);
    }

    size_t nboccup = 0;
    size_t nbempty = 0;

    // Create the FAST detector if not set yet
    if( pfast_ == nullptr ) {
        pfast_ = cv::FastFeatureDetector::create(nfast_th_);
    }

    // std::cout << "\ndetectGridFAST (cellsize: " << ncellsize << ") : \n";
    // std::cout << "\n FAST grid search over #" << nbcells;
    // std::cout << " cells (" << nwcells << ", " << nhcells << ")\n";
 
    std::vector<std::vector<cv::Point2f>> vvdetectedpx(nbcells);

    auto cvrange = cv::Range(0, nbcells);

    parallel_for_(cvrange, [&](const cv::Range& range) {
        for( int i = range.start ; i < range.end ; i++ ) {

        size_t r = floor(i / nwcells);
        size_t c = i % nwcells;

        if( voccupcells[r][c] ) {
                nboccup++;
                continue;
        }

        nbempty++;

        size_t x = c*ncellsize;
        size_t y = r*ncellsize;
        
        cv::Rect hroi(x,y,ncellsize,ncellsize);

        if( x+ncellsize < ncols-1 && y+ncellsize < nrows-1 ) {

            std::vector<cv::KeyPoint> vkps;
            pfast_->detect(im(hroi), vkps, mask(hroi));

            if( vkps.empty() ) {
                continue;
            } else {
                std::sort(vkps.begin(), vkps.end(), compare_response);
            }

            if( vkps.at(0).response >= 20 ) {
                cv::Point2f pxpt = vkps.at(0).pt;
                
                pxpt.x += x;
                pxpt.y += y;

                cv::circle(mask, pxpt, nhalfcell, cv::Scalar(0), -1);

                vvdetectedpx.at(i).push_back(pxpt);
            }

        }
    }
    });

    for( const auto &vpx : vvdetectedpx ) {
        if( !vpx.empty() ) {
            vdetectedpx.insert(vdetectedpx.end(), vpx.begin(), vpx.end());
        }
    }

    size_t nbkps = vdetectedpx.size();

    // Update FAST th.
    // int nfast_th = pfast_->getThreshold();
    if( nbkps < 0.5 * nbempty && nbempty > 10 ) {
        nfast_th_ *= 0.66;
        pfast_->setThreshold(nfast_th_);
    } else if ( nbkps == nbempty ) {
        nfast_th_ *= 1.5;
        pfast_->setThreshold(nfast_th_);
    }


    // Compute Corners with Sub-Pixel Accuracy
    if( !vdetectedpx.empty() )
    {
        /// Set the need parameters to find the refined corners
        cv::Size winSize = cv::Size(3,3);
        cv::Size zeroZone = cv::Size(-1,-1);
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + 
                                        cv::TermCriteria::MAX_ITER, 30, 0.01);

        cv::cornerSubPix(im, vdetectedpx, winSize, zeroZone, criteria);
    }

    // std::cout << "\n \t>>> Found : " << vdetectedpx.size();

    return vdetectedpx;
}




void FeatureExtractor::setMask(const cv::Mat &im, const std::vector<cv::Point2f> &vpts, const int dist, cv::Mat &mask) const
{
    if( mask.empty() ) {
        mask = cv::Mat(im.rows, im.cols, CV_8UC1, cv::Scalar(255));
    }

    for (auto &pt : vpts) {
        cv::circle(mask, pt, dist, 0, -1);
    }
}