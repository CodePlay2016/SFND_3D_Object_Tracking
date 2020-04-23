
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        // if (enclosingBoxes.size() == 1)
        if (enclosingBoxes.size() >= 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1) {
        // create randomized color for current 3D object
        
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);

        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
    }
    

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (auto match : kptMatches) {
        cv::Point2f pt_pre = kptsPrev[match.queryIdx].pt;
        cv::Point2f pt_cur = kptsCurr[match.trainIdx].pt;
        if (boundingBox.roi.contains(pt_cur) && boundingBox.roi.contains(pt_pre))
            boundingBox.kptMatches.push_back(match);
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    int line_count = 0;
    double dist_threshold = 100;
    vector<double> ratios;
    bool cal_median = true;
    cv::Point2f pt_pre1, pt_pre2, pt_cur1, pt_cur2;
    for (int i=0; i<kptMatches.size(); i++) {
        auto match1 = kptMatches[i];
        pt_pre1 = kptsPrev[match1.queryIdx].pt;
        pt_cur1 = kptsCurr[match1.trainIdx].pt;
        double d_max = 0; int max_index = -1;
        for (int j=i+1; j<kptMatches.size(); j++) {
            auto match2 = kptMatches[j];
            pt_cur2 = kptsCurr[match2.trainIdx].pt;
            pt_pre2 = kptsPrev[match2.queryIdx].pt;
            double d_cur = cv::norm(pt_cur2 - pt_cur1);
            double d_pre = cv::norm(pt_pre2 - pt_pre1);
            if (d_pre > std::numeric_limits<double>::epsilon() && d_cur >= dist_threshold)
            { // avoid division by zero

                double distRatio = d_cur / d_pre;
                ratios.push_back(distRatio);
            }
        }
    }

    if (ratios.size() == 0) {
        TTC = NAN;
        return;
    }

    double dt = 1 / frameRate;

    if (cal_median) {
        std::sort(ratios.begin(), ratios.end());
        int medianIndex = ratios.size() / 2;
        double medianDistRatio = ratios[medianIndex];
        TTC = dt / (medianDistRatio - 1);
    } else {
        double meanDistRatio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
        TTC = 1 / (meanDistRatio - 1) * dt;
    }
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1/frameRate; // time between two measurements in seconds
    double outlier_threshold = 0.1; // 0.1

    double minXPrev = 1e9, minXCurr = 1e9;
    double mean=0;
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it)
        mean += it->x;
    mean /= lidarPointsPrev.size();
    for(auto it=lidarPointsPrev.begin(); it!=lidarPointsPrev.end(); ++it) {
        if (abs(it->x - mean) > outlier_threshold)
            continue;
        minXPrev = minXPrev>it->x ? it->x : minXPrev;
    }

    mean=0;
    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it)
        mean += it->x;
    mean /= lidarPointsCurr.size();
    for(auto it=lidarPointsCurr.begin(); it!=lidarPointsCurr.end(); ++it) {
        if (abs(it->x - mean) > outlier_threshold)
            continue;
        minXCurr = minXCurr>it->x ? it->x : minXCurr;
    }
    TTC = minXCurr * dT / (minXPrev-minXCurr);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
    unordered_set<int> matched_boxes;
    for (int i=0; i < prevFrame.boundingBoxes.size(); i++) {
        auto box_pre = prevFrame.boundingBoxes[i];
        int max_num_matches = 0; int match_index = -1;
        for (int j=0; j < currFrame.boundingBoxes.size(); j++) {
            if (matched_boxes.find(j) != matched_boxes.end())
                continue;
            auto box_cur = currFrame.boundingBoxes[j];
            int num_matches = 0;
            for (auto match : matches) {
                cv::Point2f pt_pre = prevFrame.keypoints[match.queryIdx].pt;
                cv::Point2f pt_cur = currFrame.keypoints[match.trainIdx].pt;
                if (box_pre.roi.contains(pt_pre) && box_cur.roi.contains(pt_cur))
                    num_matches += 1;
            }
            if (num_matches >= max_num_matches) {
                match_index = j;
                max_num_matches = num_matches;
            }
        }
        matched_boxes.insert(match_index);
        bbBestMatches[i] = match_index;
    }
}
