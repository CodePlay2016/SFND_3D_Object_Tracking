# SFND 3D Object Tracking

## FP.1 Match 3D Objects
> **Criteira:** Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences.

```c++
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
```

## FP.2 Compute Lidar-based TTC

>**Criteira: **Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

* By calculating the mean value in x direction for all points both for previous frame and current frame, the LiDAR points that exceed more than threshold to the mean value will be considered as outliers.

```c++
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    double dT = 1/frameRate; // time between two measurements in seconds
    double outlier_threshold = 0.1;

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
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes

> **Criteira:** Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

```c++
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (auto match : kptMatches) {
        cv::Point2f pt_pre = kptsPrev[match.queryIdx].pt;
        cv::Point2f pt_cur = kptsCurr[match.trainIdx].pt;
        if (boundingBox.roi.contains(pt_cur) && boundingBox.roi.contains(pt_pre))
            boundingBox.kptMatches.push_back(match);
    }
}
```

## FP.4 Compute Camera-based TTC

> **Criteira:** Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

```c++
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
        std::sort(distRatios.begin(), distRatios.end());
        int medianIndex = distRatios.size() / 2;
        double medianDistRatio = distRatios[medianIndex];
        TTC = dt / (medianDistRatio - 1);
    } else {
        double meanDistRatio = std::accumulate(ratios.begin(), ratios.end(), 0.0) / ratios.size();
        TTC = 1 / (meanDistRatio - 1) * dt;
    }
}
```

## FP.5 Performance Evaluation 1

> **Criteira:** Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

#### Observation

![image-20200423212951898](/Users/hufangquan/study/SensorFusion/Camera/Projects/SFND_3D_Object_Tracking/README.assets/image-20200423212951898.png)

Sometimes the TTC estimated by LiDAR could be small. Situations like this are likely to happen due to LiDAR point outliers that are too close to the sensor in this frame, which leads to the over-estimation to the relative speed.

![image-20200423213740967](/Users/hufangquan/study/SensorFusion/Camera/Projects/SFND_3D_Object_Tracking/README.assets/image-20200423213740967.png)

Sometimes the TTC estimated by LiDAR could be too large. Situations like this are likely to happen due to LiDAR point outliers that are too close to the sensor in the previous frame, or there are outliers that are too far away in this frame,  which leads to the under-estimation to the relative speed.

#### Solution

By calculating the mean value in x direction for all points both for previous frame and current frame, the LiDAR points that exceed more than threshold to the mean value will be considered as outliers.

## FP.6 Performance Evaluation 2

> **Criteira:** Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

I evaluated the top 3 detector/decriptor combination I recommended in the Mid-term Project. The results are shown below. 

#### FAST + BRIEF

<img src="/Users/hufangquan/study/SensorFusion/Camera/Projects/SFND_3D_Object_Tracking/README.assets/image-20200424002731216.png" alt="image-20200424002731216" style="zoom:33%;" />

The average run time for FAST+BRIEF combination is *41ms*. As shown above, there are some frames with wierd outliers due to some mismatches generated by the combination.

#### FAST + ORB

<img src="/Users/hufangquan/study/SensorFusion/Camera/Projects/SFND_3D_Object_Tracking/README.assets/image-20200424004623998.png" alt="image-20200424004623998" style="zoom:33%;" />

The average run time for FAST+ORB combination is *38ms*. Similar to FAST+ORB, there are also some frames with wierd outliers due to some mismatches generated by the combination.

#### AKAZE + AKAZE

<img src="/Users/hufangquan/study/SensorFusion/Camera/Projects/SFND_3D_Object_Tracking/README.assets/image-20200424005938175.png" alt="image-20200424005938175" style="zoom:33%;" />

The average run time for AKAZE combination is *148ms*. The estimated TTCs are more close to those estimated from LiDAR points.