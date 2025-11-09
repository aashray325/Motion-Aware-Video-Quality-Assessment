# stage_a.py
# Stage A: Motion Estimation and Global Motion Compensation

import numpy as np
import cv2


class StageAMotionEstimation:
    
    def __init__(self, feature_params=None, lk_params=None):
        #Initialize Stage A with parameters
        self.feature_params = feature_params or {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 7,
            'blockSize': 7
        }
        
        self.lk_params = lk_params or {
            'winSize': (21, 21),
            'maxLevel': 3,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }
    
    def detect_klt_features(self, frame_gray):
        #Detect KLT feature points
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        return p0
    
    def estimate_feature_correspondences(self, prev_gray, curr_gray, p0):
        #Estimate feature correspondences using Lucas-Kanade
        if p0 is None or len(p0) == 0:
            return None, None, None
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **self.lk_params
        )
        
        return p1, st, err
    
    def estimate_homography_ransac(self, p0, p1, st, ransac_threshold=5.0):
        #Estimate homography matrix using RANSAC
        if p0 is None or p1 is None or st is None:
            return np.eye(3), None
        
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        
        if len(good_new) < 4 or len(good_old) < 4:
            return np.eye(3), None
        
        H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransac_threshold)
        
        if H is None:
            return np.eye(3), None
        
        return H, mask
    
    def apply_homography_warp(self, prev_frame, H):
        #Apply homography transformation to warp previous frame
        if H is None:
            return prev_frame
        
        h, w = prev_frame.shape[:2]
        warped_frame = cv2.warpPerspective(prev_frame, H, (w, h))
        
        return warped_frame
    
    def process_frame_pair(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        p0 = self.detect_klt_features(prev_gray)
        
        if p0 is None:
            return np.eye(3), prev_frame, 0
        
        p1, st, err = self.estimate_feature_correspondences(prev_gray, curr_gray, p0)
        
        if p1 is None or st is None:
            return np.eye(3), prev_frame, 0
        
        H, mask = self.estimate_homography_ransac(p0, p1, st)
        
        num_inliers = 0
        if mask is not None:
            num_inliers = np.sum(mask)
        
        warped_prev = self.apply_homography_warp(prev_frame, H)
        
        return H, warped_prev, num_inliers
