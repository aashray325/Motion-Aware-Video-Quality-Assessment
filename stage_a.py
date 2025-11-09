# stage_a.py
# Stage A: Motion Estimation

import numpy as np
import cv2


class StageAMotionEstimation:
    def __init__(self, feature_params=None, lk_params=None, debug=False):
        self.feature_params = feature_params or {
            'maxCorners': 1000,
            'qualityLevel': 0.01,
            'minDistance': 4,
            'blockSize': 7
        }

        self.lk_params = lk_params or {
            'winSize': (31, 31),
            'maxLevel': 3,
            'criteria': (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10, 0.03
            ),
        }

        self.debug = debug

    def detect_klt_features(self, frame_gray):
        return cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)

    def estimate_feature_correspondences(self, prev_gray, curr_gray, p0):
        if p0 is None or len(p0) == 0:
            return None, None, None
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, p0, None, **self.lk_params
        )
        return p1, st, err

    def estimate_homography_ransac(self, p0, p1, st, ransac_threshold=3.0):
        if p0 is None or p1 is None or st is None:
            return np.eye(3), None
        good_new, good_old = p1[st == 1], p0[st == 1]
        if len(good_new) < 4 or len(good_old) < 4:
            return np.eye(3), None
        H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransac_threshold)
        return (H if H is not None else np.eye(3)), mask

    def apply_homography_warp(self, prev_frame, H):
        if H is None:
            return prev_frame
        h, w = prev_frame.shape[:2]
        return cv2.warpPerspective(prev_frame, H, (w, h))

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
        warped_prev = self.apply_homography_warp(prev_frame, H)
        num_inliers = int(np.sum(mask)) if mask is not None else 0
        return H, warped_prev, num_inliers

