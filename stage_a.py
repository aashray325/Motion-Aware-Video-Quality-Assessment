# import numpy as np
# import cv2


# class StageAMotionEstimation:
    
#     def __init__(self, feature_params=None, lk_params=None, debug=True):
#         """
#         Initialize Stage A with parameters
        
#         Params tweaked for low-texture (rice paddy) video:
#         - maxCorners: Increased 100 -> 1000
#         - qualityLevel: Decreased 0.3 -> 0.01
#         - winSize: Increased (21, 21) -> (31, 31)
#         """
        
#         # We need MORE features, even if they are lower quality.
#         # RANSAC is good at filtering out the bad ones.
#         self.feature_params = feature_params or {
#             'maxCorners': 1000,       # <-- INCREASED
#             'qualityLevel': 0.01,     # <-- DECREASED
#             'minDistance': 4,         # <-- DECREASED
#             'blockSize': 7
#         }
        
#         # A larger window can help track low-texture areas
#         self.lk_params = lk_params or {
#             'winSize': (31, 31),      # <-- INCREASED
#             'maxLevel': 3,
#             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#         }
        
#         # --- DEBUG ---
#         self.debug = debug
#         # if self.debug:
#         #     print("--- STAGE A DEBUG MODE ON ---")
#         #     # Create persistent windows
#         #     cv2.namedWindow("Warped Frame (Stage A)")
#         #     cv2.namedWindow("Feature Tracking (Stage A)")
#         # # --- END DEBUG ---
    
#     def detect_klt_features(self, frame_gray):
#         """Detect KLT feature points"""
#         p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
#         return p0
    
#     def estimate_feature_correspondences(self, prev_gray, curr_gray, p0):
#         """Estimate feature correspondences using Lucas-Kanade"""
#         if p0 is None or len(p0) == 0:
#             return None, None, None
        
#         p1, st, err = cv2.calcOpticalFlowPyrLK(
#             prev_gray, curr_gray, p0, None, **self.lk_params
#         )
        
#         return p1, st, err
    
#     def estimate_homography_ransac(self, p0, p1, st, ransac_threshold=5.0):
#         """Estimate homography matrix using RANSAC"""
#         if p0 is None or p1 is None or st is None:
#             return np.eye(3), None
        
#         good_new = p1[st == 1]
#         good_old = p0[st == 1]
        
#         if len(good_new) < 4 or len(good_old) < 4:
#             return np.eye(3), None
        
#         H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransac_threshold)
        
#         if H is None:
#             return np.eye(3), None
        
#         return H, mask
    
#     def apply_homography_warp(self, prev_frame, H):
#         """Apply homography transformation to warp previous frame"""
#         if H is None:
#             return prev_frame
        
#         h, w = prev_frame.shape[:2]
#         warped_frame = cv2.warpPerspective(prev_frame, H, (w, h))
        
#         return warped_frame
    
#     def process_frame_pair(self, prev_frame, curr_frame):
#         """STAGE A COMPLETE: Process a pair of consecutive frames"""
#         prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
#         curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
#         p0 = self.detect_klt_features(prev_gray)
        
#         if p0 is None:
#             return np.eye(3), prev_frame, 0
        
#         p1, st, err = self.estimate_feature_correspondences(prev_gray, curr_gray, p0)
        
#         if p1 is None or st is None:
#             return np.eye(3), prev_frame, 0
        
#         H, mask = self.estimate_homography_ransac(p0, p1, st, ransac_threshold=1.0)
        
#         num_inliers = 0
#         if mask is not None:
#             num_inliers = np.sum(mask)
        
#         warped_prev = self.apply_homography_warp(prev_frame, H)
        
#         # --- DEBUG VISUALIZATION ---
#         if self.debug:
#             # 1. Create the feature tracking visualization
#             debug_frame = curr_frame.copy()
#             good_new = p1[st == 1]
#             good_old = p0[st == 1]
            
#             # Draw the tracks
#             for i, (new, old) in enumerate(zip(good_new, good_old)):
#                 a, b = new.ravel()
#                 c, d = old.ravel()
#                 # If RANSAC marked this as an "inlier" (good point), draw green
#                 if mask is not None and mask[i] == 1:
#                     debug_frame = cv2.line(debug_frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 1)
#                     debug_frame = cv2.circle(debug_frame, (int(a), int(b)), 3, (0, 255, 0), -1)
#                 # If RANSAC marked as "outlier" (e.g., the man), draw red
#                 else:
#                     debug_frame = cv2.line(debug_frame, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 1)
#                     debug_frame = cv2.circle(debug_frame, (int(a), int(b)), 3, (0, 0, 255), -1)
            
#             # 2. Show the two debug windows
#             # cv2.imshow("Feature Tracking (Stage A)", debug_frame)
#             # cv2.imshow("Warped Frame (Stage A)", warped_prev)
            
#             # 3. This is ESSENTIAL for windows to update
#             cv2.waitKey(1) 
#         # --- END DEBUG ---
            
#         return H, warped_prev, num_inliers

# stage_a.py
# Stage A: Motion Estimation and Global Motion Compensation

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

