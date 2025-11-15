# stage_a.py
# Stage A: Motion Estimation with Visualization

import numpy as np
import cv2

class StageAMotionEstimation:
    def __init__(self, debug=True):
        self.debug = debug
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=1000,
                                   qualityLevel=0.01,
                                   minDistance=5,
                                   blockSize=7)
        self.frame_count = 0
        self.snapshot_saved = False
        if self.debug:
            cv2.namedWindow("Stage A: Motion Tracking", cv2.WINDOW_NORMAL)

    def process_frame_pair(self, prev_frame, curr_frame):
        self.frame_count += 1

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # detect features in previous frame
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        if p0 is None or len(p0) < 5:
            return np.eye(3), curr_frame, None

        # compute optical flow to current frame
        p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **self.lk_params)
        if p1 is None:
            return np.eye(3), curr_frame, None

        good_old = p0[st.ravel() == 1]
        good_new = p1[st.ravel() == 1]
        if len(good_new) < 4:
            return np.eye(3), curr_frame, None

        # compute homography (global motion) between old and new
        H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 2.0)
        if H is None:
            H = np.eye(3)

        # warp prev_frame so that background motion is compensated
        h, w = prev_gray.shape
        warped_prev = cv2.warpPerspective(prev_frame, H, (w, h))
        # also warp feature points to align background
        good_old_warped = cv2.perspectiveTransform(good_old.reshape(-1, 1, 2), H).reshape(-1, 2)

        if self.debug:
            vis = curr_frame.copy()

            motion_vec = good_new.reshape(-1, 2) - good_old_warped
            motion_mag = np.linalg.norm(motion_vec, axis=1)

            # compute threshold using median + MAD
            med = np.median(motion_mag)
            mad = np.median(np.abs(motion_mag - med)) + 1e-8
            thresh = med + 1.5 * mad

            moving_mask = motion_mag > thresh
            total_pts = len(motion_mag)
            moving_pts = int(np.sum(moving_mask))
            static_pts = total_pts - moving_pts

            for idx in range(total_pts):
                x, y = good_new[idx].ravel()
                color = (0, 0, 255) if motion_mag[idx] > thresh else (0, 255, 0)
                cv2.circle(vis, (int(x), int(y)), 2, color, -1)
                # draw arrow from old->new
                old_x, old_y = good_old_warped[idx]
                cv2.arrowedLine(vis, (int(old_x), int(old_y)), (int(x), int(y)), color, 1, tipLength=0.3)

            cv2.putText(vis, f"Pts: {total_pts}  Moving: {moving_pts}  Static: {static_pts}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Stage A: Motion Tracking", vis)
            cv2.waitKey(1)

            if self.frame_count == 150 and not self.snapshot_saved:
                cv2.imwrite("stageA_motion_tracking.png", vis)
                self.snapshot_saved = True

        return H, warped_prev, (good_old, good_new)




