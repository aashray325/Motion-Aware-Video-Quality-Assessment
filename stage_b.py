# stage_b.py
# Stage B: Motion Saliency Awareness Map

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class StageBMSAGeneration:
    def __init__(self, debug_video_name="debug", debug=True):
        self.debug = debug
        self.frame_count = 0
        self.debug_video_name = debug_video_name
        self.snapshot_saved = False
        if self.debug:
            print(f"Stage B: {self.debug_video_name})")
            cv2.namedWindow("Stage B: Final MSA Map")

    def calculate_difference_map(self, curr_frame, warped_prev):
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(float)
            prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            curr_gray, prev_gray = curr_frame.astype(float), warped_prev.astype(float)
        diff_map = np.abs(curr_gray - prev_gray)
        return diff_map / 255.0

    def process_frame_pair(self, prev_frame, curr_frame, warped_prev, smoothing_sigma=0.5):
        self.frame_count += 1
        diff_map_uint8 = (self.calculate_difference_map(curr_frame, warped_prev) * 255).astype(np.uint8)

        NOISE_GATE_THRESHOLD = 10  
        _, msa_binary = cv2.threshold(diff_map_uint8, NOISE_GATE_THRESHOLD, 255, cv2.THRESH_BINARY)

        msa_clean = cv2.morphologyEx(msa_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(msa_clean, connectivity=8)
        keep = np.zeros_like(msa_clean)
        MIN_AREA_PX = 150
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] >= MIN_AREA_PX:
                keep[labels == lab] = 255

        keep = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=1)
        msa_map = gaussian_filter(keep.astype(np.float32) / 255.0, sigma=smoothing_sigma)

        if self.debug:
            cv2.imshow("Stage B: Final MSA Map (Cleaned)", msa_map)
            cv2.waitKey(1)
            if self.frame_count == 150 and not self.snapshot_saved:
                fname = f"{self.debug_video_name}_B_MSA_MAP.png"
                cv2.imwrite(fname, (msa_map * 255).astype(np.uint8))
                self.snapshot_saved = True

        return msa_map

