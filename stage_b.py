# # stage_b.py
# # Stage B: Motion Saliency Awareness (MSA) Generation
# # IMPROVED VERSION — connected component filtering, larger kernel,
# # no per-frame normalization (preserves true motion magnitude)

# import numpy as np
# import cv2
# from scipy.ndimage import gaussian_filter


# class StageBMSAGeneration:
#     """
#     Stage B: Motion Saliency Awareness (MSA) Generation
#     Creates a spatial weight map (MSA map) from the residual difference map.
#     """

#     def __init__(self, debug_video_name="debug", debug=True):
#         self.debug = debug
#         self.frame_count = 0
#         self.debug_video_name = debug_video_name  # e.g., "distorted_static_real"
#         self.snapshot_saved = False

#         if self.debug:
#             print(f"--- STAGE B DEBUG ON (Saving for: {self.debug_video_name}) ---")
#             cv2.namedWindow("Stage B: Final MSA Map (Cleaned)")

#     # ------------------------------------------------------------
#     def calculate_difference_map(self, curr_frame, warped_prev):
#         """Compute absolute grayscale difference between warped previous and current frame."""
#         if len(curr_frame.shape) == 3:
#             curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(float)
#             prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY).astype(float)
#         else:
#             curr_gray = curr_frame.astype(float)
#             prev_gray = warped_prev.astype(float)

#         diff_map = np.abs(curr_gray - prev_gray)
#         diff_map_norm = diff_map / 255.0
#         return diff_map_norm

#     # ------------------------------------------------------------
#     def process_frame_pair(self, prev_frame, curr_frame, warped_prev, smoothing_sigma=0.5):
#         """
#         STAGE B COMPLETE: Generate motion-saliency map for a frame pair.
#         Uses connected-component filtering to keep real motion and removes noise.
#         """
#         self.frame_count += 1

#         # 1. Residual motion magnitude
#         diff_map_norm = self.calculate_difference_map(curr_frame, warped_prev)

#         # 2. Convert to 8-bit map
#         diff_map_uint8 = (diff_map_norm * 255).astype(np.uint8)

#         # 3. Aggressive threshold to ignore camera noise
#         NOISE_GATE_THRESHOLD = 20
#         _, msa_map_binary = cv2.threshold(
#             diff_map_uint8, NOISE_GATE_THRESHOLD, 255, cv2.THRESH_BINARY
#         )

#         # 4. Morphological opening to remove tiny specks
#         kernel_open = np.ones((3, 3), np.uint8)
#         msa_map_cleaned = cv2.morphologyEx(
#             msa_map_binary, cv2.MORPH_OPEN, kernel_open, iterations=1
#         )

#         # 5. Keep only sufficiently large connected regions (e.g., cars)
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#             msa_map_cleaned, connectivity=8
#         )
#         keep = np.zeros_like(msa_map_cleaned, dtype=np.uint8)
#         MIN_AREA_PX = 200  # ignore small blobs
#         for lab in range(1, num_labels):
#             area = stats[lab, cv2.CC_STAT_AREA]
#             if area >= MIN_AREA_PX:
#                 keep[labels == lab] = 255

#         # 6. Dilate to make sure object regions fully covered
#         kernel_dilate = np.ones((5, 5), np.uint8)
#         keep = cv2.dilate(keep, kernel_dilate, iterations=1)

#         # 7. Convert to float map (0–1) and smooth slightly
#         msa_map = (keep / 255.0).astype(np.float32)
#         msa_map = gaussian_filter(msa_map, sigma=smoothing_sigma)

#         # NOTE: No per-frame min/max normalization — preserves absolute motion strength

#         # --- DEBUG VISUALIZATION & SNAPSHOT ---
#         if self.debug:
#             cv2.imshow("Stage B: Final MSA Map (Cleaned)", msa_map)
#             cv2.waitKey(1)

#             if self.frame_count == 150 and not self.snapshot_saved:
#                 filename = f"{self.debug_video_name}_B_MSA_MAP.png"
#                 print(f"  ... Saving Stage B snapshot: {filename}")
#                 cv2.imwrite(filename, (msa_map * 255).astype(np.uint8))
#                 self.snapshot_saved = True

#         return msa_map

# stage_b.py
# Stage B: Motion Saliency Awareness (MSA) Generation

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
            print(f"--- STAGE B DEBUG ON (Saving for: {self.debug_video_name}) ---")
            cv2.namedWindow("Stage B: Final MSA Map (Cleaned)")

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

        NOISE_GATE_THRESHOLD = 10  # lower for sensitivity
        _, msa_binary = cv2.threshold(diff_map_uint8, NOISE_GATE_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        msa_clean = cv2.morphologyEx(msa_binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

        # Keep only large components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(msa_clean, connectivity=8)
        keep = np.zeros_like(msa_clean)
        MIN_AREA_PX = 200
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
                print(f"  ... Saving Stage B snapshot: {fname}")
                cv2.imwrite(fname, (msa_map * 255).astype(np.uint8))
                self.snapshot_saved = True

        return msa_map

