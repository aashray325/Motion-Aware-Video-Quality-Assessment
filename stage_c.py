# # stage_c.py
# # Stage C: Weighted Spatial Pooling
# # MODIFIED with debug visualization

# import numpy as np
# import cv2
# from skimage.metrics import structural_similarity as ssim


# class StageCWeightedSpatialPooling:
#     """
#     Stage C: Weighted Spatial Pooling
#     Applies MSA weights to SSIM to get frame-level quality scores
#     """
    
#     def __init__(self, window_size=11, debug=True):
#         """Initialize Stage C"""
#         self.window_size = window_size
        
#         # --- DEBUG ---
#         self.debug = debug
#         if self.debug:
#             print("--- STAGE C DEBUG MODE ON ---")
#             cv2.namedWindow("Stage C: SSIM Map (Raw)")
#             cv2.namedWindow("Stage C: Error Map (Cleaned)")
#             cv2.namedWindow("Stage C: Saliency Overlap (Penalty)")
#         # --- END DEBUG ---

    
#     def compute_ssim_map(self, ref_frame, dist_frame):
#         """Compute pixel-wise SSIM between reference and distorted frames"""
#         if len(ref_frame.shape) == 3:
#             ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
#             dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
#         else:
#             ref_gray = ref_frame
#             dist_gray = dist_frame
        
#         # Compute SSIM map (full=True returns the map instead of just the score)
#         try:
#             # The 'data_range' is important. Grayscale is 0-255.
#             _, ssim_map = ssim(ref_gray, dist_gray, win_size=self.window_size, full=True, data_range=255)
#         except:
#             # Fallback for small frames/windows
#             _, ssim_map = ssim(ref_gray, dist_gray, win_size=5, full=True, data_range=255)
        
#         return ssim_map
    
#     def weighted_spatial_pooling(self, ssim_map, msa_map):
#         """
#         STAGE C: Weighted spatial pooling using MSA
#         """
#         if ssim_map.shape != msa_map.shape:
#             msa_map = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))
        
#         # 1. Create an "error map". Where SSIM is 1 (good), error is 0.
#         error_map = 1.0 - ssim_map
        
#         # 2. Add a "Noise Gate" to the error map.
#         # This is the value we've been tuning.
#         ERROR_THRESHOLD = 0.30
        
#         # Create a "clean" error map that only contains "real" errors.
#         clean_error_map = np.where(error_map < ERROR_THRESHOLD, 0.0, error_map)

#         # 3. Create the "saliency-weighted error" using the CLEAN map
#         saliency_weighted_error = clean_error_map * msa_map
        
#         # 4. Calculate the average "dumb" error (our old baseline)
#         avg_baseline_error = np.mean(error_map)
        
#         # 5. Calculate the average "smart" error (only on moving parts)
#         avg_saliency_error = np.sum(saliency_weighted_error) / (np.sum(msa_map) + 1e-8)
        
#         # 6. Our final score is the "dumb" error plus the "smart" error.
#         total_error = avg_baseline_error + avg_saliency_error
        
#         # 7. Convert the total error back to a "quality" score
#         final_quality_score = 1.0 - total_error
        
        
#         # --- DEBUG VISUALIZATION ---
#         if self.debug:
#             # Show the raw SSIM map (good=1, bad=0)
#             cv2.imshow("Stage C: SSIM Map (Raw)", ssim_map)
            
#             # Show the "clean" error map (0-1, 0=good)
#             # This will show us if the 0.40 threshold is filtering *everything*
#             cv2.imshow("Stage C: Error Map (Cleaned)", clean_error_map)

#             # Show the "penalty" map. This is what's causing the score drop.
#             # We normalize it so we can see it (it's usually very dim)
#             penalty_map_visible = saliency_weighted_error.copy()
#             if np.max(penalty_map_visible) > 1e-8:
#                  penalty_map_visible /= np.max(penalty_map_visible)
            
#             cv2.imshow("Stage C: Saliency Overlap (Penalty)", penalty_map_visible)

#             cv2.waitKey(1)
#         # --- END DEBUG ---

        
#         return max(0, final_quality_score)
        
    
#     def process_frame_pair(self, ref_frame, dist_frame, msa_map):
#         """
#         STAGE C COMPLETE: Compute weighted frame quality
#         """
#         # Compute per-pixel SSIM
#         ssim_map = self.compute_ssim_map(ref_frame, dist_frame)
        
#         # Apply MSA weighting
#         frame_quality = self.weighted_spatial_pooling(ssim_map, msa_map)
        
#         return frame_quality

# stage_c.py
# Stage C: Weighted Spatial Pooling
# MODIFIED with debug visualization

# import numpy as np
# import cv2
# from skimage.metrics import structural_similarity as ssim


# import numpy as np
# import cv2
# from skimage.metrics import structural_similarity as ssim
# import os # <-- Import OS

# class StageCWeightedSpatialPooling:
    
#     # --- START OF CHECKPOINT FIX ---
#     def __init__(self, window_size=11, debug_video_name="debug", debug=True):
#         self.window_size = window_size
#         self.debug = debug
#         self.frame_count = 0
#         self.debug_video_name = debug_video_name
#         self.snapshot_saved = False

#         if self.debug:
#             print(f"--- STAGE C DEBUG ON (Saving for: {self.debug_video_name}) ---")
#             cv2.namedWindow("Stage C: Error Map")
#             cv2.namedWindow("Stage C: Weighted Error (Penalty)")
#     # --- END OF CHECKPOINT FIX ---

    
#     def compute_ssim_map(self, ref_frame, dist_frame):
#         """Compute pixel-wise SSIM between reference and distorted frames"""
#         if len(ref_frame.shape) == 3:
#             ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
#             dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
#         else:
#             ref_gray = ref_frame
#             dist_gray = dist_frame
        
#         # Compute SSIM map (full=True returns the map instead of just the score)
#         try:
#             # The 'data_range' is important. Grayscale is 0-255.
#             _, ssim_map = ssim(ref_gray, dist_gray, win_size=self.window_size, full=True, data_range=255)
#         except:
#             # Fallback for small frames/windows
#             _, ssim_map = ssim(ref_gray, dist_gray, win_size=5, full=True, data_range=255)
        
#         return ssim_map
    
#     def weighted_spatial_pooling(self, ssim_map, msa_map):
        
#         # --- START OF CHECKPOINT FIX ---
#         self.frame_count += 1
#         # --- END OF CHECKPOINT FIX ---

#         if ssim_map.shape != msa_map.shape:
#             msa_map = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))
        
#         # --- NEW ROBUST LOGIC ---
#         SALIENCY_PENALTY = 5.0 
#         error_map = 1.0 - ssim_map
#         base_weight = np.ones_like(msa_map)
#         weight_map = base_weight + (msa_map * SALIENCY_PENALTY)
#         weighted_error_map = error_map * weight_map
        
#         sum_of_weighted_error = np.sum(weighted_error_map)
#         sum_of_weights = np.sum(weight_map)
        
#         if sum_of_weights < 1e-8:
#             final_avg_error = np.mean(error_map)
#         else:
#             final_avg_error = sum_of_weighted_error / sum_of_weights
            
#         final_quality_score = 1.0 - final_avg_error
        
#         # --- DEBUG VISUALIZATION & SAVING ---
#         if self.debug:
#             cv2.imshow("Stage C: Error Map", error_map)
            
#             # Normalize penalty map to see it
#             penalty_visible = weighted_error_map / (1.0 + SALIENCY_PENALTY)
#             cv2.imshow("Stage C: Weighted Error (Penalty)", penalty_visible)
#             cv2.waitKey(1)
            
#             # --- START OF CHECKPOINT SAVE ---
#             if self.frame_count == 150 and not self.snapshot_saved:
#                 # Save the Error Map
#                 filename_err = f"{self.debug_video_name}_C_ERROR_MAP.png"
#                 print(f"  ... Saving Stage C snapshot: {filename_err}")
#                 cv2.imwrite(filename_err, (error_map * 255).astype(np.uint8))
                
#                 # Save the Penalty Map
#                 filename_pen = f"{self.debug_video_name}_C_PENALTY_MAP.png"
#                 print(f"  ... Saving Stage C snapshot: {filename_pen}")
#                 cv2.imwrite(filename_pen, (penalty_visible * 255).astype(np.uint8))
                
#                 self.snapshot_saved = True
#             # --- END OF CHECKPOINT SAVE ---

#         return max(0, final_quality_score)
        
    
#     def process_frame_pair(self, ref_frame, dist_frame, msa_map):
#         ssim_map = self.compute_ssim_map(ref_frame, dist_frame)
#         frame_quality = self.weighted_spatial_pooling(ssim_map, msa_map)
#         return frame_quality

# stage_c.py
# Stage C: Weighted Spatial Pooling (Enhanced motion sensitivity)

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class StageCWeightedSpatialPooling:
    def __init__(self, window_size=11, debug_video_name="debug", debug=True):
        self.window_size = window_size
        self.debug = debug
        self.frame_count = 0
        self.debug_video_name = debug_video_name
        self.snapshot_saved = False
        if self.debug:
            print(f"--- STAGE C DEBUG ON (Saving for: {self.debug_video_name}) ---")
            cv2.namedWindow("Stage C: Error Map")
            cv2.namedWindow("Stage C: Weighted Error (Penalty)")

    def compute_ssim_map(self, ref_frame, dist_frame):
        if len(ref_frame.shape) == 3:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray, dist_gray = ref_frame, dist_frame
        try:
            _, ssim_map = ssim(ref_gray, dist_gray, win_size=7, full=True, data_range=255)
        except Exception:
            _, ssim_map = ssim(ref_gray, dist_gray, win_size=5, full=True, data_range=255)
        return ssim_map

    def weighted_spatial_pooling(self, ssim_map, msa_map):
        self.frame_count += 1
        if ssim_map.shape != msa_map.shape:
            msa_map = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))
        error_map = 1.0 - ssim_map

        # Stronger motion emphasis
        SALIENCY_PENALTY = 10.0
        GAMMA = 1.5
        msa_enhanced = np.power(msa_map, 1.0 / GAMMA)
        weight_map = 1.0 + (msa_enhanced * SALIENCY_PENALTY)
        weighted_error = error_map * weight_map

        sum_werr, sum_w = np.sum(weighted_error), np.sum(weight_map)
        final_avg_err = np.mean(error_map) if sum_w < 1e-8 else sum_werr / sum_w
        final_quality = 1.0 - final_avg_err

        if self.debug:
            cv2.imshow("Stage C: Error Map", error_map)
            pen_vis = weighted_error / (1.0 + SALIENCY_PENALTY)
            cv2.imshow("Stage C: Weighted Error (Penalty)", pen_vis)
            cv2.waitKey(1)
            if self.frame_count % 50 == 0:
                print(f"[Stage C] Frame {self.frame_count}: "
                      f"sum(msa)={np.sum(msa_map):.1f} mean(msa)={np.mean(msa_map):.4f} "
                      f"base_err={np.mean(error_map):.4f} w_err={final_avg_err:.4f}")
            if self.frame_count == 150 and not self.snapshot_saved:
                cv2.imwrite(f"{self.debug_video_name}_C_ERROR_MAP.png",
                            (error_map * 255).astype(np.uint8))
                cv2.imwrite(f"{self.debug_video_name}_C_PENALTY_MAP.png",
                            (pen_vis * 255).astype(np.uint8))
                self.snapshot_saved = True

        return max(0.0, final_quality)

    def process_frame_pair(self, ref_frame, dist_frame, msa_map):
        ssim_map = self.compute_ssim_map(ref_frame, dist_frame)
        return self.weighted_spatial_pooling(ssim_map, msa_map)
