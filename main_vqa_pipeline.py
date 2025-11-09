# import numpy as np
# import cv2
# from pathlib import Path

# from stage_a import StageAMotionEstimation
# from stage_b import StageBMSAGeneration
# from stage_c import StageCWeightedSpatialPooling
# from stage_d import StageDGMICalculation
# from stage_e import StageEWeightedTemporalPooling

# import numpy as np
# import cv2
# from pathlib import Path

# from stage_a import StageAMotionEstimation
# from stage_b import StageBMSAGeneration
# from stage_c import StageCWeightedSpatialPooling
# from stage_d import StageDGMICalculation
# from stage_e import StageEWeightedTemporalPooling

# # ===== CONFIGURATION =====
# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 960
# MAX_FRAMES = None
# # =========================

# class MotionAwareVQAPipeline:
#     """Complete VQA Pipeline with all stages (A-E)"""
    
#     def __init__(self, frame_width=1280, frame_height=960):
#         self.frame_width = frame_width
#         self.frame_height = frame_height
        
#         # We will re-initialize these in process_video_pair
#         # so we can pass them the video name for debugging
#         self.stage_a = StageAMotionEstimation(debug=False) # Turn off Stage A debug
#         self.stage_b = None
#         self.stage_c = None
        
#         self.stage_d = StageDGMICalculation()
#         self.stage_e = StageEWeightedTemporalPooling()
    
#     def process_video_pair(self, ref_video_path, dist_video_path, max_frames=None):
#         if not Path(ref_video_path).exists():
#             print(f"✗ Reference video not found: {ref_video_path}")
#             return None
        
#         if not Path(dist_video_path).exists():
#             print(f"✗ Distorted video not found: {dist_video_path}")
#             return None
        
#         print(f"\nProcessing: {Path(dist_video_path).name}")
        
#         # --- START OF CHECKPOINT FIX ---
#         # Get the unique video name (e.g., "distorted_static_real")
#         video_name = Path(dist_video_path).stem
        
#         # Pass the name to the stages so they can save debug images
#         self.stage_b = StageBMSAGeneration(debug_video_name=video_name, debug=True)
#         self.stage_c = StageCWeightedSpatialPooling(debug_video_name=video_name, debug=True)
#         # --- END OF CHECKPOINT FIX ---

#         ref_cap = cv2.VideoCapture(ref_video_path)
#         dist_cap = cv2.VideoCapture(dist_video_path)
        
#         if not ref_cap.isOpened() or not dist_cap.isOpened():
#             print("✗ Cannot open one or both video files")
#             return None
        
#         ret, prev_ref_frame = ref_cap.read()
#         if not ret:
#             print("✗ Cannot read first frame")
#             return None
        
#         prev_ref_frame = cv2.resize(prev_ref_frame, (self.frame_width, self.frame_height))
        
#         frame_count = 0
#         H_list = []
#         frame_quality_scores = [] # "Smart" scores
#         baseline_scores = []      # "Dumb" scores

#         msa_sums = []
#         msa_maps = []
        
#         while True:
#             ret_ref, ref_frame = ref_cap.read()
#             ret_dist, dist_frame = dist_cap.read()
            
#             if not ret_ref or not ret_dist:
#                 break
            
#             if max_frames and frame_count >= max_frames:
#                 break
            
#             ref_frame = cv2.resize(ref_frame, (self.frame_width, self.frame_height))
#             dist_frame = cv2.resize(dist_frame, (self.frame_width, self.frame_height))
            
#             # --- STAGE A ---
#             H, warped_prev, _ = self.stage_a.process_frame_pair(prev_ref_frame, ref_frame)
#             H_list.append(H)
            
#             # --- STAGE B ---
#             msa_map = self.stage_b.process_frame_pair(prev_ref_frame, ref_frame, warped_prev)
            
#             # --- STAGE C ---
            
#             # 1. Get the raw error map
#             ssim_map = self.stage_c.compute_ssim_map(ref_frame, dist_frame)
            
#             # 2. Calculate the "dumb" baseline score (simple average)
#             baseline_frame_score = np.mean(ssim_map)
#             baseline_scores.append(baseline_frame_score)
            
#             # 3. Calculate the "smart" motion-aware score
#             frame_quality = self.stage_c.weighted_spatial_pooling(ssim_map, msa_map)
#             frame_quality_scores.append(frame_quality)
            
#             # --- End of Stage C ---
            
#             frame_count += 1
#             prev_ref_frame = ref_frame.copy()

#             sum_msa = np.sum(msa_map)
#             mean_msa = np.mean(msa_map)
#             sum_error = np.sum(1.0 - ssim_map)            # total error mass
#             mean_error = np.mean(1.0 - ssim_map)
#             # compute the weighted error map used in StageC (this duplicates internal logic)
#             SALIENCY_PENALTY = 1.0   # match StageC value
#             weight_map = np.ones_like(msa_map) + (msa_map * SALIENCY_PENALTY)
#             weighted_error_map = (1.0 - ssim_map) * weight_map
#             sum_weighted_error = np.sum(weighted_error_map)

#             print(f"Frame {frame_count:04d}: sum_msa={sum_msa:.2f}, mean_msa={mean_msa:.4f}, "
#                   f"sum_err={sum_error:.2f}, sum_w_err={sum_weighted_error:.2f}, "
#                   f"baseline_mean={baseline_frame_score:.4f}, frame_quality={frame_quality:.4f}")
        
#         ref_cap.release()
#         dist_cap.release()
        
#         print(f"  ✓ Processed {frame_count} frames")
        
#         # --- STAGE D ---
#         gmi_values = self.stage_d.process_homography_list(H_list)
        
#         # --- STAGE E ---
#         final_score, statistics = self.stage_e.process_scores(
#             frame_quality_scores, gmi_values, baseline_scores
#         )
        
#         baseline_score = statistics['mean_frame_quality']
        
#         return final_score, baseline_score


# def run_vqa_analysis(ref_path, dist_path):
#     pipeline = MotionAwareVQAPipeline(
#         frame_width=FRAME_WIDTH,
#         frame_height=FRAME_HEIGHT
#     )
    
#     results = pipeline.process_video_pair(
#         ref_video_path=ref_path,
#         dist_video_path=dist_path,
#         max_frames=MAX_FRAMES
#     )
    
#     if results:
#         final_aware_score, final_baseline_score = results
#         return final_aware_score, final_baseline_score
#     else:
#         return None, None

# if __name__ == "__main__":
#     print("Running main_vqa_pipeline.py in standalone mode")
    
#     REF_VIDEO = "data/reference.mp4"
#     DIST_VIDEO = "data/distorted_motion.mp4"
    
#     aware_score, baseline_score = run_vqa_analysis(REF_VIDEO, DIST_VIDEO)
    
#     if aware_score is not None:
#         print("\nSTANDALONE TEST RESULTS")
#         print(f"  Reference: {REF_VIDEO}")
#         print(f"  Distorted: {DIST_VIDEO}")
#         print(f"  Motion-Aware Score: {aware_score:.4f}")
#         print(f"  Baseline Score:     {baseline_score:.4f}")

# main_vqa_pipeline.py
# Complete VQA pipeline integrating all stages

# main_vqa_pipeline.py
# Complete VQA pipeline integrating all stages

import numpy as np
import cv2
from pathlib import Path
from stage_a import StageAMotionEstimation
from stage_b import StageBMSAGeneration
from stage_c import StageCWeightedSpatialPooling
from stage_d import StageDGMICalculation
from stage_e import StageEWeightedTemporalPooling


FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MAX_FRAMES = None


class MotionAwareVQAPipeline:
    def __init__(self, frame_width=1920, frame_height=1080):
        self.frame_width, self.frame_height = frame_width, frame_height
        self.stage_a = StageAMotionEstimation(debug=False)
        self.stage_d = StageDGMICalculation()
        self.stage_e = StageEWeightedTemporalPooling(use_gmi=True)

    def process_video_pair(self, ref_video_path, dist_video_path, max_frames=None):
        ref_cap, dist_cap = cv2.VideoCapture(ref_video_path), cv2.VideoCapture(dist_video_path)
        if not ref_cap.isOpened() or not dist_cap.isOpened():
            print("✗ Could not open videos."); return None

        video_name = Path(dist_video_path).stem
        self.stage_b = StageBMSAGeneration(debug_video_name=video_name, debug=True)
        self.stage_c = StageCWeightedSpatialPooling(debug_video_name=video_name, debug=True)

        ret, prev_ref = ref_cap.read()
        if not ret:
            print("✗ Cannot read first frame"); return None
        prev_ref = cv2.resize(prev_ref, (self.frame_width, self.frame_height))

        frame_count, H_list = 0, []
        frame_quality_scores, baseline_scores, msa_sums = [], [], []

        while True:
            ret_r, ref_frame = ref_cap.read()
            ret_d, dist_frame = dist_cap.read()
            if not ret_r or not ret_d or (max_frames and frame_count >= max_frames):
                break

            ref_frame = cv2.resize(ref_frame, (self.frame_width, self.frame_height))
            dist_frame = cv2.resize(dist_frame, (self.frame_width, self.frame_height))

            H, warped_prev, _ = self.stage_a.process_frame_pair(prev_ref, ref_frame)
            msa_map = self.stage_b.process_frame_pair(prev_ref, ref_frame, warped_prev)
            msa_sums.append(np.sum(msa_map))

            ssim_map = self.stage_c.compute_ssim_map(ref_frame, dist_frame)
            baseline_scores.append(np.mean(ssim_map))
            frame_quality_scores.append(self.stage_c.weighted_spatial_pooling(ssim_map, msa_map))
            H_list.append(H)
            frame_count += 1
            prev_ref = ref_frame.copy()

        print(f"  ✓ Processed {frame_count} frames")

        # Normalize MSA globally (scaling stage)
        global_max = max(msa_sums) if msa_sums else 1.0
        scale_factor = np.mean(msa_sums) / (global_max + 1e-8)
        frame_quality_scores = [q * scale_factor for q in frame_quality_scores]

        gmi_values = self.stage_d.process_homography_list(H_list)
        final_score, stats = self.stage_e.process_scores(frame_quality_scores, gmi_values, baseline_scores)
        return final_score, stats["mean_frame_quality"]


def run_vqa_analysis(ref_path, dist_path):
    pipeline = MotionAwareVQAPipeline(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    res = pipeline.process_video_pair(ref_path, dist_path, MAX_FRAMES)
    if res:
        return res
    return None, None


if __name__ == "__main__":
    print("Running main_vqa_pipeline.py standalone test")
    REF_VIDEO = "data/reference.mp4"
    DIST_VIDEO = "data/distorted_motion_real.mp4"
    aware, base = run_vqa_analysis(REF_VIDEO, DIST_VIDEO)
    if aware:
        print(f"Motion-Aware Score: {aware:.4f} | Baseline: {base:.4f}")

