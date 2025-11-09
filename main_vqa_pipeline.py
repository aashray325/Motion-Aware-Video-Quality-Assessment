import numpy as np
import cv2
from pathlib import Path

from stage_a import StageAMotionEstimation
from stage_b import StageBMSAGeneration
from stage_c import StageCWeightedSpatialPooling
from stage_d import StageDGMICalculation
from stage_e import StageEWeightedTemporalPooling

FRAME_WIDTH = 360
FRAME_HEIGHT = 640
MAX_FRAMES = None  

class MotionAwareVQAPipeline:
    """Complete VQA Pipeline with all stages (A-E)"""
    
    def __init__(self, frame_width=360, frame_height=640):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.stage_a = StageAMotionEstimation()
        self.stage_b = StageBMSAGeneration()
        self.stage_c = StageCWeightedSpatialPooling()
        self.stage_d = StageDGMICalculation()
        self.stage_e = StageEWeightedTemporalPooling()
    
    def process_video_pair(self, ref_video_path, dist_video_path, max_frames=None):
        if not Path(ref_video_path).exists():
            print(f"Reference video not found: {ref_video_path}")
            return None
        
        if not Path(dist_video_path).exists():
            print(f"Distorted video not found: {dist_video_path}")
            return None
        
        print(f"\nProcessing: {Path(dist_video_path).name}")
        
        ref_cap = cv2.VideoCapture(ref_video_path)
        dist_cap = cv2.VideoCapture(dist_video_path)
        
        if not ref_cap.isOpened() or not dist_cap.isOpened():
            print("Cannot open one or both video files")
            return None
        
        ret, prev_ref_frame = ref_cap.read()
        if not ret:
            print("Cannot read first frame")
            return None
        
        prev_ref_frame = cv2.resize(prev_ref_frame, (self.frame_width, self.frame_height))
        
        frame_count = 0
        H_list = []
        frame_quality_scores = [] 
        baseline_scores = []      
        
        while True:
            ret_ref, ref_frame = ref_cap.read()
            ret_dist, dist_frame = dist_cap.read()
            
            if not ret_ref or not ret_dist:
                break
            
            if max_frames and frame_count >= max_frames:
                break
            
            ref_frame = cv2.resize(ref_frame, (self.frame_width, self.frame_height))
            dist_frame = cv2.resize(dist_frame, (self.frame_width, self.frame_height))
            
            # STAGE A
            H, warped_prev, _ = self.stage_a.process_frame_pair(prev_ref_frame, ref_frame)
            H_list.append(H)
            
            # STAGE B
            msa_map = self.stage_b.process_frame_pair(prev_ref_frame, ref_frame, warped_prev)
            
            # STAGE C
            ssim_map = self.stage_c.compute_ssim_map(ref_frame, dist_frame)
            
            baseline_frame_score = np.mean(ssim_map)
            baseline_scores.append(baseline_frame_score)
            
            frame_quality = self.stage_c.weighted_spatial_pooling(ssim_map, msa_map)
            frame_quality_scores.append(frame_quality)
            
            frame_count += 1
            prev_ref_frame = ref_frame.copy()
        
        ref_cap.release()
        dist_cap.release()
        
        print(f"Processed {frame_count} frames")
        
        # STAGE D
        gmi_values = self.stage_d.process_homography_list(H_list)
        
        # STAGE E
        final_score, statistics = self.stage_e.process_scores(
            frame_quality_scores, gmi_values, baseline_scores
        )
        
        baseline_score = statistics['mean_frame_quality']
        
        return final_score, baseline_score

def run_vqa_analysis(ref_path, dist_path):
    pipeline = MotionAwareVQAPipeline(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT
    )
    
    results = pipeline.process_video_pair(
        ref_video_path=ref_path,
        dist_video_path=dist_path,
        max_frames=MAX_FRAMES
    )
    
    if results:
        final_aware_score, final_baseline_score = results
        return final_aware_score, final_baseline_score
    else:
        return None, None

if __name__ == "__main__":
    print("--- Running main_vqa_pipeline.py in standalone mode ---")
    
    REF_VIDEO = "data/reference.mp4"
    DIST_VIDEO = "data/distorted_motion.mp4"
    
    aware_score, baseline_score = run_vqa_analysis(REF_VIDEO, DIST_VIDEO)
    
    if aware_score is not None:
        print("\nSTANDALONE TEST RESULTS")
        print(f"  Reference: {REF_VIDEO}")
        print(f"  Distorted: {DIST_VIDEO}")
        print(f"  Motion-Aware Score: {aware_score:.4f}")
        print(f"  Baseline Score:     {baseline_score:.4f}")