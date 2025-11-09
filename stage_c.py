# stage_c.py
# Stage C: Weighted Spatial Pooling

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class StageCWeightedSpatialPooling:

    
    def __init__(self, window_size=11):

        self.window_size = window_size
    
    def compute_ssim_map(self, ref_frame, dist_frame):
        #Compute pixel-wise SSIM between reference and distorted frames
        if len(ref_frame.shape) == 3:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_frame
            dist_gray = dist_frame

        try:
            _, ssim_map = ssim(ref_gray, dist_gray, win_size=self.window_size, full=True)
        except:
            _, ssim_map = ssim(ref_gray, dist_gray, win_size=5, full=True)
        
        return ssim_map
    
    def weighted_spatial_pooling(self, ssim_map, msa_map):

        if ssim_map.shape != msa_map.shape:
            msa_map = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))

        error_map = 1.0 - ssim_map
        
        # Create the saliency weighted error.
        saliency_weighted_error = error_map * msa_map
        
        # Calculate the average dumb baseline error 
        avg_baseline_error = np.mean(error_map)
        
        # Calculate the average smart error on moving parts
        avg_saliency_error = np.sum(saliency_weighted_error) / (np.sum(msa_map) + 1e-8)
        
        total_error = avg_baseline_error + avg_saliency_error
        
        # Convert the total error back to a quality score
        final_quality_score = 1.0 - total_error
        
        return max(0, final_quality_score)
        
    
    def process_frame_pair(self, ref_frame, dist_frame, msa_map):

        ssim_map = self.compute_ssim_map(ref_frame, dist_frame)
        
        frame_quality = self.weighted_spatial_pooling(ssim_map, msa_map)
        
        return frame_quality
