# stage_b.py
# Stage B: Motion Saliency Awareness (MSA) Generation

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class StageBMSAGeneration:

    
    def __init__(self):
        pass

    
    def calculate_difference_map(self, curr_frame, warped_prev):
        #This is the "residual map" of true object motion.
        
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(float)
            prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            curr_gray = curr_frame.astype(float)
            prev_gray = warped_prev.astype(float)
        
        diff_map = np.abs(curr_gray - prev_gray)
        diff_map_norm = diff_map / 255.0
        
        return diff_map_norm

    
    def generate_msa_map(self, curr_frame, warped_prev, smoothing_sigma=1.0):
        #Generate the final Motion Saliency Awareness (MSA) map.

        diff_map_norm = self.calculate_difference_map(curr_frame, warped_prev)

        diff_map_uint8 = (diff_map_norm * 255).astype(np.uint8)

        threshold_value, msa_map_binary = cv2.threshold(
            diff_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        msa_map = msa_map_binary / 255.0

        msa_map = gaussian_filter(msa_map, sigma=smoothing_sigma)
        
        min_msa = np.min(msa_map)
        max_msa = np.max(msa_map)
        
        if max_msa - min_msa > 1e-8:
            msa_map = (msa_map - min_msa) / (max_msa - min_msa)
        else:
            msa_map = np.zeros_like(msa_map)
            
        return msa_map
    

    def process_frame_pair(self, prev_frame, curr_frame, warped_prev):

        msa_map = self.generate_msa_map(curr_frame, warped_prev)

        return msa_map