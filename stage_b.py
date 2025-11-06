# stage_b.py
# Stage B: Motion Saliency Awareness (MSA) Generation

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class StageBMSAGeneration:
    """
    Stage B: Motion Saliency Awareness (MSA) Generation
    Creates spatial weight map emphasizing distortions on moving objects
    """
    
    def __init__(self, farneback_params=None):
        """Initialize Stage B with Farneback optical flow parameters"""
        self.farneback_params = farneback_params or {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'n_poly': 5,
            'poly_sigma': 1.2,
            'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        }
    
    def compute_dense_optical_flow(self, prev_gray, curr_gray):
        """Compute dense optical flow using Farneback algorithm"""
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=self.farneback_params['pyr_scale'],
            levels=self.farneback_params['levels'],
            winsize=self.farneback_params['winsize'],
            iterations=self.farneback_params['iterations'],
            poly_n=self.farneback_params['n_poly'],
            poly_sigma=self.farneback_params['poly_sigma'],
            flags=self.farneback_params['flags']
        )
        
        return flow
    
    def calculate_motion_magnitude(self, flow):
        """Calculate motion magnitude from optical flow vectors"""
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        motion_mag = np.sqrt(u**2 + v**2)
        
        return motion_mag
    
    def calculate_difference_map(self, curr_frame, warped_prev):
        """Calculate absolute difference between current and warped previous frame"""
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(float)
            prev_gray = cv2.cvtColor(warped_prev, cv2.COLOR_BGR2GRAY).astype(float)
        else:
            curr_gray = curr_frame.astype(float)
            prev_gray = warped_prev.astype(float)
        
        diff_map = np.abs(curr_gray - prev_gray)
        diff_map_norm = diff_map / 255.0
        
        return diff_map_norm
    
    def generate_msa_map(self, curr_frame, warped_prev, motion_mag, smoothing_sigma=1.0):
        """Generate Motion Saliency Awareness (MSA) map"""
        diff_map_norm = self.calculate_difference_map(curr_frame, warped_prev)
        
        max_motion = np.max(motion_mag)
        if max_motion > 1e-8:
            motion_mag_norm = motion_mag / max_motion
        else:
            motion_mag_norm = motion_mag
        
        # --- START OF FIX ---
        # Threshold the map to remove noise.
        # We only care about motion stronger than 10% (0.1).
        threshold = 0.2
        _, msa_map = cv2.threshold(diff_map_norm, threshold, 1.0, cv2.THRESH_BINARY)
        # --- END OF FIX ---
        
        msa_map = gaussian_filter(msa_map, sigma=smoothing_sigma)
        
        # --- START OF MISSING CODE ---
        # This part was missing from your file
        min_msa = np.min(msa_map)
        max_msa = np.max(msa_map)
        
        if max_msa - min_msa > 1e-8:
            msa_map = (msa_map - min_msa) / (max_msa - min_msa)
        else:
            msa_map = np.zeros_like(msa_map)
            
        return msa_map  # <--- THE MISSING RETURN STATEMENT
        # --- END OF MISSING CODE ---
    
    def process_frame_pair(self, prev_frame, curr_frame, warped_prev):
        """STAGE B COMPLETE: Generate MSA map for frame pair"""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = self.compute_dense_optical_flow(prev_gray, curr_gray)
        motion_mag = self.calculate_motion_magnitude(flow)
        diff_map = self.calculate_difference_map(curr_frame, warped_prev)
        msa_map = self.generate_msa_map(curr_frame, warped_prev, motion_mag)
        
        return msa_map, motion_mag, diff_map
