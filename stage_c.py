# stage_c.py
# Stage C: Weighted Spatial Pooling

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim


class StageCWeightedSpatialPooling:
    """
    Stage C: Weighted Spatial Pooling
    Applies MSA weights to SSIM to get frame-level quality scores
    """
    
    def __init__(self, window_size=11):
        """Initialize Stage C"""
        self.window_size = window_size
    
    def compute_ssim_map(self, ref_frame, dist_frame):
        """Compute pixel-wise SSIM between reference and distorted frames"""
        if len(ref_frame.shape) == 3:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_frame
            dist_gray = dist_frame
        
        # Compute SSIM map (full=True returns the map instead of just the score)
        try:
            _, ssim_map = ssim(ref_gray, dist_gray, win_size=self.window_size, full=True)
        except:
            # If window size is too large, use smaller window
            _, ssim_map = ssim(ref_gray, dist_gray, win_size=5, full=True)
        
        return ssim_map
    
    def weighted_spatial_pooling(self, ssim_map, msa_map):
        """
        STAGE C: Weighted spatial pooling using MSA
        
        Q_n = sum(SSIM * MSA) / sum(MSA)
        """
        # Ensure same shape
        if ssim_map.shape != msa_map.shape:
            msa_resized = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))
        else:
            msa_resized = msa_map
        
        # Weighted average
        numerator = np.sum(ssim_map * msa_resized)
        denominator = np.sum(msa_resized)
        
        if denominator < 1e-8:
            frame_quality = np.mean(ssim_map)
        else:
            frame_quality = numerator / denominator
        
        return frame_quality
    
    def process_frame_pair(self, ref_frame, dist_frame, msa_map):
        """
        STAGE C COMPLETE: Compute weighted frame quality
        
        Args:
            ref_frame: Reference frame
            dist_frame: Distorted frame
            msa_map: MSA weight map from Stage B
            
        Returns:
            frame_quality: Weighted frame quality score (0-1)
        """
        # Compute per-pixel SSIM
        ssim_map = self.compute_ssim_map(ref_frame, dist_frame)
        
        # Apply MSA weighting
        frame_quality = self.weighted_spatial_pooling(ssim_map, msa_map)
        
        return frame_quality
