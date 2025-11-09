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
        
        This new logic is more robust.
        It calculates a "penalty" based on where motion and errors overlap.
        """
        if ssim_map.shape != msa_map.shape:
            msa_map = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))
        
        # --- NEW LOGIC ---
        
        # 1. Create an "error map". Where SSIM is 1 (good), error is 0.
        #    Where SSIM is 0 (bad), error is 1.
        error_map = 1.0 - ssim_map
        
        # 2. Create the "saliency-weighted error".
        #    This is 0 everywhere, *except* where there is
        #    both an ERROR and MOTION (msa_map > 0).
        saliency_weighted_error = error_map * msa_map
        
        # 3. Calculate the average "dumb" error (our old baseline)
        avg_baseline_error = np.mean(error_map)
        
        # 4. Calculate the average "smart" error (only on moving parts)
        #    We must add 1e-8 to avoid dividing by zero if msa_map is all black
        avg_saliency_error = np.sum(saliency_weighted_error) / (np.sum(msa_map) + 1e-8)
        
        # 5. Our final score is the "dumb" error plus the "smart" error.
        #    This means errors on moving parts are "double-counted"
        #    and penalized much more heavily.
        total_error = avg_baseline_error + avg_saliency_error
        
        # 6. Convert the total error back to a "quality" score
        #    (where 1.0 is good and 0.0 is bad).
        final_quality_score = 1.0 - total_error
        
        # Ensure the score is not negative
        return max(0, final_quality_score)
        
        # --- END NEW LOGIC ---
    
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
