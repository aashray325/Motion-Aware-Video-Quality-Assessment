# stage_b.py
# Stage B: Motion Saliency Awareness (MSA) Generation

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class StageBMSAGeneration:
    """
    Stage B: Motion Saliency Awareness (MSA) Generation
    Creates a spatial weight map (MSA map) from the residual difference map.
    """
    
    def __init__(self):
        """Initialize Stage B."""
        # No parameters needed as we are not using Farneback.
        pass

    
    def calculate_difference_map(self, curr_frame, warped_prev):
        """
        Calculates the absolute difference between the current frame
        and the warped (stabilized) previous frame from Stage A.
        This is the "residual map" of true object motion.
        """
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
        """
        Generate the final Motion Saliency Awareness (MSA) map.
        This function thresholds the residual map to remove noise.
        """
        # 1. Get the raw residual motion (the "smart" motion)
        diff_map_norm = self.calculate_difference_map(curr_frame, warped_prev)
        
        # --- START OF FIX ---
        # The old threshold was a "magic number" (0.2) and was too low,
        # picking up background noise.
        #
        # We will now use Otsu's Binarization, which *automatically*
        # finds the best possible threshold to separate the
        # foreground (man) from the background (noise).

        # 2. Convert our 0.0-1.0 float map to a 0-255 integer map (uint8)
        #    so we can use OpenCV's thresholding tools.
        diff_map_uint8 = (diff_map_norm * 255).astype(np.uint8)

        # 3. Apply Otsu's threshold.
        #    This is the "smart" threshold that finds the man.
        threshold_value, msa_map_binary = cv2.threshold(
            diff_map_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        
        # 4. Convert back to a 0.0-1.0 float map for our calculations
        msa_map = msa_map_binary / 255.0
        # --- END OF FIX ---
        
        # 5. Smooth the map to avoid hard edges
        msa_map = gaussian_filter(msa_map, sigma=smoothing_sigma)
        
        # 6. Normalize the final map to a 0.0 - 1.0 range
        min_msa = np.min(msa_map)
        max_msa = np.max(msa_map)
        
        if max_msa - min_msa > 1e-8:
            msa_map = (msa_map - min_msa) / (max_msa - min_msa)
        else:
            # If map is all black, just return zeros
            msa_map = np.zeros_like(msa_map)
            
        return msa_map
    

    def process_frame_pair(self, prev_frame, curr_frame, warped_prev):
        """
        STAGE B COMPLETE: Generate MSA map for frame pair.
        This is the only function called by main_vqa_pipeline.py.
        """
        # Call generate_msa_map to do all the work
        msa_map = self.generate_msa_map(curr_frame, warped_prev)
        
        # Return *only* the msa_map, as it's all we need
        return msa_map