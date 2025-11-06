# stage_d.py
# Stage D: Global Motion Indicator (GMI) Calculation

import numpy as np


class StageDGMICalculation:
    """
    Stage D: Global Motion Indicator (GMI) Calculation
    Computes GMI values from homography matrices for temporal weighting
    """
    
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        """
        Initialize Stage D
        
        Args:
            alpha: Weight for translation component
            beta: Weight for rotation component
            gamma: Weight for scale/zoom component
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def extract_motion_parameters(self, H):
        """
        Extract motion parameters from homography matrix
        
        Args:
            H: Homography matrix (3x3)
            
        Returns:
            tx, ty, theta, scale: Motion parameters
        """
        if H is None:
            return 0, 0, 0, 1
        
        # Extract translation
        tx = H[0, 2]
        ty = H[1, 2]
        
        # Extract rotation and scale
        a = H[0, 0]
        b = H[0, 1]
        c = H[1, 0]
        d = H[1, 1]
        
        # Rotation angle
        theta = np.arctan2(b, a)
        
        # Scale (average of row norms)
        scale = np.sqrt(a**2 + b**2)
        
        return tx, ty, theta, scale
    
    def calculate_gmi_frame(self, H):
        """
        Calculate GMI for a single frame
        
        GMI = alpha * sqrt(tx^2 + ty^2) + beta * |theta| + gamma * |scale - 1|
        
        Args:
            H: Homography matrix for this frame
            
        Returns:
            gmi: Global Motion Indicator value
        """
        tx, ty, theta, scale = self.extract_motion_parameters(H)
        
        # Calculate individual components
        translation = np.sqrt(tx**2 + ty**2)
        rotation = np.abs(theta)
        zoom = np.abs(scale - 1.0)
        
        # Weighted combination
        gmi = self.alpha * translation + self.beta * rotation + self.gamma * zoom
        
        return gmi
    
    def process_homography_list(self, H_list):
        """
        STAGE D COMPLETE: Calculate GMI for all frames
        
        Args:
            H_list: List of homography matrices
            
        Returns:
            gmi_values: GMI value for each frame (normalized 0-1)
        """
        gmi_values = []
        
        for H in H_list:
            gmi = self.calculate_gmi_frame(H)
            gmi_values.append(gmi)
        
        # Normalize GMI values to 0-1 range
        gmi_values = np.array(gmi_values)
        
        if len(gmi_values) > 0 and np.max(gmi_values) > 1e-8:
            gmi_min = np.min(gmi_values)
            gmi_max = np.max(gmi_values)
            gmi_values = (gmi_values - gmi_min) / (gmi_max - gmi_min + 1e-8)
        
        return gmi_values
