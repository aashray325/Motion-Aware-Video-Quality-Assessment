# stage_d.py
# Stage D: Global Motion Indicator (GMI) Calculation

import numpy as np


class StageDGMICalculation:

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def extract_motion_parameters(self, H):
        #Extract motion parameters from homography matrix
        
        if H is None:
            return 0, 0, 0, 1

        tx = H[0, 2]
        ty = H[1, 2]

        a = H[0, 0]
        b = H[0, 1]
        c = H[1, 0]
        d = H[1, 1]

        theta = np.arctan2(b, a)
        
        scale = np.sqrt(a**2 + b**2)
        
        return tx, ty, theta, scale
    
    def calculate_gmi_frame(self, H):

        tx, ty, theta, scale = self.extract_motion_parameters(H)

        translation = np.sqrt(tx**2 + ty**2)
        rotation = np.abs(theta)
        zoom = np.abs(scale - 1.0)

        gmi = self.alpha * translation + self.beta * rotation + self.gamma * zoom
        
        return gmi
    
    def process_homography_list(self, H_list):

        gmi_values = []
        
        for H in H_list:
            gmi = self.calculate_gmi_frame(H)
            gmi_values.append(gmi)
        
        gmi_values = np.array(gmi_values)
        
        if len(gmi_values) > 0 and np.max(gmi_values) > 1e-8:
            gmi_min = np.min(gmi_values)
            gmi_max = np.max(gmi_values)
            gmi_values = (gmi_values - gmi_min) / (gmi_max - gmi_min + 1e-8)
        
        return gmi_values
