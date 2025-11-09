# stage_d.py
# Stage D: Global Motion Indicator (GMI) Calculation

import numpy as np


class StageDGMICalculation:
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, scale_factor=2.0):
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.scale_factor = scale_factor

    def extract_motion_parameters(self, H):
        if H is None:
            return 0, 0, 0, 1
        tx, ty = H[0, 2], H[1, 2]
        a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
        theta = np.arctan2(b, a)
        scale = np.sqrt(a ** 2 + b ** 2)
        return tx, ty, theta, scale

    def calculate_gmi_frame(self, H):
        tx, ty, theta, scale = self.extract_motion_parameters(H)
        translation = np.sqrt(tx ** 2 + ty ** 2)
        rotation = np.abs(theta)
        zoom = np.abs(scale - 1.0)
        gmi = self.alpha * translation + self.beta * rotation + self.gamma * zoom
        return gmi * self.scale_factor

    def process_homography_list(self, H_list):
        gmis = np.array([self.calculate_gmi_frame(H) for H in H_list])
        if len(gmis) > 0 and np.max(gmis) > 1e-8:
            gmin, gmax = np.min(gmis), np.max(gmis)
            gmis = (gmis - gmin) / (gmax - gmin + 1e-8)
        return gmis
