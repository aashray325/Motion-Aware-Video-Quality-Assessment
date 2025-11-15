# stage_c.py
# Stage C: Weighted Spatial Pooling Calculation

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

class StageCWeightedSpatialPooling:
    def __init__(self, window_size=7, debug_video_name="debug", debug=True, pooling_mode="localized"):
        self.window_size = window_size
        self.debug = debug
        self.frame_count = 0
        self.debug_video_name = debug_video_name
        self.snapshot_saved = False
        self.SALIENCY_PENALTY = 20.0
        self.GAMMA = 1.2
        self.POOLING_MODE = pooling_mode
        self.TOP_PERCENTILE = 92.0

        if self.debug:
            cv2.namedWindow("Stage C: Error Map")
            cv2.namedWindow("Stage C: Weighted Error (Penalty)")
            cv2.namedWindow("Stage C: Weighted Error Overlay")

    def compute_ssim_map(self, ref_frame, dist_frame):
        h = min(ref_frame.shape[0], dist_frame.shape[0])
        w = min(ref_frame.shape[1], dist_frame.shape[1])
        ref_frame = cv2.resize(ref_frame, (w, h)).astype(np.uint8)
        dist_frame = cv2.resize(dist_frame, (w, h)).astype(np.uint8)

        try:
            _, ssim_map = ssim(ref_frame, dist_frame, win_size=self.window_size,
                               full=True, data_range=255, channel_axis=-1)
        except Exception:
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            dist_gray = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2GRAY)
            _, ssim_map = ssim(ref_gray, dist_gray,
                               win_size=min(self.window_size, 5),
                               full=True, data_range=255)
        if ssim_map.ndim == 3:
            ssim_map = np.mean(ssim_map, axis=2)
        return ssim_map

    def weighted_spatial_pooling(self, ssim_map, msa_map):
        self.frame_count += 1
        if ssim_map.shape != msa_map.shape:
            msa_map = cv2.resize(msa_map, (ssim_map.shape[1], ssim_map.shape[0]))

        error_map = 1.0 - ssim_map
        msa_enhanced = np.power(msa_map + 1e-12, 1.0 / self.GAMMA)

        if self.POOLING_MODE == "localized" and np.sum(msa_enhanced) >= 1e-12:
            thresh = np.percentile(msa_enhanced, self.TOP_PERCENTILE)
            salient_mask = (msa_enhanced >= thresh).astype(np.float32)
            if np.sum(salient_mask) < 10:
                salient_mask = (msa_enhanced >= np.percentile(msa_enhanced, 70.0)).astype(np.float32)

            weighted_error = error_map * (1.0 + self.SALIENCY_PENALTY * msa_enhanced) * salient_mask
            mean_salient_error = np.sum(weighted_error) / (np.sum(salient_mask) + 1e-8)
            mean_error_global = np.mean(error_map)
            mean_error = 0.75 * mean_salient_error + 0.25 * mean_error_global
        else:
            weight_map = 1.0 + (msa_enhanced * self.SALIENCY_PENALTY)
            weighted_error_map = error_map * weight_map
            mean_error = np.sum(weighted_error_map) / (np.sum(weight_map) + 1e-8)
            mean_salient_error = np.sum(error_map * (msa_enhanced > 0.001)) / (np.sum(msa_enhanced > 0.001) + 1e-8)

        final_quality = 1.0 - mean_error

        if self.debug:
            pen_vis = error_map * (1.0 + self.SALIENCY_PENALTY * msa_enhanced)
            pen_vis = pen_vis / (np.max(pen_vis) + 1e-8)
            heatmap = cv2.applyColorMap((pen_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted((np.dstack([ssim_map]*3)*255).astype(np.uint8), 0.5, heatmap, 0.5, 0)

            cv2.imshow("Stage C: Error Map", np.clip(error_map, 0, 1))
            cv2.imshow("Stage C: Weighted Error (Penalty)", pen_vis)
            cv2.imshow("Stage C: Weighted Error Overlay", overlay)
            cv2.waitKey(1)

            if self.frame_count % 25 == 0:
                print(f"[Stage C] Frame {self.frame_count}: sum(msa)={np.sum(msa_map):.1f} "
                      f"mean(msa)={np.mean(msa_map):.4f} base_err={np.mean(error_map):.4f} "
                      f"salient_err={mean_salient_error:.4f} final_q={final_quality:.4f}")
            if self.frame_count == 150 and not self.snapshot_saved:
                cv2.imwrite(f"{self.debug_video_name}_C_ERROR_MAP.png", (np.clip(error_map,0,1)*255).astype(np.uint8))
                cv2.imwrite(f"{self.debug_video_name}_C_PENALTY_MAP.png", (pen_vis*255).astype(np.uint8))
                cv2.imwrite(f"{self.debug_video_name}_C_OVERLAY.png", overlay)
                self.snapshot_saved = True

        return float(np.clip(final_quality, 0.0, 1.0))

    def process_frame_pair(self, ref_frame, dist_frame, msa_map):
        ssim_map = self.compute_ssim_map(ref_frame, dist_frame)
        return self.weighted_spatial_pooling(ssim_map, msa_map)
