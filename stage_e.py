# stage_e.py
# Stage E: Weighted Temporal Pooling

import numpy as np


class StageEWeightedTemporalPooling:
    def __init__(self, use_gmi=True):
        self.use_gmi = use_gmi

    def weighted_temporal_pooling(self, frame_quality_scores, gmi_values):
        fq = np.array(frame_quality_scores)
        gmi = np.array(gmi_values)
        if not self.use_gmi:
            return float(np.mean(fq))
        weights = 1.0 + 2.0 * gmi
        return float(np.sum(fq * weights) / (np.sum(weights) + 1e-8))

    def process_scores(self, frame_quality_scores, gmi_values, baseline_scores):
        final_score = self.weighted_temporal_pooling(frame_quality_scores, gmi_values)
        return final_score, {
            "final_score": final_score,
            "mean_frame_quality": np.mean(baseline_scores),
            "median_frame_quality": np.median(baseline_scores),
            "mean_gmi": np.mean(gmi_values),
            "max_gmi": np.max(gmi_values),
            "min_gmi": np.min(gmi_values),
            "num_frames": len(frame_quality_scores),
        }
