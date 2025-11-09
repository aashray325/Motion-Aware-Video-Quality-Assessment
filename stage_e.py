# stage_e.py
# Stage E: Weighted Temporal Pooling

import numpy as np


class StageEWeightedTemporalPooling:

    
    def __init__(self):
        pass
    
    def weighted_temporal_pooling(self, frame_quality_scores, gmi_values):
    
        frame_quality_scores = np.array(frame_quality_scores)
        gmi_values = np.array(gmi_values)
        
        if np.sum(gmi_values) < 1e-8:
            final_score = np.mean(frame_quality_scores)
        else:
            final_score = np.sum(frame_quality_scores * gmi_values) / (np.sum(gmi_values) + 1e-8)
        
        return final_score
    
    def process_scores(self, frame_quality_scores, gmi_values, baseline_scores):

        final_score = self.weighted_temporal_pooling(frame_quality_scores, gmi_values)
        
        statistics = {
            'final_score': final_score,
            'mean_frame_quality': np.mean(baseline_scores),
            'median_frame_quality': np.median(baseline_scores),
            'mean_gmi': np.mean(gmi_values),
            'max_gmi': np.max(gmi_values),
            'min_gmi': np.min(gmi_values),
            'num_frames': len(frame_quality_scores),
        }
        
        return final_score, statistics