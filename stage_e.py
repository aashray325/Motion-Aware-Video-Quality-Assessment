# stage_e.py
# Stage E: Weighted Temporal Pooling

import numpy as np


class StageEWeightedTemporalPooling:
    """
    Stage E: Weighted Temporal Pooling
    Combines per-frame quality scores with GMI weights for final VQA score
    """
    
    def __init__(self):
        """Initialize Stage E"""
        pass
    
    def weighted_temporal_pooling(self, frame_quality_scores, gmi_values):
        """
        STAGE E: Weighted temporal pooling using GMI
        
        Final_Score = sum(Q_n * GMI_n) / sum(GMI_n)
        
        Args:
            frame_quality_scores: Per-frame quality scores (Q_n) - list or array
            gmi_values: Global Motion Indicator values - list or array
            
        Returns:
            final_score: Motion-aware VQA score (0-1)
        """
        frame_quality_scores = np.array(frame_quality_scores)
        gmi_values = np.array(gmi_values)
        
        # Ensure GMI is not all zeros
        if np.sum(gmi_values) < 1e-8:
            # If no motion detected, use unweighted average
            final_score = np.mean(frame_quality_scores)
        else:
            # Weighted average
            final_score = np.sum(frame_quality_scores * gmi_values) / np.sum(gmi_values)
        
        return final_score
    
    def process_scores(self, frame_quality_scores, gmi_values):
        """
        STAGE E COMPLETE: Compute final motion-aware VQA score
        
        Args:
            frame_quality_scores: List of per-frame quality scores
            gmi_values: List of GMI values for each frame
            
        Returns:
            final_score: Final motion-aware VQA score (0-1)
            statistics: Dictionary with additional statistics
        """
        final_score = self.weighted_temporal_pooling(frame_quality_scores, gmi_values)
        
        # Compute statistics
        statistics = {
            'final_score': final_score,
            'mean_frame_quality': np.mean(frame_quality_scores),
            'median_frame_quality': np.median(frame_quality_scores),
            'mean_gmi': np.mean(gmi_values),
            'max_gmi': np.max(gmi_values),
            'min_gmi': np.min(gmi_values),
            'num_frames': len(frame_quality_scores),
        }
        
        return final_score, statistics
