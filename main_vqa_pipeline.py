import numpy as np
import cv2
from pathlib import Path
from stage_a import StageAMotionEstimation
from stage_b import StageBMSAGeneration
from stage_c import StageCWeightedSpatialPooling
from stage_d import StageDGMICalculation
from stage_e import StageEWeightedTemporalPooling

FRAME_WIDTH = 1280
FRAME_HEIGHT = 960
MAX_FRAMES = None


class MotionAwareVQAPipeline:
    def __init__(self, frame_width=1280, frame_height=960):
        self.frame_width, self.frame_height = frame_width, frame_height
        self.stage_a = StageAMotionEstimation(debug=True)
        self.stage_d = StageDGMICalculation()
        self.stage_e = StageEWeightedTemporalPooling(use_gmi=False)

    def process_video_pair(self, ref_video_path, dist_video_path, max_frames=None):
        ref_cap = cv2.VideoCapture(ref_video_path)
        dist_cap = cv2.VideoCapture(dist_video_path)
        if not ref_cap.isOpened() or not dist_cap.isOpened():
            print("Could not open videos.")
            return None

        video_name = Path(dist_video_path).stem
        self.stage_b = StageBMSAGeneration(debug_video_name=video_name, debug=True)
        self.stage_c = StageCWeightedSpatialPooling(debug_video_name=video_name, debug=True)

        ret_ref, prev_ref = ref_cap.read()
        ret_dist, prev_dist = dist_cap.read()
        if not ret_ref or not ret_dist:
            print("Cannot read first frames")
            return None

        prev_ref = cv2.resize(prev_ref, (self.frame_width, self.frame_height))
        prev_dist = cv2.resize(prev_dist, (self.frame_width, self.frame_height))

        frame_count = 0
        H_list, frame_quality_scores, baseline_scores, msa_sums = [], [], [], []

        while True:
            ret_r, ref_frame = ref_cap.read()
            ret_d, dist_frame = dist_cap.read()
            if not ret_r or not ret_d:
                break
            if max_frames and frame_count >= max_frames:
                break

            ref_frame = cv2.resize(ref_frame, (self.frame_width, self.frame_height))
            dist_frame = cv2.resize(dist_frame, (self.frame_width, self.frame_height))

            H, warped_prev_ref, _ = self.stage_a.process_frame_pair(prev_ref, ref_frame)
            H_list.append(H)

            try:
                h, w = prev_dist.shape[:2]
                warped_prev_dist = cv2.warpPerspective(prev_dist, H, (w, h))
            except Exception:
                warped_prev_dist = prev_dist.copy()

            msa_map = self.stage_b.process_frame_pair(prev_dist, dist_frame, warped_prev_dist)
            msa_sums.append(np.sum(msa_map))

            ssim_map = self.stage_c.compute_ssim_map(ref_frame, dist_frame)
            baseline_scores.append(np.mean(ssim_map))
            frame_quality = self.stage_c.weighted_spatial_pooling(ssim_map, msa_map)
            frame_quality_scores.append(frame_quality)

            if self.stage_b.debug:
                overlay = cv2.addWeighted(
                    cv2.cvtColor((msa_map * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                    0.4, dist_frame, 0.6, 0)
                cv2.imshow("Pipeline: MSA Overlay", overlay)
                cv2.waitKey(1)
                if frame_count == 150:
                    cv2.imwrite(f"{video_name}_PIPELINE_OVERLAY.png", overlay)

            frame_count += 1
            prev_ref, prev_dist = ref_frame.copy(), dist_frame.copy()

        ref_cap.release()
        dist_cap.release()
        print(f"Processed {frame_count} frames")

        gmi_values = self.stage_d.process_homography_list(H_list)
        final_score, stats = self.stage_e.process_scores(frame_quality_scores, gmi_values, baseline_scores)

        try:
            print(f"[Diag] msa_sum: min={np.min(msa_sums):.1f}, "
                  f"max={np.max(msa_sums):.1f}, mean={np.mean(msa_sums):.2f}")
        except Exception:
            pass

        return final_score, stats["mean_frame_quality"]


def run_vqa_analysis(ref_path, dist_path):
    pipeline = MotionAwareVQAPipeline(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    return pipeline.process_video_pair(ref_path, dist_path, MAX_FRAMES) or (None, None)


if __name__ == "__main__":
    print("Running main_vqa_pipeline.py standalone test")
    REF_VIDEO = "data/reference.mp4"
    DIST_VIDEO = "data/distorted_motion_real.mp4"
    aware, base = run_vqa_analysis(REF_VIDEO, DIST_VIDEO)
    if aware is not None:
        print(f"Motion-Aware Score: {aware:.4f} | Baseline: {base:.4f}")



