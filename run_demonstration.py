import os
import main_vqa_pipeline # Import your modified main script

# -----------------------------------------------------------------
# 1. DEFINE YOUR VIDEO PATHS
# -----------------------------------------------------------------
DATA_DIR = "data"
REF_VIDEO = os.path.join(DATA_DIR, "reference.mp4")
DIST_STATIC_VIDEO = os.path.join(DATA_DIR, "distorted_static.mp4")
DIST_MOTION_VIDEO = os.path.join(DATA_DIR, "distorted_motion.mp4")

def run_experiment():
    """
    Runs the full qualitative demonstration.
    """
    print("=========================================================")
    print("FCV PROJECT: MOTION-AWARE VQA DEMONSTRATION")
    print("=========================================================\n")
    print(f"Reference Video: {REF_VIDEO}")
    print(f"Distorted (Static): {DIST_STATIC_VIDEO}")
    print(f"Distorted (Motion): {DIST_MOTION_VIDEO}\n")

    # -----------------------------------------------------------------
    # 2. RUN ANALYSIS
    # -----------------------------------------------------------------
    print("--- Running Analysis ---")
    
    # --- Run on STATIC video ---
    # This calls the function in your main_vqa_pipeline.py
    aware_static, baseline_static = main_vqa_pipeline.run_vqa_analysis(
        REF_VIDEO, DIST_STATIC_VIDEO
    )
    
    # --- Run on MOTION video ---
    # This calls the function in your main_vqa_pipeline.py again
    aware_motion, baseline_motion = main_vqa_pipeline.run_vqa_analysis(
        REF_VIDEO, DIST_MOTION_VIDEO
    )

    if aware_static is None or aware_motion is None:
        print("✗ Error during processing. Check video paths.")
        return

    # -----------------------------------------------------------------
    # 3. FINAL ANALYSIS & RESULTS
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL ANALYSIS")
    print("=" * 70)
    
    print("\n--- Baseline (Motion-Blind) Model Results ---")
    print(f"  Score for Static Artifact: {baseline_static:.4f}")
    print(f"  Score for Motion Artifact: {baseline_motion:.4f}")
    print("  Hypothesis: Scores should be almost identical.")
    if abs(baseline_static - baseline_motion) < 0.01:
        print("  ✓ Result: CONFIRMED. The baseline model is motion-blind.\n")
    else:
        print("  ✗ Result: FAILED. Baseline scores are different. Check your videos.\n")

    print("--- YOUR Motion-Aware Model Results ---")
    print(f"  Score for Static Artifact: {aware_static:.4f}")
    print(f"  Score for Motion Artifact: {aware_motion:.4f}")
    print("  Hypothesis: Motion score should be SIGNIFICANTLY LOWER (worse).")
    if aware_motion < aware_static - 0.01:
        print("  ✓ Result: SUCCESS! Your model correctly penalized the motion artifact!\n")
    else:
        print("  ✗ Result: FAILED. Model did not penalize motion. Check Stage B/C logic.\n")
    
    print("=" * 70)

if __name__ == "__main__":
    # First, check if all 3 videos exist
    if not all(os.path.exists(p) for p in [REF_VIDEO, DIST_STATIC_VIDEO, DIST_MOTION_VIDEO]):
        print("✗ ERROR: Missing video files in 'data/' folder.")
        print(f"  Make sure these exist:")
        print(f"  - {REF_VIDEO}")
        print(f"  - {DIST_STATIC_VIDEO}")
        print(f"  - {DIST_MOTION_VIDEO}")
        print("\n  Please run the 'create_test_videos.py' script first.")
    else:
        run_experiment()