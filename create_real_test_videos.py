import cv2
import numpy as np
import os

# =========================================================================
# --- (A) 1. CONFIGURATION: Optimized for 1920x1080 "Bridge" video ---
# --- AGGRESSIVE DISTORTION: Using 4 boxes to "overpower" noise ---
# =========================================================================

# --- Input/Output Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

INPUT_VIDEO = os.path.join(DATA_DIR, "reference.mp4")
OUTPUT_STATIC = os.path.join(DATA_DIR, "distorted_static_real.mp4")
OUTPUT_MOTION = os.path.join(DATA_DIR, "distorted_motion_real.mp4")

# --- Artifact Properties ---
ARTIFACT_TYPE = "BOX" # A solid BOX is a stronger signal than PIXELATE
PIXELATE_BLOCK_SIZE = 32 
BOX_WIDTH = 150
BOX_HEIGHT = 100

# --- 2. COORDINATES: (Calculated for 1920x1080 bridge video) ---
# Format: [ (start_box), (end_box) ]

# --- "Moving" Test (Tracks 4 cars) ---
# We'll track 4 cars to create a massive error signal.
MOTION_PATHS = [
    # Car 1 (White car, mid-road)
    [(525, 900, 675, 1000), (775, 650, 925, 750)],
    # Car 2 (Fast car, low-road)
    [(350, 900, 500, 1000), (1750, 750, 1900, 850)],
    # Car 3 (Appears mid-screen)
    [(700, 800, 850, 900), (900, 675, 1050, 775)],
    # Car 4 (Another car, far-lane)
    [(900, 675, 1050, 775), (1200, 550, 1350, 650)]
]
# Time windows for each car
MOTION_FRAMES = [
    (30, 390), # Car 1 (long)
    (30, 210), # Car 2 (fast)
    (150, 360),# Car 3 (mid)
    (240, 420) # Car 4 (end)
]


# --- "Static" Test ---
# A large block on the building
STATIC_COORDS = (1230, 350, 1430, 550)
STATIC_FRAMES = (30, 420) # (Full duration)
# =========================================================================
# --- (B) SCRIPT LOGIC (Modified to handle multiple boxes) ---
# =========================================================================

def linear_interpolate(box_start, box_end, t):
    box_start = np.array(box_start)
    box_end = np.array(box_end)
    current_box = box_start + (box_end - box_start) * t
    return tuple(current_box.astype(int))

def apply_artifact(frame, box, artifact_type):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x1 >= x2 or y1 >= y2: return frame
    region = frame[y1:y2, x1:x2]
    if region.size == 0: return frame

    if artifact_type == "BOX":
        BOX_COLOR = (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, -1)
    elif artifact_type == "PIXELATE":
        h_reg, w_reg = region.shape[:2]
        block_w = max(1, w_reg // PIXELATE_BLOCK_SIZE)
        block_h = max(1, h_reg // PIXELATE_BLOCK_SIZE)
        small = cv2.resize(region, (block_w, block_h), interpolation=cv2.INTER_LINEAR)
        pixelated_region = cv2.resize(small, (w_reg, h_reg), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = pixelated_region
    return frame

def generate_distorted_video(input_path, output_path, is_motion_video):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print(f"Generating video: {os.path.basename(output_path)}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        if is_motion_video:
            # --- Apply MULTIPLE motion boxes ---
            for i, (start_f, end_f) in enumerate(MOTION_FRAMES):
                if start_f <= frame_count <= end_f:
                    total_frames = end_f - start_f
                    t = (frame_count - start_f) / total_frames if total_frames > 0 else 0
                    
                    start_box = MOTION_PATHS[i][0]
                    end_box = MOTION_PATHS[i][1]
                    current_box = linear_interpolate(start_box, end_box, t)
                    frame = apply_artifact(frame, current_box, ARTIFACT_TYPE)
        else:
            # --- Apply SINGLE static box ---
            start_f, end_f = STATIC_FRAMES
            if start_f <= frame_count <= end_f:
                frame = apply_artifact(frame, STATIC_COORDS, ARTIFACT_TYPE)
        
        out.write(frame)

    print(f"   ✓ Successfully saved: {output_path}")
    cap.release()
    out.release()

# =========================================================================
# --- (C) MAIN EXECUTION ---
# =========================================================================

if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO):
        print(f"Error: Input video not found at '{INPUT_VIDEO}'")
    else:
        print("--- Starting Video Distortion Generation (AGGRESSIVE) ---")
        generate_distorted_video(INPUT_VIDEO, OUTPUT_STATIC, is_motion_video=False)
        generate_distorted_video(INPUT_VIDEO, OUTPUT_MOTION, is_motion_video=True)
        print("\nAll videos generated successfully.")