# feature_extraction_multi.py
import cv2
import os
import pandas as pd
import numpy as np
import glob
import re
from ultralytics import YOLO
import time # Added for timing

# --- Configuration ---
ANNOTATION_CSV = r"C:\Users\HRITHIK S\Tennis AI Director\data\manual_annotations.csv" # Needed to get timestamps
BASE_FRAMES_DIR = r"C:\Users\HRITHIK S\Tennis AI Director\data\synced_frames" # Base dir with baseline/, sideline/, topcorner/
# --- NEW output file for combined features from all views ---
OUTPUT_FEATURES_MULTI_CSV = "extracted_features_multi_view.csv"
ANGLES = ["baseline", "sideline", "topcorner"] # Order matters for consistency if needed
YOLO_MODEL_PATH = 'yolov8s.pt' # Choose your model (s, m, l, x)
CONFIDENCE_THRESHOLD = 0.25 # Optional: Minimum confidence for detection
LAST_SIDELINE_TIMESTAMP = 640.0 # Timestamp after which sideline frames stop

# --- Helper Functions ---
def get_timestamps_from_dir(base_dir, angles):
    """Finds all unique timestamps available across any angle directory."""
    timestamps = set()
    pattern = re.compile(r"_t(\d+\.\d{3})\.jpg$") # Expecting 3 decimal places
    found_files = False
    for angle in angles:
        angle_dir = os.path.join(base_dir, angle)
        if os.path.isdir(angle_dir):
            try:
                for filepath in glob.glob(os.path.join(angle_dir, f"{angle}_frame_*.jpg")):
                    found_files = True # Mark that we found at least one file
                    match = pattern.search(os.path.basename(filepath))
                    if match:
                        try:
                            timestamps.add(float(match.group(1)))
                        except ValueError:
                            print(f"Warning: Could not parse timestamp from {filepath}")
            except Exception as e:
                 print(f"Error scanning directory {angle_dir}: {e}")
    if not found_files:
         print(f"ERROR: No JPG frame files found in any subdirectories of {base_dir}")
    elif not timestamps:
        print(f"Warning: No timestamps found matching pattern '_t###.###.jpg' in subfolders of {base_dir}")
    return sorted(list(timestamps))

def get_frame_path(base_dir, angle, timestamp):
    """Finds the specific frame file path for a given angle and timestamp."""
    # Format timestamp consistently to match filenames (e.g., 3 decimal places)
    timestamp_str = f"{timestamp:.3f}"
    # Use glob to find the specific frame file for this timestamp
    search_pattern = os.path.join(base_dir, angle, f"{angle}_frame_*_t{timestamp_str}.jpg")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        # If multiple frames somehow match (unlikely), sort and take the first
        return sorted(matching_files)[0]
    else:
        return None # Indicate file not found

# --- Main Feature Extraction Function ---
def extract_multi_view_features():
    """Extracts features from all three views for each timestamp."""
    start_time_total = time.time()
    # --- Load Timestamps ---
    try:
        # Try loading annotations CSV first to get the exact list of needed timestamps
        if os.path.exists(ANNOTATION_CSV):
             annotations_df = pd.read_csv(ANNOTATION_CSV)
             if 'timestamp' in annotations_df.columns:
                 annotations_df['timestamp'] = pd.to_numeric(annotations_df['timestamp'], errors='coerce')
                 annotations_df.dropna(subset=['timestamp'], inplace=True)
                 timestamps_to_process = sorted(annotations_df['timestamp'].unique())
                 if not timestamps_to_process.any(): # Check if list is empty after dropna
                      raise ValueError("No valid timestamps found in annotation file after cleaning.")
                 print(f"Loaded {len(timestamps_to_process)} unique timestamps from {ANNOTATION_CSV}.")
             else:
                 raise ValueError("'timestamp' column not found.")
        else:
            raise FileNotFoundError # Trigger fallback if file doesn't exist

    except Exception as e:
        # Fallback: Get timestamps from directories if annotation file fails
        print(f"Warning/Error loading timestamps from {ANNOTATION_CSV}: {e}. Getting timestamps from frame directories instead.")
        timestamps_to_process = get_timestamps_from_dir(BASE_FRAMES_DIR, ANGLES)
        if not timestamps_to_process:
             print(f"Error: No valid timestamps found in frame directories either. Cannot proceed.")
             return
        print(f"Found {len(timestamps_to_process)} unique timestamps from directories.")


    # --- Initialize YOLO Model ---
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLOv8 model ('{YOLO_MODEL_PATH}') loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model '{YOLO_MODEL_PATH}': {e}")
        print("Ensure 'ultralytics' is installed and model path is correct.")
        return

    # --- Feature Extraction Loop ---
    all_features_multi = []
    total_count = len(timestamps_to_process)
    print(f"\nStarting multi-view feature extraction for {total_count} timestamps...")

    for i, ts in enumerate(timestamps_to_process):
        loop_start_time = time.time()
        # Limit processing based on known end time of shortest video
        if ts > LAST_SIDELINE_TIMESTAMP:
            print(f"Timestamp {ts:.3f} is after shortest video end time ({LAST_SIDELINE_TIMESTAMP:.3f}). Stopping extraction.")
            break

        # Dictionary to hold features for THIS timestamp from ALL angles
        timestamp_features = {'timestamp': ts}

        # Loop through each camera angle for the current timestamp
        for angle in ANGLES:
            # Use consistent 2-char prefixes: ba, si, to
            prefix = angle[:2]
            if angle == "topcorner": prefix = "to" # Handle specific case

            frame_path = get_frame_path(BASE_FRAMES_DIR, angle, ts)
            frame = None
            if frame_path and os.path.exists(frame_path):
                frame = cv2.imread(frame_path)

            # Initialize features for this angle with defaults (-1.0 for coords/conf)
            p1_x, p1_y, p1_conf = -1.0, -1.0, -1.0
            p2_x, p2_y, p2_conf = -1.0, -1.0, -1.0
            ball_x, ball_y, ball_conf = -1.0, -1.0, -1.0

            if frame is not None:
                # Run YOLOv8 inference
                try:
                    results = model(frame, verbose=False, classes=[0, 32], conf=CONFIDENCE_THRESHOLD) # Person and sports ball

                    # Process results for this frame
                    player_boxes = [] # [x1, y1, x2, y2, conf]
                    ball_box = None   # [x1, y1, x2, y2, conf]

                    if len(results) > 0:
                        boxes = results[0].boxes.cpu().numpy()
                        for box in boxes:
                            class_id = int(box.cls[0])
                            conf = box.conf[0]
                            coords = box.xyxy[0]
                            if class_id == 0: # Person
                                player_boxes.append(list(coords) + [conf])
                            elif class_id == 32: # Sports ball
                                if ball_box is None or conf > ball_box[4]:
                                    ball_box = list(coords) + [conf]

                    player_boxes.sort(key=lambda b: (b[0] + b[2]) / 2) # Sort by x-center

                    # Extract features for this angle
                    if len(player_boxes) > 0:
                        p1_box = player_boxes[0]
                        p1_x = (p1_box[0] + p1_box[2]) / 2; p1_y = (p1_box[1] + p1_box[3]) / 2; p1_conf = p1_box[4]
                    if len(player_boxes) > 1:
                        p2_box = player_boxes[1]
                        p2_x = (p2_box[0] + p2_box[2]) / 2; p2_y = (p2_box[1] + p2_box[3]) / 2; p2_conf = p2_box[4]
                    if ball_box is not None:
                        ball_x = (ball_box[0] + ball_box[2]) / 2; ball_y = (ball_box[1] + ball_box[3]) / 2; ball_conf = ball_box[4]

                except Exception as detect_e:
                     print(f"Error during detection for {angle} at ts {ts:.3f}: {detect_e}")
            else:
                 # Frame was missing or unreadable, features remain -1.0
                 # get_frame_path warning was likely printed already if path was None
                 pass

            # Add features for this angle to the timestamp dictionary with prefix
            timestamp_features[f'{prefix}_p1_x'] = p1_x; timestamp_features[f'{prefix}_p1_y'] = p1_y; timestamp_features[f'{prefix}_p1_conf'] = p1_conf
            timestamp_features[f'{prefix}_p2_x'] = p2_x; timestamp_features[f'{prefix}_p2_y'] = p2_y; timestamp_features[f'{prefix}_p2_conf'] = p2_conf
            timestamp_features[f'{prefix}_ball_x'] = ball_x; timestamp_features[f'{prefix}_ball_y'] = ball_y; timestamp_features[f'{prefix}_ball_conf'] = ball_conf

        # Append the combined features for this timestamp to the main list
        all_features_multi.append(timestamp_features)

        # Print progress and estimated time remaining
        loop_end_time = time.time()
        elapsed_loop = loop_end_time - loop_start_time
        elapsed_total = loop_end_time - start_time_total
        avg_time_per_ts = elapsed_total / (i + 1) if i > 0 else elapsed_loop
        remaining_ts = total_count - (i + 1)
        eta_seconds = remaining_ts * avg_time_per_ts
        eta_minutes = eta_seconds / 60
        if (i+1) % 10 == 0 or i == 0: # Print every 10 timestamps and first one
             print(f"Processed timestamp {ts:.3f} [{i+1}/{total_count}] in {elapsed_loop:.2f}s. ETA: {eta_minutes:.1f} mins.")


    print(f"\nFinished feature extraction. Processed {len(all_features_multi)} timestamps.")

    # --- Save Multi-View Features ---
    if all_features_multi:
        features_multi_df = pd.DataFrame(all_features_multi)
        print("\nMulti-View Extracted Features (first 5 rows):")
        print(features_multi_df.head())
        # Check if DataFrame is not empty before info()
        if not features_multi_df.empty:
            print("\nMulti-View DataFrame Info:")
            features_multi_df.info()
        else:
            print("\nWarning: Feature DataFrame is empty after processing.")

        try:
            features_multi_df.to_csv(OUTPUT_FEATURES_MULTI_CSV, index=False)
            print(f"\nSuccessfully saved multi-view features to {OUTPUT_FEATURES_MULTI_CSV}")
        except Exception as e:
            print(f"\nError saving features to {OUTPUT_FEATURES_MULTI_CSV}: {e}")
    else:
        print("No multi-view features were extracted to save.")

# --- Run Extraction ---
if __name__ == "__main__":
    extract_multi_view_features()