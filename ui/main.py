import cv2
import os
import pandas as pd
import numpy as np
import joblib # For loading scaler
import xgboost as xgb
from ultralytics import YOLO
import time
import glob # Keep glob and re in case helper functions are needed later
import re

# --- Configuration ---
# --- !!! UPDATE THESE PATHS TO YOUR ORIGINAL VIDEO FILES !!! ---
VIDEO_PATHS = {
    "baseline": r"C:\Users\HRITHIK S\Tennis AI Director\data\raw\baseline\baseline_set1_video1_converted_h264_aac.mp4",
    "sideline": r"C:\Users\HRITHIK S\Tennis AI Director\data\raw\sideline\sideline_Set1_video1_converted_h264_aac.mp4", 
    "topcorner": r"C:\Users\HRITHIK S\Tennis AI Director\data\raw\corner top view\top_corner_set1_video1_converted_h264_aac.mp4"
}
ANGLES = ["baseline", "sideline", "topcorner"] # Ensure order matches VIDEO_PATHS keys

# --- Model, Scaler, Feature, YOLO Config ---
# --- !!! UPDATE THESE FILENAMES IF YOUR SAVED FILES ARE DIFFERENT !!! ---
MODEL_PATH = r"C:\Users\HRITHIK S\Tennis AI Director\models\tennis_director_xgb.json" # Or "tennis_director_xgb.json"
SCALER_PATH = r"C:\Users\HRITHIK S\Tennis AI Director\utils\feature_scaler.joblib"
YOLO_MODEL_PATH = r"C:\Users\HRITHIK S\Tennis AI Director\models\yolov8s.pt"              # Or yolov8n.pt for potentially faster speed
CONFIDENCE_THRESHOLD = 0.25                 # Confidence for YOLO detection

# --- !!! THIS MUST MATCH THE FEATURES THE MODEL WAS TRAINED ON !!! ---
# Example for Option B (Multi-view Coords):
FEATURE_COLS_USED_BY_MODEL = [
    'ba_p1_x', 'ba_p1_y', 'ba_p2_x', 'ba_p2_y',
    'si_p1_x', 'si_p1_y', 'si_p2_x', 'si_p2_y',
    'to_p1_x', 'to_p1_y', 'to_p2_x', 'to_p2_y',
]
# Example for Option A (Baseline + Relative + Motion - adapt if needed):
# FEATURE_COLS_USED_BY_MODEL = [
#     'p1_x', 'p1_y', 'p1_conf',
#     'p2_x', 'p2_y', 'p2_conf',
#     'p1_dist_net_est', 'p2_dist_net_est',
#     'delta_p1_x', 'delta_p1_y',
#     'delta_p2_x', 'delta_p2_y',
# ]

# --- UI Dimensions ---
THUMBNAIL_H = 180
THUMBNAIL_W = int(THUMBNAIL_H * 16 / 9)
MAIN_VIEW_H = 540
MAIN_VIEW_W = int(MAIN_VIEW_H * 16 / 9)
UI_BG_COLOR = (15, 15, 15) # Dark background
ANGLE_MAP = {0: "Baseline", 1: "Sideline", 2: "Top Corner"}
LABEL_MAP = {"Baseline": 0, "Sideline": 1, "Top Corner": 2} # Inverse map

# --- Feature Extraction Function ---
def extract_features_for_frames(frames_dict, yolo_model, expected_feature_order):
    """
    Runs YOLOv8 on necessary frames and extracts features in the specified order.
    Handles missing detections with -1.0.

    Args:
        frames_dict (dict): Dictionary like {'baseline': frame, 'sideline': frame, ...}
                            Frames can be None if read failed.
        yolo_model (YOLO): Initialized YOLO model.
        expected_feature_order (list): List of feature column names in the exact
                                       order the ML model expects.

    Returns:
        list: Feature values in the order specified by expected_feature_order,
              or None if critical error occurs.
    """
    features = {} # Store extracted features with prefixed names

    for angle in ANGLES: # Process each view required
        prefix = angle[:2]
        if angle == "topcorner": prefix = "to"

        # Initialize features for this angle with defaults
        p1_x, p1_y, p1_conf = -1.0, -1.0, -1.0
        p2_x, p2_y, p2_conf = -1.0, -1.0, -1.0
        ball_x, ball_y, ball_conf = -1.0, -1.0, -1.0

        frame = frames_dict.get(angle)

        if frame is not None:
            try:
                results = yolo_model(frame, verbose=False, classes=[0, 32], conf=CONFIDENCE_THRESHOLD)
                player_boxes = [] # [x1, y1, x2, y2, conf]
                ball_box = None   # [x1, y1, x2, y2, conf]

                if len(results) > 0:
                    boxes = results[0].boxes.cpu().numpy()
                    for box in boxes:
                        class_id = int(box.cls[0])
                        conf = box.conf[0]
                        coords = box.xyxy[0]
                        if class_id == 0: player_boxes.append(list(coords) + [conf])
                        elif class_id == 32:
                            if ball_box is None or conf > ball_box[4]: ball_box = list(coords) + [conf]

                player_boxes.sort(key=lambda b: (b[0] + b[2]) / 2) # Sort players by x-center

                # Extract features
                if len(player_boxes) > 0:
                    p1_box = player_boxes[0]; p1_x = (p1_box[0] + p1_box[2]) / 2; p1_y = (p1_box[1] + p1_box[3]) / 2; p1_conf = p1_box[4]
                if len(player_boxes) > 1:
                    p2_box = player_boxes[1]; p2_x = (p2_box[0] + p2_box[2]) / 2; p2_y = (p2_box[1] + p2_box[3]) / 2; p2_conf = p2_box[4]
                if ball_box is not None:
                    ball_x = (ball_box[0] + ball_box[2]) / 2; ball_y = (ball_box[1] + ball_box[3]) / 2; ball_conf = ball_box[4]

            except Exception as detect_e:
                 print(f"Error during YOLO detection for {angle}: {detect_e}")
                 # Keep features as -1.0

        # Store features with prefix
        features[f'{prefix}_p1_x'] = p1_x; features[f'{prefix}_p1_y'] = p1_y; features[f'{prefix}_p1_conf'] = p1_conf
        features[f'{prefix}_p2_x'] = p2_x; features[f'{prefix}_p2_y'] = p2_y; features[f'{prefix}_p2_conf'] = p2_conf
        features[f'{prefix}_ball_x'] = ball_x; features[f'{prefix}_ball_y'] = ball_y; features[f'{prefix}_ball_conf'] = ball_conf

    # --- Add calculated features IF the model expects them ---
    # Example: Relative Position (Only if included in FEATURE_COLS_USED_BY_MODEL)
    if 'p1_dist_net_est' in expected_feature_order or 'p2_dist_net_est' in expected_feature_order:
         FRAME_HEIGHT = 720 # Or get dynamically if needed
         estimated_net_y = FRAME_HEIGHT / 2
         ba_p1_y = features.get('ba_p1_y', -1.0)
         ba_p2_y = features.get('ba_p2_y', -1.0)
         features['p1_dist_net_est'] = abs(ba_p1_y - estimated_net_y) if ba_p1_y != -1.0 else -1.0
         features['p2_dist_net_est'] = abs(ba_p2_y - estimated_net_y) if ba_p2_y != -1.0 else -1.0

    # Example: Motion Features (More complex - requires storing previous features)
    # For simplicity, motion features are omitted here. They would typically be
    # calculated outside this function by comparing current features to previous ones.
    # If your model REQUIRES delta features, you'll need to implement that logic.

    # --- Return features in the exact order the model expects ---
    try:
        ordered_features = [features[col_name] for col_name in expected_feature_order]
        return ordered_features
    except KeyError as ke:
        print(f"Error: Feature extraction failed. Missing expected feature: {ke}")
        print(f"  Expected features: {expected_feature_order}")
        print(f"  Available features from extraction: {list(features.keys())}")
        return None # Indicate error

# --- Main Application ---
def run_director_video_ui():
    # --- Load Model and Scaler ---
    print("Loading model and scaler...")
    try:
        scaler = joblib.load(SCALER_PATH)
        xgb_model_loaded = xgb.XGBClassifier()
        xgb_model_loaded.load_model(MODEL_PATH)
        print("Model and scaler loaded successfully.")
    except FileNotFoundError as fnf_err: print(f"Error: {fnf_err}. Make sure model/scaler files are present."); return
    except Exception as e: print(f"Error loading model/scaler: {e}"); return

    # --- Load YOLO Model ---
    print("Loading YOLO model...")
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded successfully.")
    except Exception as e: print(f"Error loading YOLO model: {e}"); return

    # --- Open Video Captures ---
    print("Opening video files...")
    caps = {}
    fps_values = []
    for angle in ANGLES:
        path = VIDEO_PATHS.get(angle)
        if not path or not os.path.exists(path): print(f"Error: Video file not found for {angle}"); return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): print(f"Error: Could not open video file for {angle}"); return
        caps[angle] = cap
        fps = cap.get(cv2.CAP_PROP_FPS); fps_values.append(fps)
        print(f"  - {angle}: {fps:.2f} FPS")

    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 30
    wait_ms = max(1, int(1000 / avg_fps)) # Ensure wait time is at least 1ms
    print(f"Using wait time: {wait_ms} ms ({avg_fps:.2f} FPS avg)")

    # --- Processing Loop ---
    print("\nStarting UI loop... Press 'q' to quit.")
    frame_count = 0
    predicted_angle_name = "Baseline" # Start with default

    while True:
        start_loop_time = time.time()
        frames = {}
        all_read_ok = True

        # --- Read one frame from each video ---
        for angle in ANGLES:
            ret, frame = caps[angle].read()
            frames[angle] = frame # Store frame (or None if read failed)
            if not ret: all_read_ok = False; # Keep reading others even if one fails? Or break?

        # If any video ended or had read error, stop the loop
        if not all_read_ok: print("End of stream or read error reached."); break

        # --- Extract Features for current frames ---
        current_features = extract_features_for_frames(frames, yolo_model, FEATURE_COLS_USED_BY_MODEL)

        predicted_label = LABEL_MAP[predicted_angle_name] # Keep previous on error

        if current_features is not None:
            # --- Scale Features ---
            try:
                current_features_array = np.array(current_features).reshape(1, -1)
                current_features_scaled = scaler.transform(current_features_array)
            except Exception as e:
                print(f"Error scaling features at frame {frame_count}: {e}")
                current_features_scaled = None # Mark as failed

            # --- Predict Angle ---
            if current_features_scaled is not None:
                try:
                    predicted_label = xgb_model_loaded.predict(current_features_scaled)[0]
                except Exception as e:
                    print(f"Error predicting angle at frame {frame_count}: {e}")
                    # Keep previous angle on error

        # Map label to name for display
        predicted_angle_name = ANGLE_MAP.get(predicted_label, "Unknown")

        # --- Construct UI Display ---
        thumbs = []
        for angle in ANGLES:
            thumb_frame = frames.get(angle)
            if thumb_frame is None: thumb_frame = np.zeros((THUMBNAIL_H, THUMBNAIL_W, 3), dtype=np.uint8)
            try:
                thumb = cv2.resize(thumb_frame, (THUMBNAIL_W, THUMBNAIL_H), interpolation=cv2.INTER_AREA)
                cv2.putText(thumb, angle.upper(), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # Highlight selected thumbnail
                if angle == predicted_angle_name.lower().replace(" ", ""):
                     cv2.rectangle(thumb, (0, 0), (THUMBNAIL_W - 1, THUMBNAIL_H - 1), (0, 255, 255), 3)
                thumbs.append(thumb)
            except Exception as resize_e:
                 print(f"Error resizing thumb {angle}: {resize_e}")
                 thumbs.append(np.zeros((THUMBNAIL_H, THUMBNAIL_W, 3), dtype=np.uint8)) # Add placeholder

        # Ensure we always have 3 thumbnails (even if placeholders)
        while len(thumbs) < 3: thumbs.append(np.zeros((THUMBNAIL_H, THUMBNAIL_W, 3), dtype=np.uint8))

        top_row_display = np.hstack(thumbs)

        # Get main frame based on prediction
        main_frame_key = predicted_angle_name.lower().replace(" ","") # topcorner, baseline, sideline
        main_frame = frames.get(main_frame_key)
        if main_frame is None: main_frame = frames.get("baseline", np.zeros((MAIN_VIEW_H, MAIN_VIEW_W, 3), dtype=np.uint8))

        try:
            main_view_display = cv2.resize(main_frame, (MAIN_VIEW_W, MAIN_VIEW_H), interpolation=cv2.INTER_AREA)
        except Exception as resize_e:
            print(f"Error resizing main view {main_frame_key}: {resize_e}")
            main_view_display = np.zeros((MAIN_VIEW_H, MAIN_VIEW_W, 3), dtype=np.uint8)


        # Combine UI elements
        total_ui_width = top_row_display.shape[1]
        top_row_h = top_row_display.shape[0]
        main_view_h = main_view_display.shape[0]
        padding = 15
        text_area_h = 40
        ui_canvas = np.full((top_row_h + main_view_h + padding * 2 + text_area_h, total_ui_width, 3), UI_BG_COLOR, dtype=np.uint8)

        x_offset_top = (total_ui_width - top_row_display.shape[1]) // 2
        ui_canvas[padding:padding + top_row_h, x_offset_top:x_offset_top + top_row_display.shape[1]] = top_row_display

        x_offset_main = (total_ui_width - main_view_display.shape[1]) // 2
        main_view_y_start = padding + top_row_h + padding
        ui_canvas[main_view_y_start : main_view_y_start + main_view_h, x_offset_main : x_offset_main + main_view_display.shape[1]] = main_view_display

        # Status Text
        status_text = f"Frame: {frame_count}    Predicted Angle: {predicted_angle_name}"
        text_y = main_view_y_start + main_view_h + 30
        cv2.putText(ui_canvas, status_text, (10, text_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        # --- Show Window ---
        cv2.imshow("Tennis AI Director", ui_canvas)

        frame_count += 1

        # --- Handle Loop/Exit ---
        key = cv2.waitKey(wait_ms) & 0xFF # Use calculated wait time
        if key == ord('q'):
            print("Quit key pressed. Exiting.")
            break

    # --- Cleanup ---
    print("Releasing video captures...")
    for angle in ANGLES:
        if caps[angle].isOpened():
            caps[angle].release()
    cv2.destroyAllWindows()
    print("UI Closed.")

# --- Run Application ---
if __name__ == "__main__":
    run_director_video_ui()