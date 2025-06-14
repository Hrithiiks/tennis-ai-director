import cv2
import os
import pandas as pd
import numpy as np
import glob
import re

# --- Configuration ---
# Directory containing baseline/, sideline/, topcorner/ subfolders with synced frames
BASE_FRAMES_DIR = r"C:\Users\HRITHIK S\Tennis AI Director\data\synced_frames"
# CSV file to store annotations (will be created if it doesn't exist)
ANNOTATION_CSV = r"C:\Users\HRITHIK S\Tennis AI Director\data\manual_annotations.csv"
# List of camera angles (must match subfolder names)
ANGLES = ["baseline", "sideline", "topcorner"]
# Keyboard keys mapping to angle names for annotation input
KEYS_TO_ANGLES = {'b': 'Baseline', 's': 'Sideline', 't': 'Top Corner'}

# --- Helper Functions ---

def get_timestamps_from_dir(base_dir, angles):
    """Finds all unique timestamps available across angle directories."""
    timestamps = set()
    # Regex to extract timestamp like _t123.456.jpg from filename
    pattern = re.compile(r"_t(\d+\.\d{3})\.jpg$")
    for angle in angles:
        angle_dir = os.path.join(base_dir, angle)
        if os.path.isdir(angle_dir):
            try:
                # Use glob to find files matching the expected pattern
                for filepath in glob.glob(os.path.join(angle_dir, f"{angle}_frame_*.jpg")):
                    match = pattern.search(os.path.basename(filepath))
                    if match:
                        try:
                            timestamps.add(float(match.group(1)))
                        except ValueError:
                            print(f"Warning: Could not parse timestamp from {filepath}")
            except Exception as e:
                 print(f"Error scanning directory {angle_dir}: {e}")
    if not timestamps:
        print(f"Warning: No timestamps found matching pattern '_t###.###.jpg' in subfolders of {base_dir}")
    return sorted(list(timestamps))

def get_frame_path(base_dir, angle, timestamp):
    """Constructs the expected frame path based on timestamp."""
    # Format timestamp consistently to match filenames (e.g., 3 decimal places)
    timestamp_str = f"{timestamp:.3f}"
    # Use glob to find the specific frame file for this timestamp
    search_pattern = os.path.join(base_dir, angle, f"{angle}_frame_*_t{timestamp_str}.jpg")
    matching_files = glob.glob(search_pattern)
    if matching_files:
        # If multiple frames somehow match (unlikely), sort and take the first
        return sorted(matching_files)[0]
    else:
        # print(f"Debug: Frame file not found for {angle} at timestamp {timestamp_str} using pattern {search_pattern}")
        return None # Indicate file not found

# --- Main Annotation Function ---
def run_annotation():
    """Loads frames, displays them in a grid, gets user input, and saves annotations."""
    # --- Load existing annotations ---
    annotated_timestamps = set()
    df_existing = pd.DataFrame(columns=['timestamp', 'selected_camera_angle', 'notes']) # Initialize empty
    try:
        if os.path.exists(ANNOTATION_CSV):
             if os.path.getsize(ANNOTATION_CSV) > 0:
                 df_existing = pd.read_csv(ANNOTATION_CSV)
                 if 'timestamp' in df_existing.columns:
                      # Convert to numeric, coercing errors, then keep only valid floats
                      df_existing['timestamp'] = pd.to_numeric(df_existing['timestamp'], errors='coerce')
                      df_existing.dropna(subset=['timestamp'], inplace=True)
                      # Convert safely to float after handling potential NaNs
                      if not df_existing.empty:
                           annotated_timestamps = set(df_existing['timestamp'].astype(float))
                 print(f"Loaded {len(annotated_timestamps)} existing valid annotations.")
             else:
                  print(f"{ANNOTATION_CSV} is empty. Starting fresh.")
                  # Ensure header exists if file is empty
                  if not (os.path.exists(ANNOTATION_CSV) and pd.read_csv(ANNOTATION_CSV, nrows=0).shape[1] > 0):
                     df_existing.to_csv(ANNOTATION_CSV, index=False)


        else:
             print(f"{ANNOTATION_CSV} not found. Creating a new one.")
             # Create file with header
             df_existing.to_csv(ANNOTATION_CSV, index=False)

    except Exception as e:
        print(f"Error loading or processing {ANNOTATION_CSV}: {e}. Check file format.")
        print("Will attempt to proceed, but existing annotations might be lost/overwritten.")
        # Re-create file with header if loading failed badly
        df_existing = pd.DataFrame(columns=['timestamp', 'selected_camera_angle', 'notes'])
        try:
            df_existing.to_csv(ANNOTATION_CSV, index=False)
        except Exception as write_e:
             print(f"Failed to even create/overwrite {ANNOTATION_CSV}: {write_e}")


    # --- Get all available timestamps from frame files ---
    all_timestamps = get_timestamps_from_dir(BASE_FRAMES_DIR, ANGLES)
    if not all_timestamps:
        print(f"Error: No frames found matching pattern in {BASE_FRAMES_DIR}. Cannot proceed.")
        return

    # --- Determine timestamps still needing annotation ---
    annotated_timestamps_float = {float(ts) for ts in annotated_timestamps}
    timestamps_to_annotate = [ts for ts in all_timestamps if float(ts) not in annotated_timestamps_float]
    total_to_annotate = len(timestamps_to_annotate)
    print(f"Found {len(all_timestamps)} total timestamps in frames. {total_to_annotate} remaining to annotate.")

    if total_to_annotate == 0:
        if len(annotated_timestamps) > 0: print("All timestamps seem to be annotated already!")
        else: print("No timestamps found to annotate and no existing annotations.")
        return


    # --- Annotation Loop ---
    new_annotations = [] # Store annotations for this session
    active_window_title = None # Track the current window title

    try:
        for i, ts in enumerate(timestamps_to_annotate):
            print(f"\n--- Annotating Timestamp: {ts:.3f} ({i+1}/{total_to_annotate}) ---")

            # --- Load Frames for the current timestamp ---
            frames = {}
            valid_frames_found_count = 0
            for angle in ANGLES:
                f_path = get_frame_path(BASE_FRAMES_DIR, angle, ts)
                frame = None
                if f_path and os.path.exists(f_path):
                    frame = cv2.imread(f_path)
                if frame is not None:
                    frames[angle] = frame
                    valid_frames_found_count += 1
                else:
                    print(f"Warning: Frame missing or unreadable for {angle} at ts {ts:.3f}")
                    frames[angle] = None # Keep track of missing frames

            if valid_frames_found_count == 0:
                 print(f"Skipping timestamp {ts:.3f} due to all frames missing.")
                 continue

            # --- Prepare Frames for Grid Display ---
            display_h = 360 # Target height for each frame in the grid
            display_w = int(display_h * 16 / 9) # Calculate width assuming 16:9 aspect ratio

            resized_dict = {}
            for angle in ANGLES:
                 frame = frames.get(angle) # Get frame, might be None
                 angle_name = angle.upper()
                 if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                     try:
                          resized = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)
                     except Exception as resize_e:
                          print(f"Error resizing frame for {angle}: {resize_e}")
                          resized = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                          cv2.putText(resized, f"{angle_name} RESIZE ERR", (10, display_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                 else:
                     resized = np.zeros((display_h, display_w, 3), dtype=np.uint8)
                     cv2.putText(resized, f"{angle_name} MISSING", (10, display_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

                 cv2.putText(resized, angle_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                 resized_dict[angle] = resized

            # --- Create Grid Layout ---
            baseline_f = resized_dict.get("baseline", np.zeros((display_h, display_w, 3), dtype=np.uint8))
            sideline_f = resized_dict.get("sideline", np.zeros((display_h, display_w, 3), dtype=np.uint8))
            topcorner_f = resized_dict.get("topcorner", np.zeros((display_h, display_w, 3), dtype=np.uint8))
            placeholder = np.zeros((display_h, display_w, 3), dtype=np.uint8)
            cv2.putText(placeholder, "EMPTY", (display_w//2 - 40, display_h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

            top_row = np.hstack((baseline_f, sideline_f))
            bottom_row = np.hstack((topcorner_f, placeholder))
            combined_display = np.vstack((top_row, bottom_row))

            # --- Display Combined Grid Image ---
            active_window_title = f"Timestamp: {ts:.3f} - Choose: [B]aseline/[S]ideline/[T]op Corner | [N]otes | [Q]uit"
            cv2.imshow(active_window_title, combined_display)

            # --- Get User Input ---
            choice = None
            notes = ""
            while choice is None:
                key = cv2.waitKey(100) & 0xFF
                window_closed = False
                if active_window_title:
                    try:
                         if cv2.getWindowProperty(active_window_title, cv2.WND_PROP_VISIBLE) < 1: window_closed = True
                    except Exception: pass

                if window_closed:
                      print("Window closed by user. Quitting session.")
                      choice = 'quit'
                      break

                if key == 255: continue

                key_char = chr(key).lower()

                if key_char in KEYS_TO_ANGLES:
                    choice = KEYS_TO_ANGLES[key_char]
                    print(f"Selected: {choice}")
                    break
                elif key_char == 'n':
                    print("Closing window temporarily to enter notes in terminal...")
                    try:
                        if active_window_title: cv2.destroyWindow(active_window_title)
                        active_window_title = None
                    except Exception: pass
                    notes = input(f"Enter notes for timestamp {ts:.3f} (press Enter when done): ").strip()
                    print(f"Notes added: '{notes}'")
                    print("Now please select the angle by pressing 'b', 's', or 't' in the terminal.")
                    while True:
                         angle_key = input("Enter angle choice (b/s/t) or 'q' to quit: ").lower().strip()
                         if angle_key in KEYS_TO_ANGLES:
                             choice = KEYS_TO_ANGLES[angle_key]
                             print(f"Selected: {choice}")
                             break
                         elif angle_key == 'q':
                              choice = 'quit'
                              print("Quit selected.")
                              break
                         else:
                             print("Invalid key. Use 'b', 's', or 't'.")
                    break
                elif key_char == 'q':
                    print("Quit selected.")
                    choice = 'quit'
                    break
                else:
                     pass

            if active_window_title:
                try: cv2.destroyWindow(active_window_title)
                except Exception: pass
                active_window_title = None

            if choice == 'quit':
                print("Exiting annotation loop.")
                break

            if choice and choice != 'quit':
                new_annotations.append({'timestamp': ts, 'selected_camera_angle': choice, 'notes': notes})
                print(f"Annotation for {ts:.3f} recorded.")

    except KeyboardInterrupt:
        print("\nAnnotation interrupted by user (Ctrl+C). Saving progress...")
    finally:
        cv2.destroyAllWindows()
        if new_annotations:
            print(f"\nAttempting to save {len(new_annotations)} new annotations...")
            df_new = pd.DataFrame(new_annotations)
            try:
                file_exists_and_has_content = (os.path.exists(ANNOTATION_CSV) and os.path.getsize(ANNOTATION_CSV) > 0)
                df_new.to_csv(ANNOTATION_CSV, mode='a', header=not file_exists_and_has_content, index=False)
                print(f"Successfully saved/appended annotations to {ANNOTATION_CSV}")
            except Exception as save_e:
                print(f"!!!!!!!! ERROR SAVING ANNOTATIONS to {ANNOTATION_CSV}: {save_e} !!!!!!!!")
                print("Attempting to print unsaved annotations below:")
                try: print(df_new.to_string())
                except Exception as print_e: print(f"Could not print annotations: {print_e}")
        else:
            print("\nNo new annotations were made or recorded in this session.")

if __name__ == "__main__":
    run_annotation()