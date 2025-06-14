import cv2
import os
import math # Using math is optional, simple comparison often works

def extract_frames_synced(video_path, output_folder, video_source, frame_interval_sec=0.5):
    """
    Extracts frames from a video at specified time intervals using embedded
    timestamps for better synchronization.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the directory where frames will be saved within the base output folder.
        video_source (str): Name of the source video (e.g., "baseline", "sideline").
        frame_interval_sec (float): Interval (in seconds) between extracted frames.
    """
    # Ensure the specific subdirectory for this source exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    frame_number = 0
    # Initialize last_saved_time_msec slightly negative to ensure the first frame near 0 is captured
    last_saved_time_msec = -1.0
    frame_interval_msec = frame_interval_sec * 1000.0

    # Determine the timestamp of the very first frame
    first_frame_time_msec = 0.0
    # Some videos might not allow seeking right to the beginning before reading
    # Try reading the first frame to get its timestamp
    ret, _ = video_capture.read()
    if ret:
        first_frame_time_msec = video_capture.get(cv2.CAP_PROP_POS_MSEC)
        # Reset capture to the beginning
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
         print(f"Warning: Could not read the first frame to determine start time for {video_path}")
         # Proceed assuming start time is 0, but sync might be less precise

    # Adjust last_saved_time to be just before the first desired save point
    # relative to the actual start time of the video stream
    last_saved_time_msec = first_frame_time_msec - frame_interval_msec

    print(f"Starting extraction for {video_source}. First frame time: {first_frame_time_msec:.0f} ms. Target interval: {frame_interval_msec:.0f} ms.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break # End of video

        # Get the timestamp of the CURRENT frame that was just read
        current_time_msec = video_capture.get(cv2.CAP_PROP_POS_MSEC)

        # Check if the current frame's timestamp is at or after the next desired save point
        # Adding a small tolerance (e.g., half frame duration, or just 1ms) can help
        # Let's check if it's crossed the boundary: current_time >= target_time
        next_target_time_msec = last_saved_time_msec + frame_interval_msec

        if current_time_msec >= next_target_time_msec - 1: # Allow 1ms tolerance

            timestamp_sec = current_time_msec / 1000.0 # Convert to seconds for filename

            # Save the frame
            output_path = os.path.join(
                output_folder,
                # Use the actual timestamp in the filename
                f"{video_source}_frame_{frame_number:06d}_t{timestamp_sec:.3f}.jpg"
            )
            cv2.imwrite(output_path, frame)
            # print(f"Saved frame: {os.path.basename(output_path)} at {current_time_msec:.0f} ms (Target: ~{next_target_time_msec:.0f} ms)")

            # Update the time of the last saved frame to the timestamp of the frame we JUST saved
            last_saved_time_msec = current_time_msec
            frame_number += 1


    video_capture.release()
    print(f"Finished extraction for {video_source}. Saved {frame_number} frames.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Define video paths (ensure these are correct)
    baseline_video_path = r"C:\Users\HRITHIK S\TENNIS AI DIRECTOR\Recorded Videos\baseline\baseline_set1_video1_converted_h264_aac.mp4"
    sideline_video_path = r"C:\Users\HRITHIK S\TENNIS AI DIRECTOR\Recorded Videos\sideline\sideline_Set1_video1_converted_h264_aac.mp4"
    topcorner_video_path = r"C:\Users\HRITHIK S\TENNIS AI DIRECTOR\Recorded Videos\corner top view\top_corner_set1_video1_converted_h264_aac.mp4"

    # --- IMPORTANT: Use a new base output folder for the synced frames ---
    base_output_folder = "extracted_frames_synced"
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)

    # Define the desired frame interval in seconds
    frame_interval_seconds = 0.5

    # Run the extraction for each video
    extract_frames_synced(
        baseline_video_path,
        os.path.join(base_output_folder, "baseline"), # Subfolder for baseline
        "baseline",
        frame_interval_sec=frame_interval_seconds
    )
    extract_frames_synced(
        sideline_video_path,
        os.path.join(base_output_folder, "sideline"), # Subfolder for sideline
        "sideline",
        frame_interval_sec=frame_interval_seconds
    )
    extract_frames_synced(
        topcorner_video_path,
        os.path.join(base_output_folder, "topcorner"), # Subfolder for topcorner
        "topcorner",
        frame_interval_sec=frame_interval_seconds
    )

    print("-" * 30)
    print(f"Synced frame extraction complete! Frames saved in: {base_output_folder}")
    print("-" * 30)