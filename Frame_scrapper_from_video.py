import cv2
import os

# Set your paths
input_video = "data\\data.MOV"
output_folder = "data\\frames\\"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(input_video)

frame_count = 0
success = True

while success:
    # Read next frame
    success, frame = cap.read()
    
    if not success:
        break  # Exit loop if we can't read more frames
    
    # Save frame as PNG file
    output_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
    cv2.imwrite(output_path, frame)
    
    frame_count += 1
    
    # Print progress every 100 frames
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

# Release resources
cap.release()
print(f"Finished! Extracted {frame_count} frames to {output_folder}")