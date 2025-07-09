import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp

# Load YOLOv8 model (for detecting person)
model = YOLO("yolov8n.pt")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Define facial landmark indices for required parts
EYES_IDX = [33, 263]  # Left and Right eyes
NOSE_IDX = [1]        # Nose tip
MOUTH_IDX = [13, 14]  # Upper and lower lips

def detect_facial_features(image_path):
    """Detects eyes, nose, and mouth using MediaPipe Face Mesh."""
    img = cv2.imread(image_path)
    if img is None:
        return False

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return False

    face_landmarks = results.multi_face_landmarks[0].landmark

    try:
        # Check if key facial features are detected
        eyes = all(face_landmarks[idx] for idx in EYES_IDX)
        nose = all(face_landmarks[idx] for idx in NOSE_IDX)
        mouth = all(face_landmarks[idx] for idx in MOUTH_IDX)
        return eyes and nose and mouth
    except:
        return False

def detect_person(image_path, confidence_threshold=0.5):
    """Detects person using YOLOv8 (class 0)."""
    results = model.predict(image_path, classes=[0], verbose=False)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        if len(boxes) > 0 and np.max(confidences) > confidence_threshold:
            return True
    return False

def filter_images(input_folder="data/frames", output_folder="filtered_faces"):
    """Filters images with a person and facial features, saves them."""
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f" Processing: {filename}")

            if detect_person(image_path):
                if detect_facial_features(image_path):
                    img = Image.open(image_path)
                    img.save(os.path.join(output_folder, filename))
                    print(f"   Saved: {filename}")
                else:
                    print(f"  Skipped: {filename} (facial features not found)")
            else:
                print(f"  Skipped: {filename} (no person detected)")

if __name__ == "__main__":
    filter_images()
    print("\nDone! Check the 'filtered_faces' folder.")
