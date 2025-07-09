

## 🧠 Person Face Image Filter from Video Frames

This project allows you to:

1. **Extract frames from a video**.
2. **Detect persons in those frames** using YOLOv8.
3. **Filter frames where a face has both eyes, nose, and mouth** using MediaPipe FaceMesh.

---

### 📁 Folder Structure

```
├── Frame_scrapper_from_video.py   # Extracts frames from a video
├── Image_filter.py                # Filters frames by detecting face with key features
├── test.py                        # For testing purposes (optional)
├── yolov8n.pt                     # YOLOv8 model weights (nano version)
├── .gitignore                     # Git ignore file
```

---

### 🚀 How It Works

#### 1. `Frame_scrapper_from_video.py`

Extracts every frame from a video and saves them to a `data/frames/` folder.

```bash
python Frame_scrapper_from_video.py
```

* 📥 Input: `data/data.MOV`
* 📤 Output: Extracted PNG frames in `data/frames/`

---

#### 2. `Image_filter.py`

Filters images from `data/frames/`:

* Uses **YOLOv8** to detect a person.
* Uses **MediaPipe FaceMesh** to check for:

  * Two eyes
  * Nose
  * Mouth

✅ Only frames with these features are saved to `filtered_faces/`.

```bash
python Image_filter.py
```

---

### 🛠️ Requirements

Install required libraries:

```bash
pip install opencv-python ultralytics mediapipe pillow
```

---

### 🧠 Models Used

* **YOLOv8n** (`yolov8n.pt`): For person detection (lightweight and fast).
* **MediaPipe FaceMesh**: For detecting fine facial features (468 landmark points).

---

### 📌 Notes

* You can replace `yolov8n.pt` with a more accurate model (e.g., `yolov8s.pt`) if needed.
* Make sure `data/frames/` folder exists before running the filtering script.

---

### 📷 Sample Workflow

```bash
# Step 1: Extract frames
python Frame_scrapper_from_video.py

# Step 2: Filter frames with facial features
python Image_filter.py
```

