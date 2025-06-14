# 🎾 AI Tennis Broadcasting Director

An AI-powered system that automates camera angle selection in tennis broadcasts using computer vision and machine learning. The system intelligently chooses between Baseline, Sideline, and Top Corner views in near real-time, mimicking the decisions of a professional broadcast director.

---

## 📸 Demo (UI Preview)

> A real-time OpenCV interface displays the three camera feeds as thumbnails and highlights the predicted main view.

*(Add screenshot or GIF here later)*

---

## 🧠 Core Technologies

| Component        | Description                                  |
|------------------|----------------------------------------------|
| **YOLOv8**        | Pretrained object detection (players, ball)  |
| **XGBoost**       | Multiclass classifier to predict best angle  |
| **OpenCV**        | Frame extraction, display, and UI rendering  |
| **Anaconda**      | Environment and package management           |
| **Joblib**        | Saves the feature scaler                     |

---

## 🗂 Folder Structure

```
tennis-ai-director/
│
├── data/
│   ├── raw/                  # Input videos
│   ├── synced_frames/        # Extracted & aligned frames
│   ├── manual_annotations.xlsx
│   └── extracted_features_multi_view.xlsx
│
├── models/                   # YOLO & XGBoost model files
├── scripts/                  # Frame extraction, syncing, annotation tools
├── ui/
│   └── main.py               # Main executable (runs the UI + prediction)
├── utils/
│   ├── paths.py              # All path configs
│   └── feature_scaler.joblib
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Clone the repo (after push)
```bash
git clone https://github.com/your-username/tennis-ai-director.git
cd tennis-ai-director
```

### 2. Create Conda Environment
```bash
conda create -n tennis-env python=3.10
conda activate tennis-env
pip install -r requirements.txt
```

### 3. Place your videos
- Put your three video files inside `data/raw/` as:
  - `baseline/baseline.mp4`
  - `sideline/sideline.mp4`
  - `topcorner/topcorner.mp4`

### 4. Run the App
```bash
python ui/main.py
```

---

## 📊 Model Performance

- 📈 Accuracy: ~89%
- 🧠 Handles missing detections with default fallbacks
- 🏅 Especially strong at detecting net-play scenarios (Sideline)

---

## 🔮 Future Improvements

- Real-time camera switching hardware integration
- Smooth transitions with fade/pan effects
- Transformer-based temporal prediction
- Live match analytics dashboard

---

## 🙌 Contributors

- **Hrithik S**
- **Akash Tomy**
- **Naveen Mathew**

---

## 📜 License

This project was developed as part of the M.Sc. (AI) program at CUSAT and is shared for academic purposes.
