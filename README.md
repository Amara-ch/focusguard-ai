# FocusGuard AI

**A Unified Real-Time Attention and Drowsiness Monitoring System for Drivers and Students using Computer Vision**


## 📋 Abstract

FocusGuard AI is a real-time computer vision system designed to monitor attention and detect drowsiness/distraction in two critical environments: driving and studying. The system combines facial landmark detection, object detection, and behavioral analysis to generate safety/focus scores and provide instant alerts.

---

## ✨ Key Features

### 🚗 Driver Mode
- Drowsiness Detection (Eye Aspect Ratio - EAR)
- Yawn Detection (Mouth Aspect Ratio - MAR)
- Phone Usage Detection (YOLOv8)
- Head Pose Estimation (Eyes off road)
- Object Distraction Detection (food, drinks, etc.)
- Real-time Audio Alerts
- Safety Score (A-F Grade)

### 📚 Student Mode
- All Driver Mode features (with relaxed thresholds)
- Absence Detection (student left seat)
- Multi-Face Detection (cheating prevention)
- Focus Percentage Tracking
- Detailed Session Reports

---

## 🛠️ Technologies Used

- **MediaPipe** — Face mesh and landmark detection
- **YOLOv8 (Ultralytics)** — Object detection (phone, food, etc.)
- **OpenCV** — Video processing and camera handling
- **Streamlit** — Interactive Web Interface
- **NumPy** — Mathematical computations

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Amara-ch/focusguard-ai.git
cd focusguard-ai

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate # Linux / Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
