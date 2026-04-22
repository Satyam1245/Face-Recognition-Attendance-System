# Face Recognition Attendance System

A Python-based face recognition system that automatically marks attendance using a webcam and OpenCV.

---

## 🚀 Features

* Capture face dataset using webcam
* Train LBPH (Local Binary Patterns Histogram) face recognizer
* Real-time face detection and recognition
* Automatic attendance logging (prevents duplicates per day)
* Location tagging using IP-based geolocation

---

## 📁 Project Structure

```
Face-Recognition-Attendance-System/
│
├── src/
│   ├── capture_images.py
│   ├── train_model.py
│   ├── recognize.py
│   └── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

All core source files are organized inside the `src/` directory for better modularity and maintainability.

---

## ⚙️ Installation

```bash
git clone https://github.com/Satyam1245/Face-Recognition-Attendance-System.git
cd Face-Recognition-Attendance-System
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Capture Images

```bash
python src/capture_images.py
```

### 2. Train Model

```bash
python src/train_model.py
```

### 3. Run Attendance System

```bash
python src/recognize.py
```

---

## 📊 Output

* Attendance is stored in `attendance.csv`

### Format:

```
Name,Timestamp,Latitude,Longitude
```

---

## 🧠 How It Works

* Uses Haar Cascade for face detection
* Uses LBPH algorithm for face recognition
* Stores attendance with timestamp and approximate location
* Prevents duplicate entries for the same user on the same day

---

## ⚠️ Notes

* Ensure your webcam is connected
* You must capture images before training the model
* Dataset is not included in this repository
* Location is approximate (based on IP, not GPS)
* Works best in good lighting conditions

---

## 🛠️ Requirements

* Python 3.8+
* OpenCV
* NumPy
* Geocoder

---

## 📌 Future Improvements

* GUI (Tkinter / Web App)
* Deep learning (FaceNet / Dlib)
* Database integration (SQLite / MySQL)
* Dashboard for attendance analytics

---

## 📜 License

This project is licensed under the MIT License.
