import cv2
import pickle
import os
from utils import TRAINER_DIR, mark_attendance, get_location

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

trainer_path = os.path.join(TRAINER_DIR, 'trainer.yml')
labels_path = os.path.join(TRAINER_DIR, 'labels.pickle')

if not os.path.exists(trainer_path):
    raise SystemExit('Trainer file not found. Run train_model.py first.')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

with open(labels_path, 'rb') as f:
    labels = pickle.load(f)

# ✅ Define once (not inside loop)
CONF_THRESHOLD = 70

# ✅ Track marked names in current session
marked_names = set()

# Get GPS-like location once per session
latitude, longitude = get_location()
print("Current Location (approx):", latitude, longitude)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam. Check your camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (200, 200))

        id_pred, conf = recognizer.predict(roi_resized)

        if conf < CONF_THRESHOLD:
            name = labels.get(id_pred, f'ID_{id_pred}')

            cv2.putText(frame, f'{name} ({conf:.1f})', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # ✅ Prevent repeated marking in same session
            if name not in marked_names:
                marked = mark_attendance(name, latitude=latitude, longitude=longitude)

                if marked:
                    print(f"Marked attendance for {name} at ({latitude}, {longitude})")
                    marked_names.add(name)

        else:
            cv2.putText(frame, 'Unknown', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Recognition - press q to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()