import cv2
import numpy as np
import os
from utils import DATASET_DIR, TRAINER_DIR, ensure_dirs
import pickle

ensure_dirs()

face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Prepare data
faces = []
ids = []
labels = {}  # id -> name

for folder in os.listdir(DATASET_DIR):
    folder_path = os.path.join(DATASET_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # Expected folder format: user_<id>_<name>
    parts = folder.split('_')

    if len(parts) < 3:
        print(f"Skipping invalid folder: {folder}")
        continue

    try:
        user_id = int(parts[1])
    except ValueError:
        print(f"Invalid user ID in folder: {folder}")
        continue

    user_name = "_".join(parts[2:])
    labels[user_id] = user_name

    for file in os.listdir(folder_path):
        if not file.lower().endswith('.jpg'):
            continue

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        faces.append(img)
        ids.append(user_id)

if len(faces) == 0:
    raise SystemExit('No training images found. Run capture_images.py first.')

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(ids))

trainer_path = os.path.join(TRAINER_DIR, 'trainer.yml')
recognizer.write(trainer_path)
print('Model trained and saved to', trainer_path)

labels_path = os.path.join(TRAINER_DIR, 'labels.pickle')
with open(labels_path, 'wb') as f:
    pickle.dump(labels, f)

print('Labels saved to', labels_path)
