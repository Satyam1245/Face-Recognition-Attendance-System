import cv2
import os
from utils import DATASET_DIR, ensure_dirs


ensure_dirs()


face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)


# Get user input
user_id = input('Enter numeric user id (e.g. 1): ').strip()
user_name = input('Enter user name (no spaces preferred): ').strip()
if not user_id.isdigit():
    raise SystemExit('user id must be numeric')


user_folder = os.path.join(DATASET_DIR, f'user_{user_id}_{user_name}')
os.makedirs(user_folder, exist_ok=True)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot access webcam. Check your camera.")
count = 0
print('Press q to quit early. Capturing 30 face samples...')


while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (200, 200))
        file_path = os.path.join(user_folder, f'{user_id}_{count}.jpg')
        cv2.imwrite(file_path, face_resized)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, f'{user_name} {count}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.imshow('Capture - press q to exit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count >= 30:
        break


cap.release()
cv2.destroyAllWindows()
print(f'Captured {count} samples to {user_folder}')