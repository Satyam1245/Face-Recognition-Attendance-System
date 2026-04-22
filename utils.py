import os
from datetime import datetime
import csv
import geocoder

DATASET_DIR = 'dataset'
TRAINER_DIR = 'trainer'
ATTENDANCE_FILE = 'attendance.csv'

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)


def ensure_dirs():
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(TRAINER_DIR, exist_ok=True)


def get_location():
    """
    Get approximate GPS location (latitude, longitude) using IP-based geolocation.
    Returns (lat, lon) or (None, None) if not available.
    """
    try:
        g = geocoder.ip('me')
        if g.ok and g.latlng:
            return g.latlng[0], g.latlng[1]
    except Exception:
        pass
    return None, None



def mark_attendance(name, latitude=None, longitude=None, attendance_file=ATTENDANCE_FILE):
    """
    Append attendance with name, timestamp, and location.
    Prevents duplicate entries for the same person on the same day.
    """

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Ensure file exists with header
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp", "Latitude", "Longitude"])

    # 🔹 STRICT duplicate check (accurate)
    with open(attendance_file, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            if len(row) >= 2:
                try:
                    ts = datetime.fromisoformat(row[1])
                    if row[0] == name and ts.strftime("%Y-%m-%d") == today_str:
                        return False  # already marked today
                except Exception:
                    continue

    # Fetch location if not provided
    if latitude is None or longitude is None:
        try:
            latitude, longitude = get_location()
        except Exception:
            latitude, longitude = "", ""

    # Append new attendance entry
    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            datetime.now().isoformat(),
            latitude if latitude else "",
            longitude if longitude else ""
        ])

    return True