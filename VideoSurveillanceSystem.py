# Import necessary libraries
import threading  
import cv2  
import time 
import os  
import smtplib  
from email.message import EmailMessage  
from twilio.rest import Client 
from ultralytics import YOLO  
import numpy as np  
from dotenv import load_dotenv  

# Load environment variables from .env
load_dotenv()

# Load the YOLO model trained for weapon detection
model = YOLO("weapon_detect.pt")

# Fetch email and SMS credentials from environment
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
PASSWORD = os.getenv("EMAIL_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
account_sid = os.getenv("TWILIO_SID")
auth_token = os.getenv("TWILIO_TOKEN")
twilio_from = os.getenv("TWILIO_FROM")
twilio_to = os.getenv("TWILIO_TO")

# Timing configurations
cooldown = 10  # Cooldown in seconds for sending weapon alerts
blocked_cooldown = 10  # Cooldown for blocked camera alerts
block_detection_time = 3  # Duration to confirm camera blockage

# Log events to a text file with timestamps
def log_event(event_text):
    try:
        with open("alert_log.txt", "a", encoding="utf-8") as log_file:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"[{timestamp}] {event_text}\n")
    except Exception as e:
        print(f"Logging error: {e}")

# Send email alert with a snapshot of the incident
def send_email_alert_with_snapshot(image_path, cam_id, blocked=False, reason="Blocked"):
    msg = EmailMessage()
    subject_prefix = "Blocked" if blocked else "Weapon Detected"
    msg["Subject"] = f"Alert: {subject_prefix} on Camera {cam_id}"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    context = f"Camera Alert: {reason}" if blocked else "A weapon has been detected."
    msg.set_content(f"{context}\nCamera ID: {cam_id}")

    # Attach the snapshot image to the email
    with open(image_path, "rb") as img:
        msg.add_attachment(img.read(), maintype="image", subtype="jpeg", filename=os.path.basename(image_path))

    # Send the email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, PASSWORD)
            server.send_message(msg)
        print(f"📩 Email sent for Camera {cam_id} ({reason if blocked else 'Weapon'})")
        log_event(f"📩 Email sent for Camera {cam_id} ({reason if blocked else 'Weapon'})")
    except Exception as e:
        print(f"❌ Email failed for Camera {cam_id}: {e}")
        log_event(f"❌ Email failed for Camera {cam_id}: {e}")

# Send SMS alert using Twilio
def send_sms_alert(cam_id, blocked=False, reason="Blocked"):
    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            from_=twilio_from,
            body=f"{'Blocked' if blocked else 'Weapon Detected'} on Camera {cam_id}!",
            to=twilio_to
        )
        print(f"📱 SMS sent for Camera {cam_id} (SID: {message.sid})")
        log_event(f"📱 SMS sent for Camera {cam_id} (SID: {message.sid})")
    except Exception as e:
        print(f"❌ SMS failed for Camera {cam_id}: {e}")
        log_event(f"❌ SMS failed for Camera {cam_id}: {e}")

# Main detection logic for each camera
def detect_from_camera(cam_index):
    cap = cv2.VideoCapture(cam_index)
    last_alert_time = 0
    last_blocked_time = 0
    block_start_time = None
    partial_block_start_time = None

    if not cap.isOpened():
        print(f"❌ Could not open Camera {cam_index}")
        return

    print(f"📷 Camera {cam_index} started")
    log_event(f"📷 Camera {cam_index} started")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Failed to read from Camera {cam_index}")
            log_event(f"❌ Failed to read from Camera {cam_index}")
            break

        current_time = time.time()

        # Preprocess for blockage detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = gray.mean()
        dark_pixels = np.sum(gray < 30)
        dark_ratio = dark_pixels / gray.size

        # Check for full blockage (very low brightness)
        is_fully_blocked = avg_brightness < 30
        # Check for partial blockage (high ratio of dark pixels)
        is_partially_blocked = dark_ratio > 0.5

        # === Fully Blocked Camera Logic ===
        if is_fully_blocked:
            if block_start_time is None:
                block_start_time = current_time
            elif current_time - block_start_time >= block_detection_time and current_time - last_blocked_time > blocked_cooldown:
                print(f"🚫 Camera {cam_index} fully blocked")
                snapshot_path = f"snapshots/fully_blocked/cam{cam_index}_{int(current_time)}.jpg"
                os.makedirs("snapshots/fully_blocked", exist_ok=True)
                cv2.imwrite(snapshot_path, frame)
                send_email_alert_with_snapshot(snapshot_path, cam_index, blocked=True, reason="Fully Blocked")
                send_sms_alert(cam_index, blocked=True, reason="Fully Blocked")
                last_blocked_time = current_time
                block_start_time = None
        else:
            block_start_time = None  # Reset timer if not fully blocked

        # === Partially Blocked Camera Logic ===
        if not is_fully_blocked and is_partially_blocked:
            if partial_block_start_time is None:
                partial_block_start_time = current_time
            elif current_time - partial_block_start_time >= block_detection_time and current_time - last_blocked_time > blocked_cooldown:
                print(f"🚫 Camera {cam_index} partially blocked")
                snapshot_path = f"snapshots/partially_blocked/cam{cam_index}_{int(current_time)}.jpg"
                os.makedirs("snapshots/partially_blocked", exist_ok=True)
                cv2.imwrite(snapshot_path, frame)
                send_email_alert_with_snapshot(snapshot_path, cam_index, blocked=True, reason="Partially Blocked")
                send_sms_alert(cam_index, blocked=True, reason="Partially Blocked")
                last_blocked_time = current_time
                partial_block_start_time = None
        else:
            partial_block_start_time = None  # Reset timer if not partially blocked

        # === Weapon Detection Using YOLO ===
        results = model.predict(source=frame, conf=0.5, imgsz=512)
        for result in results:
            weapon_detected = False
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # Class 0 is assumed to be 'weapon'
                    weapon_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Weapon", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Send alert if weapon detected and cooldown has passed
            if weapon_detected and (current_time - last_alert_time > cooldown):
                print(f"🚨 Weapon detected on Camera {cam_index}")
                snapshot_path = f"snapshots/weapon/cam{cam_index}_{int(current_time)}.jpg"
                os.makedirs("snapshots/weapon", exist_ok=True)
                cv2.imwrite(snapshot_path, frame)
                send_email_alert_with_snapshot(snapshot_path, cam_index)
                send_sms_alert(cam_index)
                last_alert_time = current_time

        # Show live video feed
        cv2.imshow(f"Camera {cam_index}", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up on exit
    cap.release()
    cv2.destroyWindow(f"Camera {cam_index}")
    print(f"📴 Camera {cam_index} stopped")
    log_event(f"📴 Camera {cam_index} stopped")

# Start monitoring on selected camera indices
camera_indices = [0]  # Add more indices if multiple cameras are used
threads = []
for index in camera_indices:
    t = threading.Thread(target=detect_from_camera, args=(index,))
    t.start()
    threads.append(t)

# Wait for all threads to finish
for t in threads:
    t.join()