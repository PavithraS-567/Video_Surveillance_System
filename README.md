# Video Surveillance System - Weapon Detection

A real-time weapon detection system using YOLOv8, capable of alerting via Email and Twilio SMS when a weapon is detected or a camera is blocked.

## 🔧 Features
- Weapon detection using YOLOv8
- Blocked/partially blocked camera detection
- Alerts via email and SMS (Twilio)
- Auto-snapshots and logging

## 📁 Files
- `VideoSurveillanceSystem.py`: Main file for live surveillance
- `detection.py`: Simple test detection
- `alert_log.txt`: Logs of all alerts
- `.env`: **DO NOT UPLOAD** (contains credentials)

## 📦 Requirements
Install using:
ultralytics
opencv-python
numpy
python-dotenv
twilio

```bash
pip install -r requirements.txt


