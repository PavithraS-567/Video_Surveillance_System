from ultralytics import YOLO

model=YOLO("weapon_detect.pt")

result=model(source=0, show=True)