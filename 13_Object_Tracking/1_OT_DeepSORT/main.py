from ultralytics import YOLO
import cv2

from writer import writer
from detector import detector
from tracker import Tracker

model = YOLO("yolov8m.pt")
# model = YOLO("yolov8s-seg.pt")

show_video = True
video_path = "./13_Object_Tracking/files/walk.mp4"
cap = cv2.VideoCapture(video_path)
video_writer = writer(cap=cap)
obj_tracker = Tracker()

ret, frame = cap.read()
while ret:
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bboxes_xywh, confs, class_name = detector(frame=frame, model=model)
    
    frame = obj_tracker.update_track(bboxes_xywh, confs, frame, class_name)

    video_writer.write(frame=frame)

    if show_video:
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ret, frame = cap.read()

cap.release()
video_writer.close()
cv2.destroyAllWindows()