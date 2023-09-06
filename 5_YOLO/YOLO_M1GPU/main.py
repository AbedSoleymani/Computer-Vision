import cv2
from ultralytics import YOLO  # pip install ultralytics
import numpy as np
import labels

cap = cv2. VideoCapture("./5_YOLO/YOLO_M1GPU/video/dogs.mp4")
model = YOLO ("yolov8m.pt")


while True:

    ret, frame = cap.read()

    results = model (frame, device="mps")
    result = results [0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes = np.array(result.boxes.cls.cpu(), dtype="int")
    for cls, bbox in zip (classes, bboxes) :
        (x1, y1, x2, y2) = bbox
        cv2.rectangle (img=frame,
                        pt1=(x1, y1),
                        pt2=(x2, y2),
                        color=(0, 0, 225),
                        thickness=2)
        
        cv2.putText(img=frame,
                    text=labels.labels[cls],
                    org=(x1, y1 - 5),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale= 2,
                    color=(0, 0, 225),
                    thickness=2)
        cv2.imshow("Annotated Video", frame)
        
    key = cv2.waitKey(1)
    if key == 27 or not ret:
        break

cap.release()
cv2.destroyAllWindows()