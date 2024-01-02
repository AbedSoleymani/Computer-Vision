import cv2
import numpy as np

from device import device
from class_names import class_names

def detector(frame, model):

    results = model(frame, device=device, classes=0, conf=0.8) # `classes=0` Just for detecting person
    
    boxes = results[-1].boxes
    probs = results[-1].probs
    cls = boxes.cls.tolist()
    xyxy = boxes.xyxy
    conf = boxes.conf
    xywh = boxes.xywh

    pred_cls = np.array(cls)
    conf = conf.detach().cpu().numpy()
    xyxy = xyxy.detach().cpu().numpy()
    bboxes_xywh = xywh.cpu().numpy()
    bboxes_xywh = np.array(bboxes_xywh, dtype=float)

    class_name = ""
    if cls:
        class_name = class_names[int(cls[-1])]

    return bboxes_xywh, conf, class_name