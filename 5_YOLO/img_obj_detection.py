import cv2
import matplotlib.pyplot as plt
from utils import detect_objects, print_objects, plot_boxes

""" <nms_thresh>
    YOLO uses Non-Maximal Suppression (NMS) to only keep the best bounding box.
    YOLO removes bounding boxes that have a detection probability less than NMS threshold.
"""
""" <iou_thresh>
    The second step in NMS, is to select the bounding boxes with the highest
    detection probability and eliminate all the bounding boxes whose
    Intersection Over Union (IOU) value is higher than a given IOU threshold
    with respect to the best bounding boxes.
"""

def img_obj_detection(model,
                      class_names,
                      dir='./5_YOLO/imgs/surf.jpg',
                      nms_thresh=0.6,
                      iou_thresh=0.4):
    image = cv2.imread(dir)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (model.width, model.height))
    boxes = detect_objects(model, resized_image, iou_thresh, nms_thresh)
    print_objects(boxes, class_names)
    plot_boxes(original_image, boxes, class_names, plot_labels = True)