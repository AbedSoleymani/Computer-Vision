from model_setup import model_setup
from img_obj_detection import img_obj_detection

model, class_names = model_setup()
img_obj_detection(model=model,
                  class_names=class_names,
                  dir='./5_YOLO/imgs/wine.jpg')
