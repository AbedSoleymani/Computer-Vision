from utils import *
from darknet import Darknet

def model_setup(cfg_file='./5_YOLO/cfg/yolov3.cfg',
                weight_file='./5_YOLO/pre-trained_model/yolov3.weights',
                namesfile='./5_YOLO/names/coco.names'):
    
    model = Darknet(cfg_file)
    model.load_weights(weight_file)
    class_names = load_class_names(namesfile)

    return model, class_names