# YOLO: utilizing Apple Silicon GPU accelator
In this repo, we implemented the medium version of YOLO model (`yolov8m`) with `mps` device.
We had 80% reduction in executing speed: reducing from 156 msec average proccessing time per frame to 32 msec! 5x faster!!