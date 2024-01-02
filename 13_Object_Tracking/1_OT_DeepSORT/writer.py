import cv2

class writer():
   def __init__(self, cap,
                output_path = "./13_Object_Tracking/files/deep_SORT_output.mp4"):

      frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

      fps = cap.get(cv2.CAP_PROP_FPS)
      fourcc = cv2.VideoWriter_fourcc(*'mp4v')
      self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
   def write(self, frame):
      self.video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
   
   def close(self):
      self.video_writer.release()