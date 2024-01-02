from deep_sort.deep_sort import DeepSort
import cv2

class Tracker():
    def __init__(self,
                 pretrained_weights = './13_Object_Tracking/1_OT_DeepSORT/deep_sort/deep/checkpoint/ckpt.t7'):
        
        self.tracker = DeepSort(model_path=pretrained_weights, max_age=20)
        self.unique_track_ids = set()

    def update_track(self, bboxes_xywh, confs, frame, class_name):

        tracks = self.tracker.update(bbox_xywh=bboxes_xywh,
                                     confidences=confs,
                                     ori_img=frame)
        text_color = (0, 0, 0) # Black
    
        for track in self.tracker.tracker.tracks:
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_tlbr()
            w = x2 - x1
            h = y2 - y1

            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
            color_id = track_id % len(colors)
            color = colors[color_id]

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            cv2.putText(frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

            self.unique_track_ids.add(track_id)


        cv2.putText(frame, f"{class_name} Count: {len(self.unique_track_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        return frame