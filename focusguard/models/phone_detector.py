import cv2
import time
from ultralytics import YOLO


class PhoneDetector:
    PHONE_CLASS_ID = 67

    def __init__(self, model_path='yolov8n.pt', confidence=0.45, alert_frames=8,
                 min_area_ratio=0.008):
        '''
        Balanced filtering: removes lipstick/pen but keeps real phones.
        - confidence: 0.45 (was 0.40 too low, 0.55 too high)
        - min_area_ratio: 0.8% of frame minimum (filters tiny pen/lipstick)
        - NO aspect ratio filter (was too strict)
        '''
        print('[PhoneDetector] Loading YOLOv8 model...')
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.alert_frames = alert_frames
        self.min_area_ratio = min_area_ratio
        self.phone_frame_counter = 0
        self.phone_detected = False
        self.total_phone_events = 0
        self.event_logged = False
        print('[PhoneDetector] Initialized OK (balanced mode)')
        print('  conf >= %.2f, min_area >= %.1f%% of frame' % (
            confidence, min_area_ratio * 100))

    def analyze(self, frame):
        h, w = frame.shape[:2]
        frame_area = float(w * h)
        results = self.model(frame, verbose=False, conf=self.confidence,
                             classes=[self.PHONE_CLASS_ID])
        boxes = []
        confs = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id != self.PHONE_CLASS_ID:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue
                    area_ratio = (bw * bh) / frame_area
                    # Only filter: must be at least 0.8% of frame
                    # (filters out tiny pen/lipstick detections)
                    if area_ratio < self.min_area_ratio:
                        continue
                    boxes.append((x1, y1, x2, y2))
                    confs.append(conf)

        phone_visible = len(boxes) > 0
        if phone_visible:
            self.phone_frame_counter += 1
            if self.phone_frame_counter >= self.alert_frames:
                self.phone_detected = True
                if not self.event_logged:
                    self.total_phone_events += 1
                    self.event_logged = True
        else:
            self.phone_frame_counter = max(0, self.phone_frame_counter - 1)
            if self.phone_frame_counter == 0:
                self.phone_detected = False
                self.event_logged = False

        return {
            'phone_visible': phone_visible,
            'phone_detected': self.phone_detected,
            'boxes': boxes,
            'confidences': confs,
            'total_events': self.total_phone_events
        }

    def draw_boxes(self, frame, result):
        for (x1, y1, x2, y2), conf in zip(result['boxes'], result['confidences']):
            color = (0, 0, 255) if result['phone_detected'] else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = 'PHONE %.2f' % conf
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 130, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return frame
