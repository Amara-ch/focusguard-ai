import cv2
from ultralytics import YOLO


class DistractionObjectDetector:
    '''
    Mode-aware object distraction detector.
    - DRIVER mode: bottles/cups/food = distraction
    - STUDENT mode: only food/drink = distraction (study tools allowed)
    '''

    # Full COCO class map for distraction-relevant items
    ALL_DISTRACTION_CLASSES = {
        39: 'bottle',
        41: 'cup',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        63: 'laptop',
        64: 'mouse',
        66: 'keyboard',
        73: 'book',
        76: 'scissors',
    }

    # DRIVER MODE: anything not in driving = distraction
    DRIVER_DISTRACTIONS = {
        39: 'bottle', 41: 'cup', 44: 'spoon', 45: 'bowl',
        46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
        50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake',
        63: 'laptop', 64: 'mouse', 66: 'keyboard',
        73: 'book', 76: 'scissors',
    }

    # STUDENT MODE: study tools (book, laptop, pencil, mouse, keyboard) ALLOWED
    # Only FOOD and DRINKS are distractions
    STUDENT_DISTRACTIONS = {
        39: 'bottle',
        41: 'cup',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
    }

    def __init__(self, model_path='yolov8n.pt', confidence=0.40, alert_frames=10,
                 mode='driver'):
        '''
        mode: 'driver' or 'student'
        '''
        print('[DistractionObjectDetector] Loading YOLOv8 (mode=' + mode + ')...')
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.alert_frames = alert_frames
        self.mode = mode.lower()

        if self.mode == 'student':
            self.distraction_map = self.STUDENT_DISTRACTIONS
            print('  Student mode: only food/drink count as distraction')
            print('  Study tools (book, laptop, pencil, etc.) ALLOWED')
        else:
            self.distraction_map = self.DRIVER_DISTRACTIONS
            print('  Driver mode: all non-driving items = distraction')

        self.frame_counter = 0
        self.is_distracted = False
        self.total_events = 0
        self.event_logged = False
        self.last_objects = []
        print('[DistractionObjectDetector] Initialized OK')

    def analyze(self, frame):
        class_ids = list(self.distraction_map.keys())
        results = self.model(frame, verbose=False, conf=self.confidence,
                             classes=class_ids)
        detections = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.distraction_map:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = float(box.conf[0])
                        name = self.distraction_map[cls_id]
                        detections.append({
                            'name': name, 'conf': conf,
                            'box': (x1, y1, x2, y2)
                        })

        object_visible = len(detections) > 0
        if object_visible:
            self.frame_counter += 1
            if self.frame_counter >= self.alert_frames:
                self.is_distracted = True
                if not self.event_logged:
                    self.total_events += 1
                    self.event_logged = True
        else:
            self.frame_counter = 0
            self.is_distracted = False
            self.event_logged = False

        self.last_objects = [d['name'] for d in detections]
        return {
            'detections': detections,
            'object_visible': object_visible,
            'is_distracted': self.is_distracted,
            'total_events': self.total_events,
            'object_names': self.last_objects
        }

    def draw_boxes(self, frame, result):
        for det in result['detections']:
            x1, y1, x2, y2 = det['box']
            color = (0, 0, 255) if result['is_distracted'] else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = '%s %.2f' % (det['name'].upper(), det['conf'])
            cv2.rectangle(frame, (x1, y1 - 22), (x1 + 160, y1), color, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame
