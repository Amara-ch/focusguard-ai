import numpy as np
import cv2
import threading
import time
import sys
import platform


class YawnDetector:
    UPPER_LIP_INDICES = [13, 312, 82]
    LOWER_LIP_INDICES = [14, 317, 87]
    LEFT_MOUTH = 78
    RIGHT_MOUTH = 308

    def __init__(self, mar_threshold=0.40, yawn_min_frames=10):
        self.mar_threshold = mar_threshold
        self.yawn_min_frames = yawn_min_frames
        self.open_frame_counter = 0
        self.is_yawning = False
        self.total_yawns = 0
        self.yawn_logged = False
        print("[YawnDetector] Initialized OK")

    @staticmethod
    def _euclidean(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_point(self, landmarks, idx, w, h):
        lm = landmarks.landmark[idx]
        return (int(lm.x * w), int(lm.y * h))

    def _compute_mar(self, landmarks, w, h):
        verticals = []
        for u_idx, l_idx in zip(self.UPPER_LIP_INDICES, self.LOWER_LIP_INDICES):
            up = self._get_point(landmarks, u_idx, w, h)
            lo = self._get_point(landmarks, l_idx, w, h)
            verticals.append(self._euclidean(up, lo))
        avg_vertical = np.mean(verticals)
        left = self._get_point(landmarks, self.LEFT_MOUTH, w, h)
        right = self._get_point(landmarks, self.RIGHT_MOUTH, w, h)
        horizontal = self._euclidean(left, right)
        if horizontal == 0:
            return 0.0
        return avg_vertical / horizontal

    def analyze(self, landmarks, w, h):
        if landmarks is None:
            return {'mar': 0.0, 'mouth_open': False, 'is_yawning': False,
                    'total_yawns': self.total_yawns, 'mouth_points': []}
        mar = self._compute_mar(landmarks, w, h)
        mouth_open = mar > self.mar_threshold
        if mouth_open:
            self.open_frame_counter += 1
            if self.open_frame_counter >= self.yawn_min_frames:
                self.is_yawning = True
                if not self.yawn_logged:
                    self.total_yawns += 1
                    self.yawn_logged = True
        else:
            self.open_frame_counter = 0
            self.is_yawning = False
            self.yawn_logged = False
        mouth_points = [
            self._get_point(landmarks, self.LEFT_MOUTH, w, h),
            self._get_point(landmarks, 13, w, h),
            self._get_point(landmarks, self.RIGHT_MOUTH, w, h),
            self._get_point(landmarks, 14, w, h),
        ]
        return {'mar': mar, 'mouth_open': mouth_open,
                'is_yawning': self.is_yawning,
                'total_yawns': self.total_yawns,
                'mouth_points': mouth_points}

    def draw_mouth_status(self, frame, result):
        for pt in result['mouth_points']:
            cv2.circle(frame, pt, 3, (0, 165, 255), -1)
        return frame


class SoundAlarm:
    def __init__(self):
        self.is_playing = False
        self.stop_flag = threading.Event()
        self.thread = None
        if platform.system() == "Windows":
            try:
                import winsound
                self.winsound = winsound
                self.method = "winsound"
                print("[SoundAlarm] Using Windows winsound OK")
            except Exception as e:
                self.method = "none"
                print(f"[SoundAlarm] winsound failed: {e}")
        else:
            self.method = "none"

    def _beep_loop(self):
        while not self.stop_flag.is_set():
            if self.method == "winsound":
                try:
                    self.winsound.Beep(1000, 300)
                except Exception:
                    break
            time.sleep(0.1)

    def play(self):
        if self.is_playing or self.method == "none":
            return
        self.is_playing = True
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._beep_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.is_playing:
            return
        self.stop_flag.set()
        self.is_playing = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def cleanup(self):
        self.stop()


def run_test():
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from focusguard.core.vision_engine import VisionEngine
    from focusguard.models.eye_state import EyeStateDetector

    print("FocusGuard AI - YAWN DETECTOR FIXED VERSION")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No webcam!")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vision = VisionEngine()
    eye_detector = EyeStateDetector(ear_threshold=0.21, drowsy_frames=45)
    yawn_detector = YawnDetector(mar_threshold=0.40, yawn_min_frames=10)
    alarm = SoundAlarm()

    print("Press 'q' to quit")
    prev_time = time.time()
    fps = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            landmarks, face_found = vision.process_frame(frame)
            eye_result = eye_detector.analyze(landmarks, w, h)
            yawn_result = yawn_detector.analyze(landmarks, w, h)
            frame = eye_detector.draw_eye_status(frame, eye_result)
            frame = yawn_detector.draw_mouth_status(frame, yawn_result)

            frame_count += 1
            if time.time() - prev_time >= 1.0:
                fps = frame_count / (time.time() - prev_time)
                frame_count = 0
                prev_time = time.time()

            cv2.rectangle(frame, (0, 0), (w, 130), (40, 40, 40), -1)
            cv2.putText(frame, "FocusGuard AI - Driver Mode v2",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            ear_color = (0, 0, 255) if eye_result['eyes_closed'] else (0, 255, 0)
            cv2.putText(frame, f"EAR: {eye_result['ear']:.2f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 2)
            cv2.putText(frame, f"Eyes: {'CLOSED' if eye_result['eyes_closed'] else 'OPEN'}",
                        (140, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 2)
            cv2.putText(frame, f"Blinks: {eye_result['blinks']}",
                        (310, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (470, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            mar_color = (0, 165, 255) if yawn_result['mouth_open'] else (0, 255, 0)
            cv2.putText(frame, f"MAR: {yawn_result['mar']:.2f}",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mar_color, 2)
            cv2.putText(frame, f"Mouth: {'OPEN' if yawn_result['mouth_open'] else 'OK'}",
                        (140, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mar_color, 2)
            cv2.putText(frame, f"Yawns: {yawn_result['total_yawns']}",
                        (310, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            face_color = (0, 255, 0) if face_found else (0, 0, 255)
            cv2.putText(frame, f"FACE: {'OK' if face_found else 'NO'}",
                        (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)

            alert_active = False
            alert_text = ""
            if eye_result['is_drowsy']:
                alert_text = "!! DROWSINESS ALERT !!"
                alert_active = True
            elif yawn_result['is_yawning']:
                alert_text = "!! YAWN DETECTED !!"
                alert_active = True

            if alert_active:
                cv2.rectangle(frame, (0, 130), (w, h), (0, 0, 255), 8)
                cv2.rectangle(frame, (w // 2 - 220, h // 2 - 40),
                              (w // 2 + 220, h // 2 + 40), (0, 0, 255), -1)
                cv2.putText(frame, alert_text,
                            (w // 2 - 200, h // 2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                alarm.play()
            else:
                alarm.stop()

            cv2.putText(frame, "Press 'q' to quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
            cv2.imshow("FocusGuard AI Driver Mode v2", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        alarm.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        vision.release()
        print(f"Blinks: {eye_detector.total_blinks}, Yawns: {yawn_detector.total_yawns}")


if __name__ == "__main__":
    run_test()
