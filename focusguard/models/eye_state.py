"""
FocusGuard AI - Eye State Detector
===================================
Detects eye open/closed state using Eye Aspect Ratio (EAR).
EAR formula: ratio of vertical to horizontal eye landmark distances.
When eyes close, EAR drops sharply.

Reference: "Real-Time Eye Blink Detection using Facial Landmarks"
           by Soukupová & Čech (2016)

Author: humbleunitydev
Project: FocusGuard AI
"""

import numpy as np
import cv2


class EyeStateDetector:
    """
    Calculates Eye Aspect Ratio (EAR) from MediaPipe FaceMesh landmarks
    and classifies eye state as OPEN or CLOSED.
    """

    # MediaPipe FaceMesh landmark indices for the eyes
    # (refined landmarks model - 478 total points)
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]   # 6 points around left eye
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # 6 points around right eye

    def __init__(self, ear_threshold=0.21, drowsy_frames=45):
        """
        Initialize the eye state detector.

        Args:
            ear_threshold: EAR value below which eyes are considered closed.
                          Typical range: 0.18 - 0.25 (calibrate per person)
            drowsy_frames: Number of consecutive closed-eye frames to trigger
                          drowsiness alert. At ~15 FPS, 45 frames ≈ 3 seconds.
        """
        self.ear_threshold = ear_threshold
        self.drowsy_frames = drowsy_frames
        self.closed_frame_counter = 0
        self.is_drowsy = False
        self.total_blinks = 0
        self.was_closed = False

        print("[EyeStateDetector] Initialized ✓")
        print(f"  EAR threshold: {ear_threshold}")
        print(f"  Drowsy after: {drowsy_frames} frames")

    @staticmethod
    def _euclidean(point1, point2):
        """Calculate Euclidean distance between two (x,y) points."""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def _compute_ear(self, eye_points):
        """
        Compute Eye Aspect Ratio for one eye.

        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

        Args:
            eye_points: list of 6 (x,y) tuples around the eye

        Returns:
            float: EAR value (typically 0.15 to 0.35)
        """
        # Vertical distances
        vertical_1 = self._euclidean(eye_points[1], eye_points[5])
        vertical_2 = self._euclidean(eye_points[2], eye_points[4])

        # Horizontal distance
        horizontal = self._euclidean(eye_points[0], eye_points[3])

        # Avoid division by zero
        if horizontal == 0:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def _extract_eye_points(self, landmarks, indices, image_width, image_height):
        """
        Extract pixel coordinates for given landmark indices.

        Args:
            landmarks: MediaPipe face landmarks (normalized 0-1)
            indices: list of landmark indices
            image_width, image_height: frame dimensions

        Returns:
            list of (x, y) pixel tuples
        """
        points = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)
            points.append((x, y))
        return points

    def analyze(self, landmarks, image_width, image_height):
        """
        Main analysis function. Computes EAR for both eyes and determines state.

        Args:
            landmarks: MediaPipe face landmarks
            image_width, image_height: frame dimensions

        Returns:
            dict with keys:
                'ear': average EAR value (float)
                'eyes_closed': True/False
                'is_drowsy': True/False (sustained closure)
                'blinks': total blink count (int)
                'left_eye_points': pixel points for drawing
                'right_eye_points': pixel points for drawing
        """
        if landmarks is None:
            return {
                'ear': 0.0,
                'eyes_closed': False,
                'is_drowsy': False,
                'blinks': self.total_blinks,
                'left_eye_points': [],
                'right_eye_points': []
            }

        # Extract eye landmark points
        left_eye = self._extract_eye_points(
            landmarks, self.LEFT_EYE_INDICES, image_width, image_height
        )
        right_eye = self._extract_eye_points(
            landmarks, self.RIGHT_EYE_INDICES, image_width, image_height
        )

        # Compute EAR for both eyes
        left_ear = self._compute_ear(left_eye)
        right_ear = self._compute_ear(right_eye)

        # Average EAR (more stable than single eye)
        avg_ear = (left_ear + right_ear) / 2.0

        # Classify eye state
        eyes_closed = avg_ear < self.ear_threshold

        # Drowsiness detection (sustained closure)
        if eyes_closed:
            self.closed_frame_counter += 1
            if self.closed_frame_counter >= self.drowsy_frames:
                self.is_drowsy = True
        else:
            # Eyes opened - count blink if just transitioned
            if self.was_closed and self.closed_frame_counter < self.drowsy_frames:
                self.total_blinks += 1
            self.closed_frame_counter = 0
            self.is_drowsy = False

        self.was_closed = eyes_closed

        return {
            'ear': avg_ear,
            'eyes_closed': eyes_closed,
            'is_drowsy': self.is_drowsy,
            'blinks': self.total_blinks,
            'left_eye_points': left_eye,
            'right_eye_points': right_eye
        }

    def draw_eye_status(self, frame, result):
        """
        Draw eye landmarks and status overlay on frame.

        Args:
            frame: BGR image
            result: dict returned by analyze()
        """
        # Draw circles around each eye landmark
        for point in result['left_eye_points']:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)  # Yellow
        for point in result['right_eye_points']:
            cv2.circle(frame, point, 2, (0, 255, 255), -1)

        # Draw eye contour lines
        if len(result['left_eye_points']) == 6:
            pts_left = np.array(result['left_eye_points'], dtype=np.int32)
            pts_right = np.array(result['right_eye_points'], dtype=np.int32)
            cv2.polylines(frame, [pts_left], True, (0, 255, 0), 1)
            cv2.polylines(frame, [pts_right], True, (0, 255, 0), 1)

        return frame


# ============================================================
# STANDALONE TEST RUNNER
# ============================================================
def run_test():
    """
    Test the eye state detector with live webcam.
    Combines VisionEngine (face) + EyeStateDetector (EAR).
    """
    import sys
    import os
    # Add parent folder to import path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))

    from focusguard.core.vision_engine import VisionEngine
    import time

    print("\n" + "=" * 50)
    print("FocusGuard AI - Eye State Detector Test")
    print("=" * 50)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Webcam not accessible!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vision = VisionEngine()
    eye_detector = EyeStateDetector(ear_threshold=0.21, drowsy_frames=45)

    print("[INFO] Press 'q' to quit")
    print("[INFO] Try closing your eyes for 3 seconds to trigger DROWSY alert\n")

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

            # Get face landmarks
            landmarks, face_found = vision.process_frame(frame)

            # Analyze eye state
            eye_result = eye_detector.analyze(landmarks, w, h)

            # Draw eye landmarks
            frame = eye_detector.draw_eye_status(frame, eye_result)

            # FPS calc
            frame_count += 1
            if time.time() - prev_time >= 1.0:
                fps = frame_count / (time.time() - prev_time)
                frame_count = 0
                prev_time = time.time()

            # ============ OVERLAY UI ============
            # Top banner
            cv2.rectangle(frame, (0, 0), (w, 110), (40, 40, 40), -1)
            cv2.putText(frame, "FocusGuard AI - Driver Mode (Eye State)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # EAR value
            ear_color = (0, 255, 0) if not eye_result['eyes_closed'] else (0, 0, 255)
            cv2.putText(frame, f"EAR: {eye_result['ear']:.3f}",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        ear_color, 2)

            # Eye state
            eye_status = "CLOSED" if eye_result['eyes_closed'] else "OPEN"
            cv2.putText(frame, f"Eyes: {eye_status}",
                        (180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        ear_color, 2)

            # Blink count
            cv2.putText(frame, f"Blinks: {eye_result['blinks']}",
                        (350, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # FPS
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (510, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)

            # Face detection status
            face_status = "FACE OK" if face_found else "NO FACE"
            face_color = (0, 255, 0) if face_found else (0, 0, 255)
            cv2.putText(frame, face_status,
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        face_color, 2)

            # ============ DROWSINESS ALERT ============
            if eye_result['is_drowsy']:
                # Red flashing border
                cv2.rectangle(frame, (0, 110), (w, h), (0, 0, 255), 8)
                # Big alert text
                cv2.rectangle(frame, (w // 2 - 200, h // 2 - 50),
                              (w // 2 + 200, h // 2 + 50), (0, 0, 255), -1)
                cv2.putText(frame, "!! DROWSINESS ALERT !!",
                            (w // 2 - 190, h // 2 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)

            cv2.putText(frame, "Press 'q' to quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            cv2.imshow("FocusGuard AI - Eye State Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        vision.release()
        print(f"\n[STATS] Total blinks detected: {eye_detector.total_blinks}")
        print("[DONE] Test completed ✓\n")


if __name__ == "__main__":
    run_test()