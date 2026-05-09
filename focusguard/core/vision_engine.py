"""
FocusGuard AI - Vision Engine
==============================
Core computer vision module for face detection and landmark extraction.
Uses MediaPipe FaceMesh to extract 468 facial landmarks in real-time.

Author: humbleunitydev
Project: FocusGuard AI - Unified Attention Monitoring System
"""

import cv2
import mediapipe as mp
import time
import numpy as np


class VisionEngine:
    """
    Core vision engine that handles face detection and landmark extraction.
    This is the foundation layer for all higher-level analysis (drowsiness,
    yawn, gaze, etc.).
    """

    def __init__(self, max_faces=1, detection_confidence=0.5, tracking_confidence=0.5):
        """
        Initialize the MediaPipe FaceMesh model.

        Args:
            max_faces: Maximum number of faces to detect (1 for driver/student)
            detection_confidence: Minimum confidence for face detection [0.0-1.0]
            tracking_confidence: Minimum confidence for landmark tracking [0.0-1.0]
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Initialize FaceMesh with refined landmarks (better for eyes/lips)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # Drawing specifications (custom green dots)
        self.dot_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1
        )

        print("[VisionEngine] Initialized successfully ✓")

    def process_frame(self, frame):
        """
        Process a single video frame and extract face landmarks.

        Args:
            frame: BGR image from cv2 (webcam frame)

        Returns:
            tuple: (landmarks, face_detected_bool)
                landmarks: list of 468 (x,y,z) normalized points or None
                face_detected_bool: True if face found, False otherwise
        """
        # MediaPipe expects RGB, OpenCV gives BGR
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Speed boost

        # Run inference
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            # Return the first face's landmarks
            return results.multi_face_landmarks[0], True
        return None, False

    def draw_landmarks(self, frame, landmarks):
        """
        Draw facial landmarks (mesh) on the frame.

        Args:
            frame: BGR image to draw on
            landmarks: MediaPipe face landmarks object
        """
        if landmarks is None:
            return frame

        # Draw the face mesh tesselation (light green network)
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # Highlight eyes (yellow contour)
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
        )

        return frame

    def release(self):
        """Cleanup resources."""
        self.face_mesh.close()
        print("[VisionEngine] Released ✓")


# ============================================================
# STANDALONE TEST RUNNER
# ============================================================
def run_test():
    """
    Standalone test: opens webcam and shows face landmarks live.
    Press 'q' to quit.
    """
    print("\n" + "=" * 50)
    print("FocusGuard AI - Vision Engine Test")
    print("=" * 50)
    print("Initializing webcam...")

    # Open default webcam (index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows

    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        print("        Try changing index to 1 or 2 in the code.")
        return

    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[OK] Webcam opened ✓")
    print("[INFO] Press 'q' to quit\n")

    # Initialize vision engine
    engine = VisionEngine()

    # FPS tracking
    prev_time = time.time()
    fps = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame, retrying...")
                continue

            # Mirror the frame (more natural for selfie view)
            frame = cv2.flip(frame, 1)

            # Process frame for face landmarks
            landmarks, face_found = engine.process_frame(frame)

            # Draw landmarks if face detected
            frame = engine.draw_landmarks(frame, landmarks)

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            # Display status overlay
            status_text = "FACE DETECTED" if face_found else "NO FACE"
            status_color = (0, 255, 0) if face_found else (0, 0, 255)

            cv2.putText(frame, f"FocusGuard AI - Vision Test",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Status: {status_text}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        status_color, 2)
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2)
            cv2.putText(frame, "Press 'q' to quit",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

            # Show window
            cv2.imshow("FocusGuard AI - Vision Engine Test", frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] Quit signal received")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.release()
        print("[DONE] Test completed successfully ✓\n")


if __name__ == "__main__":
    run_test()