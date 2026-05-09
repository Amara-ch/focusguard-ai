import cv2
import numpy as np


class HeadPoseDetector:
    NOSE_TIP = 1
    CHIN = 152
    LEFT_EYE_CORNER = 33
    RIGHT_EYE_CORNER = 263
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291

    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1)
    ], dtype=np.float64)

    def __init__(self, yaw_threshold=20.0, pitch_threshold=15.0,
                 distract_frames=15):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.distract_frames = distract_frames
        self.distract_counter = 0
        self.is_distracted = False
        self.total_distractions = 0
        self.event_logged = False
        print('[HeadPoseDetector] Initialized OK')

    def _get_2d_points(self, landmarks, w, h):
        ids = [self.NOSE_TIP, self.CHIN, self.LEFT_EYE_CORNER,
               self.RIGHT_EYE_CORNER, self.LEFT_MOUTH_CORNER,
               self.RIGHT_MOUTH_CORNER]
        pts = []
        for i in ids:
            lm = landmarks.landmark[i]
            pts.append((lm.x * w, lm.y * h))
        return np.array(pts, dtype=np.float64)

    def analyze(self, landmarks, w, h):
        if landmarks is None:
            return {
                'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                'direction': 'NO FACE', 'looking_away': False,
                'is_distracted': False,
                'total_distractions': self.total_distractions
            }

        image_pts = self._get_2d_points(landmarks, w, h)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rot_vec, trans_vec = cv2.solvePnP(
            self.MODEL_POINTS, image_pts, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                'direction': 'CALC FAIL', 'looking_away': False,
                'is_distracted': False,
                'total_distractions': self.total_distractions
            }

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
        pitch, yaw, roll = euler.flatten()[:3]

        if pitch > 0:
            pitch = pitch - 180
        else:
            pitch = pitch + 180

        if yaw < -self.yaw_threshold:
            direction = 'LOOKING LEFT'
            looking_away = True
        elif yaw > self.yaw_threshold:
            direction = 'LOOKING RIGHT'
            looking_away = True
        elif pitch < -self.pitch_threshold:
            direction = 'LOOKING DOWN'
            looking_away = True
        elif pitch > self.pitch_threshold:
            direction = 'LOOKING UP'
            looking_away = True
        else:
            direction = 'FORWARD'
            looking_away = False

        if looking_away:
            self.distract_counter += 1
            if self.distract_counter >= self.distract_frames:
                self.is_distracted = True
                if not self.event_logged:
                    self.total_distractions += 1
                    self.event_logged = True
        else:
            self.distract_counter = 0
            self.is_distracted = False
            self.event_logged = False

        return {
            'yaw': yaw, 'pitch': pitch, 'roll': roll,
            'direction': direction, 'looking_away': looking_away,
            'is_distracted': self.is_distracted,
            'total_distractions': self.total_distractions
        }

    def draw_pose(self, frame, result):
        return frame
