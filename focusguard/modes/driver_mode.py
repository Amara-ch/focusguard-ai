import cv2
import time
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from focusguard.core.vision_engine import VisionEngine
from focusguard.models.eye_state import EyeStateDetector
from focusguard.models.yawn_detector import YawnDetector, SoundAlarm
from focusguard.models.phone_detector import PhoneDetector
from focusguard.models.head_pose import HeadPoseDetector
from focusguard.models.object_distraction import DistractionObjectDetector


# ============ CAMERA SETTINGS ============
CAM_WIDTH = 1280       # Webcam capture width
CAM_HEIGHT = 720       # Webcam capture height
DISPLAY_WIDTH = 1280   # Window display width
# ==========================================


class DriverMode:
    def __init__(self):
        print('=' * 55)
        print('FocusGuard AI - DRIVER MODE (FULL SUITE)')
        print('=' * 55)

        self.vision = VisionEngine()
        self.eye_det = EyeStateDetector(ear_threshold=0.21, drowsy_frames=20)
        self.yawn_det = YawnDetector(mar_threshold=0.40, yawn_min_frames=8)
        self.phone_det = PhoneDetector(confidence=0.45, alert_frames=6)
        self.head_det = HeadPoseDetector(yaw_threshold=20, pitch_threshold=15,
                                         distract_frames=15)
        self.obj_det = DistractionObjectDetector(confidence=0.40, alert_frames=8, mode='driver')
        self.alarm = SoundAlarm()

        self.session_start = datetime.now()
        self.event_log = []
        self.safety_score = 100.0

    def log_event(self, event_type, details=''):
        self.event_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': event_type,
            'details': details
        })
        print('[EVENT] ' + event_type + ' - ' + details)
        if event_type == 'DROWSINESS':
            self.safety_score -= 5
        elif event_type == 'PHONE':
            self.safety_score -= 4
        elif event_type == 'LOOKING_AWAY':
            self.safety_score -= 3
        elif event_type == 'YAWN':
            self.safety_score -= 2
        elif event_type == 'DISTRACT_OBJECT':
            self.safety_score -= 2
        self.safety_score = max(0.0, self.safety_score)

    def get_safety_grade(self):
        s = self.safety_score
        if s >= 90: return 'A (Excellent)'
        if s >= 75: return 'B (Good)'
        if s >= 60: return 'C (Fair)'
        if s >= 40: return 'D (Poor)'
        return 'F (Dangerous)'

    def save_report(self):
        os.makedirs('reports', exist_ok=True)
        end = datetime.now()
        duration = (end - self.session_start).total_seconds()
        report = {
            'mode': 'DRIVER',
            'session_start': self.session_start.isoformat(),
            'session_end': end.isoformat(),
            'duration_seconds': round(duration, 2),
            'safety_score': round(self.safety_score, 1),
            'safety_grade': self.get_safety_grade(),
            'stats': {
                'total_blinks': self.eye_det.total_blinks,
                'total_yawns': self.yawn_det.total_yawns,
                'phone_events': self.phone_det.total_phone_events,
                'looking_away_events': self.head_det.total_distractions,
                'object_distraction_events': self.obj_det.total_events
            },
            'event_log': self.event_log
        }
        fname = 'reports/driver_session_' + end.strftime('%Y%m%d_%H%M%S') + '.json'
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        print('\n[REPORT] Saved to: ' + fname)
        return fname, report

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print('[ERROR] No webcam!')
            return

        # Set HIGH resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('[CAMERA] Resolution: %dx%d' % (actual_w, actual_h))

        # Resizable window
        win_name = 'FocusGuard AI - Driver Mode (FULL)'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, DISPLAY_WIDTH, int(DISPLAY_WIDTH * actual_h / actual_w))

        print('\nPress q to quit. Session report will be saved on exit.\n')

        prev_time = time.time()
        fps = 0
        frame_count = 0

        prev_drowsy = False
        prev_yawn_count = 0
        prev_phone_count = 0
        prev_head_count = 0
        prev_obj_count = 0
        last_alarm_state = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                landmarks, face_found = self.vision.process_frame(frame)
                eye_r = self.eye_det.analyze(landmarks, w, h)
                yawn_r = self.yawn_det.analyze(landmarks, w, h)
                phone_r = self.phone_det.analyze(frame)
                head_r = self.head_det.analyze(landmarks, w, h)
                obj_r = self.obj_det.analyze(frame)

                frame = self.eye_det.draw_eye_status(frame, eye_r)
                frame = self.yawn_det.draw_mouth_status(frame, yawn_r)
                frame = self.phone_det.draw_boxes(frame, phone_r)
                frame = self.obj_det.draw_boxes(frame, obj_r)

                if eye_r['is_drowsy'] and not prev_drowsy:
                    self.log_event('DROWSINESS', 'Eyes closed sustained')
                prev_drowsy = eye_r['is_drowsy']

                if yawn_r['total_yawns'] > prev_yawn_count:
                    self.log_event('YAWN')
                    prev_yawn_count = yawn_r['total_yawns']

                if phone_r['total_events'] > prev_phone_count:
                    self.log_event('PHONE', 'Phone detected')
                    prev_phone_count = phone_r['total_events']

                if head_r['total_distractions'] > prev_head_count:
                    self.log_event('LOOKING_AWAY', head_r['direction'])
                    prev_head_count = head_r['total_distractions']

                if obj_r['total_events'] > prev_obj_count:
                    self.log_event('DISTRACT_OBJECT',
                                   ', '.join(obj_r['object_names']))
                    prev_obj_count = obj_r['total_events']

                frame_count += 1
                if time.time() - prev_time >= 1.0:
                    fps = frame_count / (time.time() - prev_time)
                    frame_count = 0
                    prev_time = time.time()

                # Scaled font size for higher resolution
                fs1 = w / 1280.0   # font scale factor for 720p baseline
                t1 = max(1, int(fs1 * 2))

                cv2.rectangle(frame, (0, 0), (w, int(220 * fs1)), (30, 30, 30), -1)
                cv2.putText(frame, 'FocusGuard AI - DRIVER MODE',
                            (10, int(35 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9 * fs1, (0, 255, 255), t1)

                ear_color = (0, 0, 255) if eye_r['eyes_closed'] else (0, 255, 0)
                y1 = int(70 * fs1)
                cv2.putText(frame, 'EAR:%.2f' % eye_r['ear'],
                            (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * fs1, ear_color, t1)
                cv2.putText(frame, 'Eyes:' + ('CLOSE' if eye_r['eyes_closed'] else 'OPEN'),
                            (int(160 * fs1), y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, ear_color, t1)
                cv2.putText(frame, 'Blinks:%d' % eye_r['blinks'],
                            (int(340 * fs1), y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 255), t1)
                cv2.putText(frame, 'FPS:%.1f' % fps,
                            (int(500 * fs1), y1), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 0), t1)

                mar_color = (0, 165, 255) if yawn_r['mouth_open'] else (0, 255, 0)
                y2 = int(105 * fs1)
                cv2.putText(frame, 'MAR:%.2f' % yawn_r['mar'],
                            (10, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * fs1, mar_color, t1)
                cv2.putText(frame, 'Yawns:%d' % yawn_r['total_yawns'],
                            (int(160 * fs1), y2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 255), t1)
                cv2.putText(frame, 'Phone:%d' % phone_r['total_events'],
                            (int(340 * fs1), y2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 255), t1)

                hd_color = (0, 0, 255) if head_r['looking_away'] else (0, 255, 0)
                y3 = int(140 * fs1)
                cv2.putText(frame, 'Head: ' + head_r['direction'],
                            (10, y3), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * fs1, hd_color, t1)
                cv2.putText(frame, 'AwayEv:%d' % head_r['total_distractions'],
                            (int(340 * fs1), y3), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 255), t1)

                obj_color = (0, 0, 255) if obj_r['is_distracted'] else ((0, 165, 255) if obj_r['object_visible'] else (0, 255, 0))
                obj_text = ','.join(obj_r['object_names'][:3]) if obj_r['object_names'] else 'none'
                y4 = int(175 * fs1)
                cv2.putText(frame, 'Obj: ' + obj_text,
                            (10, y4), cv2.FONT_HERSHEY_SIMPLEX, 0.65 * fs1, obj_color, t1)
                cv2.putText(frame, 'ObjEv:%d' % obj_r['total_events'],
                            (int(340 * fs1), y4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 255), t1)

                score = self.safety_score
                if score >= 75: sc = (0, 255, 0)
                elif score >= 50: sc = (0, 200, 255)
                else: sc = (0, 0, 255)
                cv2.putText(frame, 'SAFETY: %.0f/100  Grade: %s' % (
                    score, self.get_safety_grade()),
                            (10, int(210 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8 * fs1, sc, t1)

                alerts = []
                if eye_r['is_drowsy']: alerts.append('!! DROWSINESS !!')
                if yawn_r['is_yawning']: alerts.append('!! YAWN !!')
                if phone_r['phone_detected']: alerts.append('!! PHONE USAGE !!')
                if head_r['is_distracted']: alerts.append('!! EYES OFF ROAD !!')
                if obj_r['is_distracted']: alerts.append('!! DISTRACTION OBJECT !!')

                alarm_should_play = len(alerts) > 0
                if alarm_should_play and not last_alarm_state:
                    print('[ALARM] STARTING -> ' + ' | '.join(alerts))
                    self.alarm.play()
                elif not alarm_should_play and last_alarm_state:
                    print('[ALARM] STOPPING')
                    self.alarm.stop()
                last_alarm_state = alarm_should_play

                if alerts:
                    cv2.rectangle(frame, (0, int(220 * fs1)), (w, h), (0, 0, 255), 6)
                    y_pos = h // 2 - (len(alerts) * 40)
                    for a in alerts:
                        cv2.rectangle(frame, (w // 2 - 280, y_pos),
                                      (w // 2 + 280, y_pos + 60),
                                      (0, 0, 255), -1)
                        cv2.putText(frame, a, (w // 2 - 250, y_pos + 42),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                    (255, 255, 255), 2)
                        y_pos += 70

                cv2.putText(frame, 'Press q to quit and save report',
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (200, 200, 200), 1)
                cv2.imshow(win_name, frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.alarm.cleanup()
            cap.release()
            cv2.destroyAllWindows()
            self.vision.release()
            fname, report = self.save_report()
            print('\n========== SESSION SUMMARY ==========')
            print('Duration : %.1f seconds' % report['duration_seconds'])
            print('Safety   : %.0f / 100 (%s)' % (
                report['safety_score'], report['safety_grade']))
            print('Blinks   : %d' % report['stats']['total_blinks'])
            print('Yawns    : %d' % report['stats']['total_yawns'])
            print('Phone    : %d events' % report['stats']['phone_events'])
            print('LookAway : %d events' % report['stats']['looking_away_events'])
            print('Objects  : %d events' % report['stats']['object_distraction_events'])
            print('=====================================\n')


if __name__ == '__main__':
    DriverMode().run()
