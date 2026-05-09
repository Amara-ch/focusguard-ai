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


CAM_WIDTH = 1280
CAM_HEIGHT = 720
DISPLAY_WIDTH = 1280


class StudentMode:
    '''
    Student Mode for online learning / exam monitoring.
    Tracks focus, distractions, and absence from screen.
    '''

    def __init__(self):
        print('=' * 55)
        print('FocusGuard AI - STUDENT MODE')
        print('=' * 55)

        self.vision = VisionEngine(max_faces=3)  # detect up to 3 people
        self.eye_det = EyeStateDetector(ear_threshold=0.21, drowsy_frames=20)
        self.yawn_det = YawnDetector(mar_threshold=0.40, yawn_min_frames=8)
        self.phone_det = PhoneDetector(confidence=0.45, alert_frames=6)
        self.head_det = HeadPoseDetector(yaw_threshold=20, pitch_threshold=15,
                                         distract_frames=15)
        self.obj_det = DistractionObjectDetector(confidence=0.40, alert_frames=8, mode='student')
        self.alarm = SoundAlarm()

        self.session_start = datetime.now()
        self.event_log = []
        self.focus_score = 100.0

        # Focus tracking
        self.focused_seconds = 0.0
        self.distracted_seconds = 0.0
        self.absent_seconds = 0.0
        self.last_tick = time.time()

        # No-face tracking
        self.no_face_counter = 0
        self.absent_alert = False
        self.absent_logged = False
        self.total_absences = 0

        # Multi-face tracking (cheating during exam)
        self.multi_face_counter = 0
        self.multi_face_alert = False
        self.multi_face_logged = False
        self.total_multi_face = 0

    def log_event(self, event_type, details=''):
        self.event_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'type': event_type,
            'details': details
        })
        print('[EVENT] ' + event_type + ' - ' + details)
        if event_type == 'PHONE':
            self.focus_score -= 5
        elif event_type == 'LOOKING_AWAY':
            self.focus_score -= 3
        elif event_type == 'DROWSINESS':
            self.focus_score -= 4
        elif event_type == 'YAWN':
            self.focus_score -= 1
        elif event_type == 'DISTRACT_OBJECT':
            self.focus_score -= 2
        elif event_type == 'ABSENT':
            self.focus_score -= 6
        elif event_type == 'MULTI_FACE':
            self.focus_score -= 8
        self.focus_score = max(0.0, self.focus_score)

    def get_focus_grade(self):
        s = self.focus_score
        if s >= 90: return 'A+ (Highly Focused)'
        if s >= 75: return 'A (Focused)'
        if s >= 60: return 'B (Average)'
        if s >= 40: return 'C (Distracted)'
        return 'D (Very Distracted)'

    def save_report(self):
        os.makedirs('reports', exist_ok=True)
        end = datetime.now()
        duration = (end - self.session_start).total_seconds()
        total_tracked = self.focused_seconds + self.distracted_seconds + self.absent_seconds
        focus_pct = (self.focused_seconds / total_tracked * 100) if total_tracked > 0 else 0
        distract_pct = (self.distracted_seconds / total_tracked * 100) if total_tracked > 0 else 0
        absent_pct = (self.absent_seconds / total_tracked * 100) if total_tracked > 0 else 0

        report = {
            'mode': 'STUDENT',
            'session_start': self.session_start.isoformat(),
            'session_end': end.isoformat(),
            'duration_seconds': round(duration, 2),
            'focus_score': round(self.focus_score, 1),
            'focus_grade': self.get_focus_grade(),
            'time_breakdown': {
                'focused_seconds': round(self.focused_seconds, 1),
                'distracted_seconds': round(self.distracted_seconds, 1),
                'absent_seconds': round(self.absent_seconds, 1),
                'focused_pct': round(focus_pct, 1),
                'distracted_pct': round(distract_pct, 1),
                'absent_pct': round(absent_pct, 1)
            },
            'stats': {
                'total_blinks': self.eye_det.total_blinks,
                'total_yawns': self.yawn_det.total_yawns,
                'phone_events': self.phone_det.total_phone_events,
                'looking_away_events': self.head_det.total_distractions,
                'object_distraction_events': self.obj_det.total_events,
                'absence_events': self.total_absences,
                'multi_face_events': self.total_multi_face
            },
            'event_log': self.event_log
        }
        fname = 'reports/student_session_' + end.strftime('%Y%m%d_%H%M%S') + '.json'
        with open(fname, 'w') as f:
            json.dump(report, f, indent=2)
        print('\n[REPORT] Saved to: ' + fname)
        return fname, report

    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print('[ERROR] No webcam!')
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print('[CAMERA] Resolution: %dx%d' % (actual_w, actual_h))

        win_name = 'FocusGuard AI - Student Mode'
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, DISPLAY_WIDTH,
                         int(DISPLAY_WIDTH * actual_h / actual_w))

        print('\nPress q to quit and save session report.\n')

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

                # Process face mesh - get up to multiple faces for cheating detection
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.vision.face_mesh.process(rgb)
                num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
                landmarks = results.multi_face_landmarks[0] if num_faces > 0 else None
                face_found = num_faces > 0

                eye_r = self.eye_det.analyze(landmarks, w, h)
                yawn_r = self.yawn_det.analyze(landmarks, w, h)
                phone_r = self.phone_det.analyze(frame)
                head_r = self.head_det.analyze(landmarks, w, h)
                obj_r = self.obj_det.analyze(frame)

                frame = self.eye_det.draw_eye_status(frame, eye_r)
                frame = self.yawn_det.draw_mouth_status(frame, yawn_r)
                frame = self.phone_det.draw_boxes(frame, phone_r)
                frame = self.obj_det.draw_boxes(frame, obj_r)

                # ========== ABSENCE DETECTION ==========
                if not face_found:
                    self.no_face_counter += 1
                    if self.no_face_counter >= 30:  # ~3-5 seconds
                        self.absent_alert = True
                        if not self.absent_logged:
                            self.total_absences += 1
                            self.log_event('ABSENT', 'Student left the seat')
                            self.absent_logged = True
                else:
                    self.no_face_counter = 0
                    self.absent_alert = False
                    self.absent_logged = False

                # ========== MULTI-FACE DETECTION (cheating) ==========
                if num_faces >= 2:
                    self.multi_face_counter += 1
                    if self.multi_face_counter >= 15:
                        self.multi_face_alert = True
                        if not self.multi_face_logged:
                            self.total_multi_face += 1
                            self.log_event('MULTI_FACE',
                                           '%d faces detected (cheating?)' % num_faces)
                            self.multi_face_logged = True
                else:
                    self.multi_face_counter = 0
                    self.multi_face_alert = False
                    self.multi_face_logged = False

                # ========== TIME TRACKING ==========
                now = time.time()
                dt = now - self.last_tick
                self.last_tick = now

                is_distracted_now = (eye_r['is_drowsy'] or yawn_r['is_yawning'] or
                                     phone_r['phone_detected'] or head_r['is_distracted'] or
                                     obj_r['is_distracted'] or self.multi_face_alert)

                if self.absent_alert:
                    self.absent_seconds += dt
                elif is_distracted_now:
                    self.distracted_seconds += dt
                elif face_found:
                    self.focused_seconds += dt

                # ========== EVENT LOGGING ==========
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

                # ========== FPS ==========
                frame_count += 1
                if time.time() - prev_time >= 1.0:
                    fps = frame_count / (time.time() - prev_time)
                    frame_count = 0
                    prev_time = time.time()

                # ========== UI ==========
                fs1 = w / 1280.0
                t1 = max(1, int(fs1 * 2))

                cv2.rectangle(frame, (0, 0), (w, int(245 * fs1)), (30, 30, 30), -1)
                cv2.putText(frame, 'FocusGuard AI - STUDENT MODE',
                            (10, int(35 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9 * fs1, (0, 255, 200), t1)

                # Faces
                face_color = (0, 255, 0) if num_faces == 1 else ((0, 0, 255) if num_faces > 1 else (200, 200, 200))
                cv2.putText(frame, 'Faces: %d' % num_faces,
                            (10, int(70 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, face_color, t1)
                cv2.putText(frame, 'FPS:%.1f' % fps,
                            (int(500 * fs1), int(70 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65 * fs1, (255, 255, 0), t1)

                # Eyes / Mouth
                ear_color = (0, 0, 255) if eye_r['eyes_closed'] else (0, 255, 0)
                cv2.putText(frame, 'Eyes:' + ('CLOSE' if eye_r['eyes_closed'] else 'OPEN'),
                            (10, int(105 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6 * fs1, ear_color, t1)
                cv2.putText(frame, 'Blinks:%d' % eye_r['blinks'],
                            (int(180 * fs1), int(105 * fs1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs1, (255, 255, 255), t1)
                cv2.putText(frame, 'Yawns:%d' % yawn_r['total_yawns'],
                            (int(340 * fs1), int(105 * fs1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs1, (255, 255, 255), t1)

                # Head/Phone/Obj
                hd_color = (0, 0, 255) if head_r['looking_away'] else (0, 255, 0)
                cv2.putText(frame, 'Head:' + head_r['direction'],
                            (10, int(140 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6 * fs1, hd_color, t1)
                cv2.putText(frame, 'Phone:%d' % phone_r['total_events'],
                            (int(340 * fs1), int(140 * fs1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs1, (255, 255, 255), t1)
                cv2.putText(frame, 'Absent:%d' % self.total_absences,
                            (int(500 * fs1), int(140 * fs1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs1, (255, 255, 255), t1)

                # Time breakdown
                total_t = self.focused_seconds + self.distracted_seconds + self.absent_seconds
                if total_t > 0:
                    fp = self.focused_seconds / total_t * 100
                    dp = self.distracted_seconds / total_t * 100
                    ap = self.absent_seconds / total_t * 100
                else:
                    fp = dp = ap = 0
                cv2.putText(frame, 'Focused:%.0f%%  Distracted:%.0f%%  Absent:%.0f%%' % (fp, dp, ap),
                            (10, int(175 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55 * fs1, (200, 255, 200), t1)

                # Focus Score
                score = self.focus_score
                if score >= 75: sc = (0, 255, 0)
                elif score >= 50: sc = (0, 200, 255)
                else: sc = (0, 0, 255)
                cv2.putText(frame, 'FOCUS: %.0f/100  Grade: %s' % (
                    score, self.get_focus_grade()),
                            (10, int(215 * fs1)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75 * fs1, sc, t1)

                # ========== ALERTS ==========
                alerts = []
                if self.absent_alert:
                    alerts.append('!! STUDENT ABSENT !!')
                if self.multi_face_alert:
                    alerts.append('!! MULTIPLE FACES (CHEATING?) !!')
                if eye_r['is_drowsy']:
                    alerts.append('!! SLEEPING !!')
                if phone_r['phone_detected']:
                    alerts.append('!! PHONE DISTRACTION !!')
                if head_r['is_distracted']:
                    alerts.append('!! LOOKING AWAY !!')
                if yawn_r['is_yawning']:
                    alerts.append('!! YAWNING !!')
                if obj_r['is_distracted']:
                    alerts.append('!! OBJECT DISTRACTION !!')

                alarm_should_play = len(alerts) > 0
                if alarm_should_play and not last_alarm_state:
                    print('[ALARM] -> ' + ' | '.join(alerts))
                    self.alarm.play()
                elif not alarm_should_play and last_alarm_state:
                    self.alarm.stop()
                last_alarm_state = alarm_should_play

                if alerts:
                    cv2.rectangle(frame, (0, int(245 * fs1)), (w, h), (0, 0, 255), 6)
                    y_pos = h // 2 - (len(alerts) * 35)
                    for a in alerts:
                        cv2.rectangle(frame, (w // 2 - 300, y_pos),
                                      (w // 2 + 300, y_pos + 55),
                                      (0, 0, 255), -1)
                        cv2.putText(frame, a, (w // 2 - 280, y_pos + 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                    (255, 255, 255), 2)
                        y_pos += 65

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
            print('Duration   : %.1f sec' % report['duration_seconds'])
            print('Focus Score: %.0f / 100 (%s)' % (
                report['focus_score'], report['focus_grade']))
            tb = report['time_breakdown']
            print('Focused    : %.1f sec (%.0f%%)' % (tb['focused_seconds'], tb['focused_pct']))
            print('Distracted : %.1f sec (%.0f%%)' % (tb['distracted_seconds'], tb['distracted_pct']))
            print('Absent     : %.1f sec (%.0f%%)' % (tb['absent_seconds'], tb['absent_pct']))
            print('Phone      : %d  Yawns: %d  Away: %d' % (
                report['stats']['phone_events'], report['stats']['total_yawns'],
                report['stats']['looking_away_events']))
            print('Absences   : %d  MultiFace: %d' % (
                report['stats']['absence_events'], report['stats']['multi_face_events']))
            print('=====================================\n')


if __name__ == '__main__':
    StudentMode().run()
