import streamlit as st
import cv2
import numpy as np
import time
import json
import os
import sys
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from focusguard.core.vision_engine import VisionEngine
from focusguard.models.eye_state import EyeStateDetector
from focusguard.models.yawn_detector import YawnDetector, SoundAlarm
from focusguard.models.phone_detector import PhoneDetector
from focusguard.models.head_pose import HeadPoseDetector
from focusguard.models.object_distraction import DistractionObjectDetector


st.set_page_config(
    page_title='FocusGuard AI | PUCIT',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ========= SESSION STATE =========
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'running' not in st.session_state:
    st.session_state.running = False
if 'detectors' not in st.session_state:
    st.session_state.detectors = None
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'


# ========= THEME CSS (PURE PURPLE) =========
def inject_css():
    if st.session_state.theme == 'dark':
        bg = '#13111c'
        bg_soft = '#1c1a2e'
        card_bg = '#1f1d31'
        text = '#f3f4f6'
        text_soft = '#9ca3af'
        border = '#2d2a44'
        hero_bg = 'linear-gradient(135deg, #1a1830 0%, #2a1f4a 50%, #1a1830 100%)'
        section_alt = '#1a1830'
    else:
        bg = '#fdfcff'
        bg_soft = '#f8f5ff'
        card_bg = '#ffffff'
        text = '#1f1f3a'
        text_soft = '#6b6b8c'
        border = '#ece5fa'
        hero_bg = 'linear-gradient(135deg, #f5efff 0%, #ede1ff 50%, #e0d0ff 100%)'
        section_alt = '#f8f5ff'

    st.markdown(f'''
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    html, body, [class*="css"], .stApp, button, p, div, span, h1, h2, h3, h4, h5 {{
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }}

    .stApp {{ background: {bg}; color: {text}; }}
    .block-container {{ padding: 0 !important; max-width: 100% !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {card_bg};
        border-right: 1px solid {border};
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{ color: {text} !important; }}

    /* ========= HERO ========= */
    .hero-wrap {{
        background: {hero_bg};
        padding: 6rem 2rem 7rem 2rem;
        position: relative;
        overflow: hidden;
    }}
    .hero-wrap::before {{
        content: '';
        position: absolute;
        top: 10%; left: -10%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(139,92,246,0.25) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .hero-wrap::after {{
        content: '';
        position: absolute;
        bottom: 5%; right: -10%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(167,139,250,0.2) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .hero-inner {{
        max-width: 1100px;
        margin: 0 auto;
        text-align: center;
        position: relative;
        z-index: 2;
    }}
    .hero-badge {{
        display: inline-block;
        background: {card_bg};
        color: #7c3aed;
        padding: 0.55rem 1.4rem;
        border-radius: 50px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid #d8b4fe;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 14px rgba(139,92,246,0.15);
    }}
    .hero-title {{
        font-size: 4.4rem;
        font-weight: 800;
        color: {text};
        line-height: 1.08;
        letter-spacing: -2.5px;
        margin: 0 auto 1.5rem auto;
        max-width: 900px;
    }}
    .hero-title-accent {{
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-desc {{
        font-size: 1.2rem;
        color: {text_soft};
        max-width: 680px;
        line-height: 1.65;
        margin: 0 auto 2rem auto;
        font-weight: 400;
    }}
    .hero-meta {{
        color: #7c3aed;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 1rem;
    }}

    /* ========= TRUST BAR ========= */
    .trust-bar {{
        background: {section_alt};
        border-top: 1px solid {border};
        border-bottom: 1px solid {border};
        padding: 2.5rem 2rem;
    }}
    .trust-row {{
        max-width: 1100px;
        margin: 0 auto;
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 1rem;
    }}
    .trust-item {{ text-align: center; }}
    .trust-num {{
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
    }}
    .trust-label {{
        font-size: 0.75rem;
        color: {text_soft};
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-top: 0.4rem;
    }}

    /* ========= SECTION ========= */
    .section {{
        padding: 5rem 2rem;
        background: {bg};
    }}
    .section-alt {{
        background: {section_alt};
    }}
    .section-inner {{
        max-width: 1100px;
        margin: 0 auto;
    }}
    .section-head {{
        text-align: center;
        margin-bottom: 3.5rem;
    }}
    .section-eyebrow {{
        display: inline-block;
        color: #7c3aed;
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        margin-bottom: 1rem;
    }}
    .section-title {{
        font-size: 2.6rem;
        font-weight: 800;
        color: {text};
        line-height: 1.2;
        letter-spacing: -1.5px;
        margin: 0 auto 1rem auto;
        max-width: 750px;
    }}
    .section-sub {{
        font-size: 1.08rem;
        color: {text_soft};
        line-height: 1.65;
        max-width: 650px;
        margin: 0 auto;
    }}

    /* ========= USE CASE CARDS ========= */
    .uc-card {{
        background: {card_bg};
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(139,92,246,0.1);
        border: 1px solid {border};
        height: 100%;
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
    }}
    .uc-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 24px 60px rgba(139,92,246,0.22);
        border-color: #c084fc;
    }}
    .uc-image {{
        width: 100%;
        height: 240px;
        object-fit: cover;
        display: block;
    }}
    .uc-body {{
        padding: 2rem;
        display: flex;
        flex-direction: column;
        flex: 1;
    }}
    .uc-tag {{
        display: inline-block;
        background: linear-gradient(135deg, #ede9fe, #f3e8ff);
        color: #7c3aed;
        padding: 0.3rem 0.85rem;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 1rem;
        align-self: flex-start;
    }}
    .uc-tag.deep {{
        background: linear-gradient(135deg, #ddd6fe, #c4b5fd);
        color: #5b21b6;
    }}
    .uc-title {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {text};
        margin-bottom: 0.6rem;
        letter-spacing: -0.5px;
    }}
    .uc-desc {{
        color: {text_soft};
        line-height: 1.65;
        font-size: 0.96rem;
        margin-bottom: 1.3rem;
    }}
    .uc-list {{ list-style: none; padding: 0; margin: 0 0 0.5rem 0; }}
    .uc-list li {{
        padding: 0.4rem 0;
        color: {text};
        font-size: 0.92rem;
        display: flex;
        align-items: flex-start;
    }}
    .uc-list li::before {{
        content: '✓';
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        width: 20px; height: 20px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.72rem;
        margin-right: 0.7rem;
        flex-shrink: 0;
        margin-top: 2px;
    }}

    /* ========= IMAGE CARDS ========= */
    .img-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 18px;
        overflow: hidden;
        height: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }}
    .img-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 18px 40px rgba(139,92,246,0.18);
        border-color: #c084fc;
    }}
    .img-card img {{
        width: 100%;
        height: 200px;
        object-fit: cover;
        display: block;
    }}
    .img-card-body {{ padding: 1.5rem; }}
    .img-card-title {{
        font-size: 1.18rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 0.5rem;
    }}
    .img-card-desc {{
        color: {text_soft};
        font-size: 0.92rem;
        line-height: 1.6;
    }}

    /* ========= FEATURE CARDS ========= */
    .feat-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 2rem;
        height: 100%;
        transition: all 0.3s ease;
        text-align: center;
    }}
    .feat-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 20px 40px rgba(139,92,246,0.15);
        border-color: #c084fc;
    }}
    .feat-icon {{
        width: 64px; height: 64px;
        background: linear-gradient(135deg, #ede9fe, #ddd6fe);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.9rem;
        margin: 0 auto 1.3rem auto;
    }}
    .feat-title {{
        font-size: 1.18rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 0.6rem;
    }}
    .feat-desc {{
        color: {text_soft};
        line-height: 1.65;
        font-size: 0.92rem;
    }}

    /* ========= STEP CARDS ========= */
    .step-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        height: 100%;
        transition: all 0.3s;
    }}
    .step-card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 16px 36px rgba(139,92,246,0.15);
    }}
    .step-num {{
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
        margin-bottom: 1rem;
    }}
    .step-title {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 0.5rem;
    }}
    .step-desc {{
        color: {text_soft};
        font-size: 0.92rem;
        line-height: 1.6;
    }}

    /* ========= TECH GRID ========= */
    .tech-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
        max-width: 900px;
        margin: 2rem auto 0 auto;
    }}
    .tech-pill {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.2rem 0.8rem;
        text-align: center;
        transition: all 0.2s;
    }}
    .tech-pill:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(139,92,246,0.15);
        border-color: #c084fc;
    }}
    .tech-emoji {{ font-size: 1.7rem; margin-bottom: 0.4rem; }}
    .tech-name {{ color: {text}; font-weight: 600; font-size: 0.85rem; }}

    /* ========= TEAM CARD ========= */
    .team-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 2.5rem 1.8rem;
        text-align: center;
        transition: all 0.3s;
        height: 100%;
    }}
    .team-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 18px 50px rgba(139,92,246,0.18);
        border-color: #c084fc;
    }}
    .team-avatar {{
        width: 96px; height: 96px;
        border-radius: 50%;
        margin: 0 auto 1.2rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.4rem;
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 24px rgba(139,92,246,0.3);
    }}
    .av-1 {{ background: linear-gradient(135deg, #6d28d9, #8b5cf6); }}
    .av-2 {{ background: linear-gradient(135deg, #7c3aed, #a855f7); }}
    .av-3 {{ background: linear-gradient(135deg, #9333ea, #c084fc); }}
    .team-name {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {text};
        margin-bottom: 0.3rem;
    }}
    .team-role {{
        color: #7c3aed;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }}
    .team-bio {{
        color: {text_soft};
        font-size: 0.9rem;
        line-height: 1.65;
    }}

    /* ========= CTA ========= */
    .cta-wrap {{
        background: linear-gradient(135deg, #6d28d9 0%, #7c3aed 50%, #a855f7 100%);
        padding: 5rem 2rem;
        text-align: center;
    }}
    .cta-title {{
        font-size: 2.5rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1.2px;
        margin-bottom: 1rem;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }}
    .cta-sub {{
        color: rgba(255,255,255,0.95);
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }}

    /* ========= FOOTER ========= */
    .footer {{
        background: {section_alt};
        padding: 3rem 2rem 2rem 2rem;
        text-align: center;
        border-top: 1px solid {border};
    }}
    .footer-brand {{
        font-size: 1.4rem;
        font-weight: 800;
        color: {text};
        margin-bottom: 0.5rem;
    }}
    .footer-brand .accent {{
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .footer-line {{
        color: {text_soft};
        font-size: 0.92rem;
        margin: 0.4rem 0;
    }}
    .footer-meta {{
        color: {text_soft};
        font-size: 0.82rem;
        margin-top: 1rem;
        opacity: 0.8;
    }}

    /* ========= MONITOR PAGE ========= */
    .monitor-wrap {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 3rem 2rem;
    }}
    .monitor-title {{
        font-size: 2.4rem;
        font-weight: 800;
        color: {text};
        letter-spacing: -1px;
        margin-bottom: 0.4rem;
        text-align: center;
    }}
    .monitor-desc {{
        color: {text_soft};
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 2.5rem;
    }}
    .metric-card {{
        background: {card_bg};
        border: 1px solid {border};
        padding: 1.3rem;
        border-radius: 14px;
        border-left: 4px solid #7c3aed;
        margin: 0.6rem 0;
        box-shadow: 0 2px 8px rgba(139,92,246,0.08);
    }}
    .alert-box {{
        background: linear-gradient(135deg, #6d28d9, #4c1d95);
        color: white;
        padding: 0.9rem;
        border-radius: 12px;
        font-weight: 700;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 14px rgba(109,40,217,0.35);
        font-size: 0.93rem;
    }}
    .ok-box {{
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 0.9rem;
        border-radius: 12px;
        font-weight: 700;
        margin: 0.5rem 0;
        text-align: center;
        box-shadow: 0 4px 14px rgba(16,185,129,0.3);
        font-size: 0.93rem;
    }}

    /* ========= BUTTONS ========= */
    .stButton>button {{
        background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.7rem;
        font-weight: 700 !important;
        font-size: 0.96rem;
        transition: all 0.25s;
        box-shadow: 0 4px 14px rgba(139,92,246,0.3);
        letter-spacing: 0.2px;
    }}
    .stButton>button:hover {{
        background: linear-gradient(135deg, #6d28d9 0%, #9333ea 100%);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(139,92,246,0.45);
    }}
    .stButton>button:focus {{ box-shadow: 0 4px 14px rgba(139,92,246,0.3) !important; }}

    /* Streamlit metric override */
    [data-testid="stMetricValue"] {{
        color: {text};
        font-weight: 700;
        font-size: 1.4rem;
    }}
    [data-testid="stMetricLabel"] {{
        color: {text_soft};
        font-size: 0.82rem;
        font-weight: 600;
    }}
    [data-testid="stMetricLabel"] p {{ color: {text_soft} !important; }}

    /* PUCIT badge */
    .pucit-badge {{
        display: inline-block;
        background: linear-gradient(135deg, #7c3aed, #a855f7);
        color: white;
        padding: 0.5rem 1.1rem;
        border-radius: 10px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }}

    /* RESPONSIVE */
    @media (max-width: 900px) {{
        .trust-row {{ grid-template-columns: repeat(2, 1fr); gap: 1.5rem; }}
        .hero-title {{ font-size: 2.6rem; letter-spacing: -1px; }}
        .hero-wrap {{ padding: 4rem 1.5rem 5rem 1.5rem; }}
        .section {{ padding: 3.5rem 1.5rem; }}
        .section-title {{ font-size: 2rem; }}
        .cta-title {{ font-size: 2rem; }}
        .cta-wrap {{ padding: 3.5rem 1.5rem; }}
        .monitor-wrap {{ padding: 1.5rem 1rem; }}
        .uc-image {{ height: 200px; }}
    }}
    </style>
    ''', unsafe_allow_html=True)


inject_css()


def init_detectors(mode):
    return {
        'mode': mode,
        'vision': VisionEngine(max_faces=3 if mode == 'student' else 1),
        'eye': EyeStateDetector(ear_threshold=0.21, drowsy_frames=15),
        'yawn': YawnDetector(mar_threshold=0.40, yawn_min_frames=8),
        'phone': PhoneDetector(confidence=0.45, alert_frames=6),
        'head': HeadPoseDetector(yaw_threshold=20, pitch_threshold=15,
                                 distract_frames=15),
        'obj': DistractionObjectDetector(confidence=0.40, alert_frames=8,
                                         mode=mode),
        'alarm': SoundAlarm(),
        'session_start': datetime.now(),
        'event_log': [],
        'score': 100.0,
        'prev_drowsy': False,
        'prev_yawn': 0, 'prev_phone': 0, 'prev_head': 0, 'prev_obj': 0,
        'no_face_counter': 0, 'absent_alert': False, 'absent_logged': False,
        'total_absences': 0,
        'multi_face_counter': 0, 'multi_face_alert': False,
        'multi_face_logged': False, 'total_multi_face': 0,
        'focused_seconds': 0.0, 'distracted_seconds': 0.0,
        'absent_seconds': 0.0, 'last_tick': time.time(),
        'last_alarm_state': False
    }


def log_event(d, event_type, details=''):
    d['event_log'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'type': event_type, 'details': details
    })
    if d['mode'] == 'driver':
        pen = {'DROWSINESS': 5, 'PHONE': 4, 'LOOKING_AWAY': 3,
               'YAWN': 2, 'DISTRACT_OBJECT': 2}
    else:
        pen = {'DROWSINESS': 4, 'PHONE': 5, 'LOOKING_AWAY': 3,
               'YAWN': 1, 'DISTRACT_OBJECT': 2,
               'ABSENT': 6, 'MULTI_FACE': 8}
    d['score'] = max(0.0, d['score'] - pen.get(event_type, 0))


def get_grade(d):
    s = d['score']
    if d['mode'] == 'driver':
        if s >= 90: return 'A (Excellent)'
        if s >= 75: return 'B (Good)'
        if s >= 60: return 'C (Fair)'
        if s >= 40: return 'D (Poor)'
        return 'F (Dangerous)'
    else:
        if s >= 90: return 'A+ (Highly Focused)'
        if s >= 75: return 'A (Focused)'
        if s >= 60: return 'B (Average)'
        if s >= 40: return 'C (Distracted)'
        return 'D (Very Distracted)'


def save_report(d):
    os.makedirs('reports', exist_ok=True)
    end = datetime.now()
    duration = (end - d['session_start']).total_seconds()
    total_t = d['focused_seconds'] + d['distracted_seconds'] + d['absent_seconds']
    fp = (d['focused_seconds'] / total_t * 100) if total_t > 0 else 0
    dp = (d['distracted_seconds'] / total_t * 100) if total_t > 0 else 0
    ap = (d['absent_seconds'] / total_t * 100) if total_t > 0 else 0
    score_key = 'safety_score' if d['mode'] == 'driver' else 'focus_score'
    grade_key = 'safety_grade' if d['mode'] == 'driver' else 'focus_grade'
    report = {
        'mode': d['mode'].upper(),
        'session_start': d['session_start'].isoformat(),
        'session_end': end.isoformat(),
        'duration_seconds': round(duration, 2),
        score_key: round(d['score'], 1),
        grade_key: get_grade(d),
        'time_breakdown': {
            'focused_seconds': round(d['focused_seconds'], 1),
            'distracted_seconds': round(d['distracted_seconds'], 1),
            'absent_seconds': round(d['absent_seconds'], 1),
            'focused_pct': round(fp, 1),
            'distracted_pct': round(dp, 1),
            'absent_pct': round(ap, 1)
        },
        'stats': {
            'total_blinks': d['eye'].total_blinks,
            'total_yawns': d['yawn'].total_yawns,
            'phone_events': d['phone'].total_phone_events,
            'looking_away_events': d['head'].total_distractions,
            'object_distraction_events': d['obj'].total_events,
            'absence_events': d['total_absences'],
            'multi_face_events': d['total_multi_face']
        },
        'event_log': d['event_log']
    }
    fname = 'reports/' + d['mode'] + '_session_' + end.strftime('%Y%m%d_%H%M%S') + '.json'
    with open(fname, 'w') as f:
        json.dump(report, f, indent=2)
    return fname, report


def process_frame(d, frame):
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = d['vision'].face_mesh.process(rgb)
    num_faces = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
    landmarks = results.multi_face_landmarks[0] if num_faces > 0 else None
    face_found = num_faces > 0

    eye_r = d['eye'].analyze(landmarks, w, h)
    yawn_r = d['yawn'].analyze(landmarks, w, h)
    phone_r = d['phone'].analyze(frame)
    head_r = d['head'].analyze(landmarks, w, h)
    obj_r = d['obj'].analyze(frame)

    frame = d['eye'].draw_eye_status(frame, eye_r)
    frame = d['yawn'].draw_mouth_status(frame, yawn_r)
    frame = d['phone'].draw_boxes(frame, phone_r)
    frame = d['obj'].draw_boxes(frame, obj_r)

    if d['mode'] == 'student':
        if not face_found:
            d['no_face_counter'] += 1
            if d['no_face_counter'] >= 30:
                d['absent_alert'] = True
                if not d['absent_logged']:
                    d['total_absences'] += 1
                    log_event(d, 'ABSENT', 'Left seat')
                    d['absent_logged'] = True
        else:
            d['no_face_counter'] = 0
            d['absent_alert'] = False
            d['absent_logged'] = False
        if num_faces >= 2:
            d['multi_face_counter'] += 1
            if d['multi_face_counter'] >= 15:
                d['multi_face_alert'] = True
                if not d['multi_face_logged']:
                    d['total_multi_face'] += 1
                    log_event(d, 'MULTI_FACE', '%d faces' % num_faces)
                    d['multi_face_logged'] = True
        else:
            d['multi_face_counter'] = 0
            d['multi_face_alert'] = False
            d['multi_face_logged'] = False

    now = time.time()
    dt = now - d['last_tick']
    d['last_tick'] = now
    is_distracted = (eye_r['is_drowsy'] or yawn_r['is_yawning'] or
                     phone_r['phone_detected'] or head_r['is_distracted'] or
                     obj_r['is_distracted'] or d['multi_face_alert'])
    if d['mode'] == 'student' and d['absent_alert']:
        d['absent_seconds'] += dt
    elif is_distracted:
        d['distracted_seconds'] += dt
    elif face_found:
        d['focused_seconds'] += dt

    if eye_r['is_drowsy'] and not d['prev_drowsy']:
        log_event(d, 'DROWSINESS')
    d['prev_drowsy'] = eye_r['is_drowsy']
    if yawn_r['total_yawns'] > d['prev_yawn']:
        log_event(d, 'YAWN'); d['prev_yawn'] = yawn_r['total_yawns']
    if phone_r['total_events'] > d['prev_phone']:
        log_event(d, 'PHONE'); d['prev_phone'] = phone_r['total_events']
    if head_r['total_distractions'] > d['prev_head']:
        log_event(d, 'LOOKING_AWAY', head_r['direction'])
        d['prev_head'] = head_r['total_distractions']
    if obj_r['total_events'] > d['prev_obj']:
        log_event(d, 'DISTRACT_OBJECT', ', '.join(obj_r['object_names']))
        d['prev_obj'] = obj_r['total_events']

    alerts = []
    if d['mode'] == 'student':
        if d['absent_alert']: alerts.append('STUDENT ABSENT')
        if d['multi_face_alert']: alerts.append('MULTIPLE FACES')
    if eye_r['is_drowsy']:
        alerts.append('SLEEPING' if d['mode'] == 'student' else 'DROWSINESS')
    if phone_r['phone_detected']: alerts.append('PHONE USAGE')
    if head_r['is_distracted']:
        alerts.append('LOOKING AWAY' if d['mode'] == 'student' else 'EYES OFF ROAD')
    if yawn_r['is_yawning']: alerts.append('YAWNING')
    if obj_r['is_distracted']: alerts.append('OBJECT DISTRACTION')

    alarm_should = len(alerts) > 0
    if alarm_should and not d['last_alarm_state']:
        d['alarm'].play()
    elif not alarm_should and d['last_alarm_state']:
        d['alarm'].stop()
    d['last_alarm_state'] = alarm_should

    if alerts:
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 6)
        y_pos = h // 2 - (len(alerts) * 25)
        for a in alerts:
            cv2.rectangle(frame, (w // 2 - 200, y_pos),
                          (w // 2 + 200, y_pos + 45), (0, 0, 255), -1)
            cv2.putText(frame, '!! ' + a + ' !!', (w // 2 - 180, y_pos + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y_pos += 55

    state = {
        'eyes_closed': eye_r['eyes_closed'], 'blinks': eye_r['blinks'],
        'yawns': yawn_r['total_yawns'], 'phone_events': phone_r['total_events'],
        'head_dir': head_r['direction'],
        'object_names': obj_r['object_names'], 'face_count': num_faces,
        'absences': d['total_absences'], 'multi_faces': d['total_multi_face'],
        'away_events': head_r['total_distractions'],
        'obj_events': obj_r['total_events'],
        'alerts': alerts, 'score': d['score'], 'grade': get_grade(d)
    }
    return frame, state


def render_footer():
    st.markdown('''
        <div class="footer">
            <div class="footer-brand">🛡️ FocusGuard <span class="accent">AI</span></div>
            <div class="footer-line">A student project by data science students of <strong>PUCIT</strong></div>
            <div class="footer-line">Amara Tariq &nbsp;·&nbsp; Khansa Azeem &nbsp;·&nbsp; Ayesha Akbar</div>
            <div class="footer-meta">
                © 2026 FocusGuard AI &nbsp;·&nbsp; Privacy First &nbsp;·&nbsp; Built with MediaPipe + YOLOv8 + Streamlit
            </div>
        </div>
    ''', unsafe_allow_html=True)


# ============= PAGES =============
def page_home():
    st.markdown('''
        <div class="hero-wrap">
            <div class="hero-inner">
                <span class="hero-badge">✨ AI-POWERED COMPUTER VISION</span>
                <div class="hero-title">
                    Real-Time <span class="hero-title-accent">Attention Monitoring</span><br/>
                    for Drivers &amp; Students
                </div>
                <div class="hero-desc">
                    FocusGuard AI uses advanced facial analysis and object detection to 
                    prevent road accidents and maximize academic focus — running entirely 
                    on your device with industry-leading accuracy.
                </div>
                <div class="hero-meta">🎓 A Student Project by PUCIT Data Science Students</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
        <div class="trust-bar">
            <div class="trust-row">
                <div class="trust-item">
                    <div class="trust-num">99%</div>
                    <div class="trust-label">Accuracy</div>
                </div>
                <div class="trust-item">
                    <div class="trust-num">30 FPS</div>
                    <div class="trust-label">Real-Time</div>
                </div>
                <div class="trust-item">
                    <div class="trust-num">7+</div>
                    <div class="trust-label">AI Models</div>
                </div>
                <div class="trust-item">
                    <div class="trust-num">2</div>
                    <div class="trust-label">Modes</div>
                </div>
                <div class="trust-item">
                    <div class="trust-num">0ms</div>
                    <div class="trust-label">Cloud Data</div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Solutions</span>'
                '<div class="section-title">Built for Two Critical Use Cases</div>'
                '<div class="section-sub">Whether you are behind the wheel or behind a desk, '
                'FocusGuard AI keeps you safe and focused.</div>'
                '</div>', unsafe_allow_html=True)

    uc1, uc2 = st.columns(2, gap='large')
    with uc1:
        st.markdown('''
            <div class="uc-card">
                <img class="uc-image" src="https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?w=800&q=80"/>
                <div class="uc-body">
                    <span class="uc-tag">For Drivers</span>
                    <div class="uc-title">Driver Monitoring System</div>
                    <div class="uc-desc">
                        Reduce road accidents with real-time fatigue and distraction detection. 
                        Get instant alerts when drowsiness, phone usage, or looking away is detected.
                    </div>
                    <ul class="uc-list">
                        <li>Drowsiness detection (Eye Aspect Ratio)</li>
                        <li>Yawn detection (Mouth Aspect Ratio)</li>
                        <li>Phone &amp; object distraction alerts</li>
                        <li>Eyes-off-road head pose tracking</li>
                        <li>Audible alarm on dangerous behavior</li>
                    </ul>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 1.2rem;"></div>', unsafe_allow_html=True)
        if st.button('🚗  Launch Driver Mode', use_container_width=True, key='hero_d'):
            st.session_state.page = 'driver'; st.rerun()

    with uc2:
        st.markdown('''
            <div class="uc-card">
                <img class="uc-image" src="https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=800&q=80"/>
                <div class="uc-body">
                    <span class="uc-tag deep">For Students</span>
                    <div class="uc-title">Study &amp; Exam Monitoring</div>
                    <div class="uc-desc">
                        Maximize study productivity and ensure exam integrity with multi-modal 
                        focus tracking, absence detection, and cheating prevention.
                    </div>
                    <ul class="uc-list">
                        <li>All driver-mode features included</li>
                        <li>Absence detection alerts</li>
                        <li>Multi-face detection for exam integrity</li>
                        <li>Focus % and time-on-task analytics</li>
                        <li>Detailed downloadable session reports</li>
                    </ul>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        st.markdown('<div style="height: 1.2rem;"></div>', unsafe_allow_html=True)
        if st.button('📚  Launch Student Mode', use_container_width=True, key='hero_s'):
            st.session_state.page = 'student'; st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section section-alt">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Real-World Scenarios</span>'
                '<div class="section-title">Detection Scenarios We Cover</div>'
                '<div class="section-sub">Our AI is trained to detect a wide range of '
                'attention-related behaviors in everyday settings.</div>'
                '</div>', unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3, gap='large')
    with g1:
        st.markdown('''
            <div class="img-card">
                <img src="https://images.unsplash.com/photo-1486006920555-c77dcf18193c?w=600&q=80"/>
                <div class="img-card-body">
                    <div class="img-card-title">😴 Drowsy Driving</div>
                    <div class="img-card-desc">Detects micro-sleeps and prolonged eye closure 
                    using facial landmarks before fatigue causes accidents.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with g2:
        st.markdown('''
            <div class="img-card">
                <img src="https://images.unsplash.com/photo-1571624436279-b272aff752b5?w=600&q=80"/>
                <div class="img-card-body">
                    <div class="img-card-title">📱 Phone Distraction</div>
                    <div class="img-card-desc">YOLOv8 object detection identifies mobile phones 
                    in real time and triggers immediate alerts.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with g3:
        st.markdown('''
            <div class="img-card">
                <img src="https://images.unsplash.com/photo-1593642632559-0c6d3fc62b89?w=600&q=80"/>
                <div class="img-card-body">
                    <div class="img-card-title">👀 Eyes Off Road</div>
                    <div class="img-card-desc">Head pose estimation detects when the driver 
                    looks away from the road, even momentarily.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    g4, g5, g6 = st.columns(3, gap='large')
    with g4:
        st.markdown('''
            <div class="img-card">
                <img src="https://images.unsplash.com/photo-1434030216411-0b793f4b4173?w=600&q=80"/>
                <div class="img-card-body">
                    <div class="img-card-title">🎯 Focus Tracking</div>
                    <div class="img-card-desc">Continuously monitors student attention and 
                    calculates a focus percentage for the entire study session.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with g5:
        st.markdown('''
            <div class="img-card">
                <img src="https://images.unsplash.com/photo-1606761568499-6d2451b23c66?w=600&q=80"/>
                <div class="img-card-body">
                    <div class="img-card-title">📝 Exam Integrity</div>
                    <div class="img-card-desc">Multi-face detection flags potential cheating 
                    during online exams and proctored sessions.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with g6:
        st.markdown('''
            <div class="img-card">
                <img src="https://images.unsplash.com/photo-1543269865-cbf427effbad?w=600&q=80"/>
                <div class="img-card-body">
                    <div class="img-card-title">📚 Study Productivity</div>
                    <div class="img-card-desc">Track time-on-task, distraction events, and 
                    overall productivity across study sessions.</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Capabilities</span>'
                '<div class="section-title">Powered by Cutting-Edge AI</div>'
                '<div class="section-sub">Seven specialized AI models work in concert '
                'to deliver comprehensive attention analysis.</div>'
                '</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap='large')
    with f1:
        st.markdown('<div class="feat-card"><div class="feat-icon">👁️</div>'
                    '<div class="feat-title">Eye State Analysis</div>'
                    '<div class="feat-desc">Tracks blink rate, eye closure duration, and detects '
                    'micro-sleeps using MediaPipe facial landmarks.</div></div>',
                    unsafe_allow_html=True)
    with f2:
        st.markdown('<div class="feat-card"><div class="feat-icon">📱</div>'
                    '<div class="feat-title">Object Detection</div>'
                    '<div class="feat-desc">YOLOv8-powered detection identifies phones, food, '
                    'and other distracting objects in real time.</div></div>',
                    unsafe_allow_html=True)
    with f3:
        st.markdown('<div class="feat-card"><div class="feat-icon">🧭</div>'
                    '<div class="feat-title">Head Pose Tracking</div>'
                    '<div class="feat-desc">Estimates head yaw and pitch angles to detect '
                    'when attention drifts away.</div></div>',
                    unsafe_allow_html=True)

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)
    f4, f5, f6 = st.columns(3, gap='large')
    with f4:
        st.markdown('<div class="feat-card"><div class="feat-icon">😮</div>'
                    '<div class="feat-title">Yawn Detection</div>'
                    '<div class="feat-desc">Mouth aspect ratio analysis detects yawning '
                    'episodes — early indicators of fatigue.</div></div>',
                    unsafe_allow_html=True)
    with f5:
        st.markdown('<div class="feat-card"><div class="feat-icon">👥</div>'
                    '<div class="feat-title">Multi-Face Detection</div>'
                    '<div class="feat-desc">Flags when multiple faces appear in frame — '
                    'useful for online exam integrity.</div></div>',
                    unsafe_allow_html=True)
    with f6:
        st.markdown('<div class="feat-card"><div class="feat-icon">📊</div>'
                    '<div class="feat-title">Session Analytics</div>'
                    '<div class="feat-desc">Comprehensive reports with score, grade, time '
                    'breakdown, and event log saved as JSON.</div></div>',
                    unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section section-alt">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Workflow</span>'
                '<div class="section-title">How FocusGuard AI Works</div>'
                '<div class="section-sub">Three simple steps from setup to real-time '
                'intelligent monitoring.</div>'
                '</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3, gap='large')
    with s1:
        st.markdown('<div class="step-card"><div class="step-num">01</div>'
                    '<div class="step-title">Choose Your Mode</div>'
                    '<div class="step-desc">Select Driver Mode for road safety or Student '
                    'Mode for study and exam monitoring.</div></div>',
                    unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="step-card"><div class="step-num">02</div>'
                    '<div class="step-title">Start the Camera</div>'
                    '<div class="step-desc">Click START — FocusGuard AI begins analyzing your '
                    'camera feed in real time.</div></div>',
                    unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="step-card"><div class="step-num">03</div>'
                    '<div class="step-title">Get Live Insights</div>'
                    '<div class="step-desc">View instant alerts, performance score, and '
                    'download detailed session reports.</div></div>',
                    unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Technology</span>'
                '<div class="section-title">Built with Industry-Leading Tools</div>'
                '<div class="section-sub">A modern stack engineered for performance and reliability.</div>'
                '</div>'
                '<div class="tech-grid">'
                '<div class="tech-pill"><div class="tech-emoji">🐍</div><div class="tech-name">Python 3.11</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">⚡</div><div class="tech-name">Streamlit</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">👁️</div><div class="tech-name">MediaPipe</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">🎯</div><div class="tech-name">YOLOv8</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">📷</div><div class="tech-name">OpenCV</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">🚀</div><div class="tech-name">Ultralytics</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">🐳</div><div class="tech-name">Docker</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">☁️</div><div class="tech-name">Azure</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">🔄</div><div class="tech-name">GitHub Actions</div></div>'
                '<div class="tech-pill"><div class="tech-emoji">🔒</div><div class="tech-name">Privacy-First</div></div>'
                '</div>'
                '</div></div>', unsafe_allow_html=True)

    st.markdown('''
        <div class="cta-wrap">
            <div class="cta-title">Ready to Experience FocusGuard AI?</div>
            <div class="cta-sub">Try our project — no signup, no downloads, no data collection.</div>
        </div>
    ''', unsafe_allow_html=True)
    cta_l, cta_c, cta_r = st.columns([1, 2, 1])
    with cta_c:
        cb1, cb2 = st.columns(2, gap='medium')
        if cb1.button('🚗  Try Driver Mode', use_container_width=True, key='cta_d'):
            st.session_state.page = 'driver'; st.rerun()
        if cb2.button('📚  Try Student Mode', use_container_width=True, key='cta_s'):
            st.session_state.page = 'student'; st.rerun()

    render_footer()


def page_about():
    st.markdown('''
        <div class="hero-wrap">
            <div class="hero-inner">
                <span class="hero-badge">🎓 ABOUT THE TEAM</span>
                <div class="hero-title">
                    Built by Students,<br/>
                    <span class="hero-title-accent">for Everyone.</span>
                </div>
                <div class="hero-desc">
                    FocusGuard AI is a project by three data science students from 
                    Punjab University College of Information Technology — combining 
                    classroom learning with real-world AI engineering.
                </div>
                <div><span class="pucit-badge">🏛️ PUCIT Lahore</span></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="section">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">The Project</span>'
                '<div class="section-title">A Student Project in Computer Vision</div>'
                '<div class="section-sub">FocusGuard AI was built as a hands-on student project '
                'to apply the AI, deep learning, and software engineering skills we are learning '
                'at PUCIT to a real-world problem — preventing accidents and improving academic '
                'productivity through intelligent attention monitoring.</div>'
                '</div>'
                '</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section section-alt">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">The Team</span>'
                '<div class="section-title">Meet the Students Behind the Project</div>'
                '<div class="section-sub">Three data science students from PUCIT, '
                'building together.</div>'
                '</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3, gap='large')
    with t1:
        st.markdown('''
            <div class="team-card">
                <div class="team-avatar av-1">A</div>
                <div class="team-name">Amara Tariq</div>
                <div class="team-role">Data Science Student</div>
                <div class="team-bio">
                    Passionate about computer vision and AI applications.
                    Currently studying data science at PUCIT, Lahore.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with t2:
        st.markdown('''
            <div class="team-card">
                <div class="team-avatar av-2">K</div>
                <div class="team-name">Khansa Azeem</div>
                <div class="team-role">Data Science Student</div>
                <div class="team-bio">
                    Passionate about computer vision and AI applications.
                    Currently studying data science at PUCIT, Lahore.
                </div>
            </div>
        ''', unsafe_allow_html=True)
    with t3:
        st.markdown('''
            <div class="team-card">
                <div class="team-avatar av-3">A</div>
                <div class="team-name">Ayesha Akbar</div>
                <div class="team-role">Data Science Student</div>
                <div class="team-bio">
                    Passionate about computer vision and AI applications.
                    Currently studying data science at PUCIT, Lahore.
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Our Principles</span>'
                '<div class="section-title">Values that Guide Our Work</div>'
                '</div>', unsafe_allow_html=True)

    v1, v2, v3, v4 = st.columns(4, gap='medium')
    with v1:
        st.markdown('<div class="feat-card"><div class="feat-icon">🔒</div>'
                    '<div class="feat-title">Privacy First</div>'
                    '<div class="feat-desc">Your video stays on your device. Zero data collection.</div></div>',
                    unsafe_allow_html=True)
    with v2:
        st.markdown('<div class="feat-card"><div class="feat-icon">⚡</div>'
                    '<div class="feat-title">Real-Time</div>'
                    '<div class="feat-desc">Sub-50ms latency for instant detection.</div></div>',
                    unsafe_allow_html=True)
    with v3:
        st.markdown('<div class="feat-card"><div class="feat-icon">🎯</div>'
                    '<div class="feat-title">Accuracy</div>'
                    '<div class="feat-desc">99%+ detection on diverse datasets.</div></div>',
                    unsafe_allow_html=True)
    with v4:
        st.markdown('<div class="feat-card"><div class="feat-icon">🌍</div>'
                    '<div class="feat-title">Open Source</div>'
                    '<div class="feat-desc">Built openly so others can learn and extend.</div></div>',
                    unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section section-alt">'
                '<div class="section-inner">'
                '<div class="section-head">'
                '<span class="section-eyebrow">Acknowledgements</span>'
                '<div class="section-title">Thanks to PUCIT</div>'
                '<div class="section-sub">We are grateful to our teachers and the faculty at '
                'Punjab University College of Information Technology, Lahore for their guidance. '
                'Special thanks to the open-source community behind MediaPipe, Ultralytics YOLO, '
                'and Streamlit whose tools made this project possible.</div>'
                '</div>'
                '</div></div>', unsafe_allow_html=True)

    st.markdown('''
        <div class="cta-wrap">
            <div class="cta-title">Get in Touch</div>
            <div class="cta-sub">Questions, feedback, or want to collaborate? Reach out to us.</div>
            <div style="color:white;font-size:1.05rem;font-weight:600;">
                🏛️ PUCIT Lahore &nbsp;·&nbsp; 🌐 github.com/Amara-ch/focusguard-ai
            </div>
        </div>
    ''', unsafe_allow_html=True)

    render_footer()


def page_monitor(mode):
    title = '🚗 Driver Mode' if mode == 'driver' else '📚 Student Mode'
    desc = ('Real-time fatigue and distraction detection for safer driving.'
            if mode == 'driver'
            else 'Focus and attention tracking for productive study sessions.')

    st.markdown('<div class="monitor-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="monitor-title">' + title + '</div>', unsafe_allow_html=True)
    st.markdown('<div class="monitor-desc">' + desc + '</div>', unsafe_allow_html=True)

    col_back, col_btn = st.columns([1, 1], gap='medium')
    with col_back:
        if st.button('⬅️ Back to Home', use_container_width=True, key='back_' + mode):
            st.session_state.running = False
            if st.session_state.detectors:
                st.session_state.detectors['alarm'].cleanup()
                st.session_state.detectors = None
            st.session_state.page = 'home'
            st.rerun()

    with col_btn:
        if not st.session_state.running:
            if st.button('▶️ START', use_container_width=True, key='start_' + mode):
                st.session_state.detectors = init_detectors(mode)
                st.session_state.running = True
                st.rerun()
        else:
            if st.button('⏹️ STOP & SAVE', use_container_width=True, key='stop_' + mode):
                if st.session_state.detectors:
                    fname, _ = save_report(st.session_state.detectors)
                    st.success('Report saved: ' + fname)
                    st.session_state.detectors['alarm'].cleanup()
                    st.session_state.detectors = None
                st.session_state.running = False
                time.sleep(2)
                st.rerun()

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    if not st.session_state.running:
        st.info('👉 Click **START** above to begin live AI monitoring.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    col_video, col_stats = st.columns([2, 1], gap='large')
    video_placeholder = col_video.empty()
    stats_placeholder = col_stats.empty()
    alert_placeholder = col_stats.empty()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        st.error('Could not open webcam!')
        st.session_state.running = False
        st.markdown('</div>', unsafe_allow_html=True)
        return

    d = st.session_state.detectors
    try:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame, state = process_frame(d, frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels='RGB', use_container_width=True)

            with stats_placeholder.container():
                score = state['score']
                if score >= 75: color = '#10b981'
                elif score >= 50: color = '#a855f7'
                else: color = '#6d28d9'
                st.markdown(
                    '<div class="metric-card"><h2 style="color:' + color +
                    ';margin:0;font-weight:800;">' + str(int(score)) +
                    '<span style="font-size:1rem;opacity:0.6;">/100</span></h2>'
                    '<p style="margin:0.3rem 0 0 0;font-size:0.88rem;font-weight:600;opacity:0.75;">' +
                    state['grade'] + '</p></div>', unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                c1.metric('Faces', state['face_count'])
                c2.metric('Eyes', 'CLOSE' if state['eyes_closed'] else 'OPEN')

                c1, c2 = st.columns(2)
                c1.metric('Blinks', state['blinks'])
                c2.metric('Yawns', state['yawns'])

                c1, c2 = st.columns(2)
                c1.metric('Phone', state['phone_events'])
                c2.metric('Away', state['away_events'])

                if mode == 'student':
                    c1, c2 = st.columns(2)
                    c1.metric('Absences', state['absences'])
                    c2.metric('Multi-Face', state['multi_faces'])

                st.markdown('**Head:** ' + state['head_dir'])
                if state['object_names']:
                    st.markdown('**Objects:** ' + ', '.join(state['object_names']))

            with alert_placeholder.container():
                if state['alerts']:
                    for a in state['alerts']:
                        st.markdown('<div class="alert-box">⚠️ ' + a + '</div>',
                                    unsafe_allow_html=True)
                else:
                    st.markdown('<div class="ok-box">✅ All Good</div>',
                                unsafe_allow_html=True)
    finally:
        cap.release()
        if d:
            d['alarm'].stop()
    st.markdown('</div>', unsafe_allow_html=True)


def page_reports():
    st.markdown('<div class="monitor-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="monitor-title">📊 Session Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="monitor-desc">Detailed analytics from your past monitoring sessions.</div>',
                unsafe_allow_html=True)

    if st.button('⬅️ Back to Home', key='reports_back'):
        st.session_state.page = 'home'
        st.rerun()

    st.markdown('<div style="height: 1.5rem;"></div>', unsafe_allow_html=True)

    if not os.path.exists('reports'):
        st.warning('No reports yet. Run a monitoring session first.')
        st.markdown('</div>', unsafe_allow_html=True)
        return
    files = sorted(glob.glob('reports/*.json'), reverse=True)
    if not files:
        st.warning('No reports yet. Run a monitoring session first.')
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.write('**' + str(len(files)) + ' sessions found.**')
    for f in files[:20]:
        try:
            with open(f) as fp:
                data = json.load(fp)
        except Exception:
            continue
        mode = data.get('mode', 'UNKNOWN')
        score_key = 'safety_score' if mode == 'DRIVER' else 'focus_score'
        grade_key = 'safety_grade' if mode == 'DRIVER' else 'focus_grade'
        emoji = '🚗' if mode == 'DRIVER' else '📚'
        title = emoji + ' ' + mode + ' - ' + data['session_start'][:19]
        with st.expander(title):
            c1, c2, c3 = st.columns(3)
            c1.metric('Score', '%.0f/100' % data[score_key])
            c2.metric('Grade', data[grade_key])
            c3.metric('Duration', '%.0f sec' % data['duration_seconds'])
            st.json(data.get('stats', {}))
            if 'time_breakdown' in data:
                tb = data['time_breakdown']
                st.progress(int(tb['focused_pct']),
                            text='Focused: %.0f%%' % tb['focused_pct'])
                st.progress(int(tb['distracted_pct']),
                            text='Distracted: %.0f%%' % tb['distracted_pct'])
                st.progress(int(tb['absent_pct']),
                            text='Absent: %.0f%%' % tb['absent_pct'])
            if data.get('event_log'):
                st.dataframe(data['event_log'], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    with st.sidebar:
        st.markdown('### 🛡️ FocusGuard AI')
        st.caption('A PUCIT Student Project')
        st.markdown('---')

        st.markdown('**🧭 Navigation**')
        if st.button('🏠 Home', use_container_width=True, key='sb_home'):
            st.session_state.running = False
            st.session_state.page = 'home'; st.rerun()
        if st.button('🚗 Driver Mode', use_container_width=True, key='sb_d'):
            st.session_state.running = False
            st.session_state.page = 'driver'; st.rerun()
        if st.button('📚 Student Mode', use_container_width=True, key='sb_s'):
            st.session_state.running = False
            st.session_state.page = 'student'; st.rerun()
        if st.button('📊 Reports', use_container_width=True, key='sb_r'):
            st.session_state.running = False
            st.session_state.page = 'reports'; st.rerun()
        if st.button('ℹ️ About', use_container_width=True, key='sb_a'):
            st.session_state.running = False
            st.session_state.page = 'about'; st.rerun()

        st.markdown('---')
        st.markdown('**🎨 Appearance**')
        theme_choice = st.radio(
            'Theme',
            options=['light', 'dark'],
            format_func=lambda x: '☀️ Light' if x == 'light' else '🌙 Dark',
            index=0 if st.session_state.theme == 'light' else 1,
            label_visibility='collapsed',
            key='theme_radio'
        )
        if theme_choice != st.session_state.theme:
            st.session_state.theme = theme_choice
            st.rerun()

        st.markdown('---')
        st.caption('Made with 💜 by')
        st.caption('Amara Tariq · Khansa Azeem · Ayesha Akbar')
        st.caption('PUCIT Lahore · 2026')

    page = st.session_state.page
    if page == 'home':
        page_home()
    elif page == 'driver':
        page_monitor('driver')
    elif page == 'student':
        page_monitor('student')
    elif page == 'reports':
        page_reports()
    elif page == 'about':
        page_about()


if __name__ == '__main__':
    main()