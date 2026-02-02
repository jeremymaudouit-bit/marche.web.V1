import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import gaussian_filter1d
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# =====================================================
# STREAMLIT CONFIG
# =====================================================
st.set_page_config(page_title="GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
st.subheader("Analyse de la marche – cinématique 2D")

# =====================================================
# MOVENET
# =====================================================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, 0), 192, 192)
    input_img = tf.cast(img, tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    return outputs['output_0'].numpy()[0, 0, :, :]

# =====================================================
# JOINTS
# =====================================================
BASE_JOINTS = {
    "Epaule G": 5, "Epaule D": 6,
    "Hanche G": 11, "Hanche D": 12,
    "Genou G": 13, "Genou D": 14,
    "Cheville G": 15, "Cheville D": 16
}

def adjust_joints(cam_position, joints):
    j = joints.copy()
    if cam_position in ["Côté gauche", "Côté droit"]:
        for k in ["Epaule", "Hanche", "Genou", "Cheville"]:
            j[f"{k} G"], j[f"{k} D"] = j[f"{k} D"], j[f"{k} G"]
    return j

def angle(a, b, c):
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# =====================================================
# BANDPASS (doux)
# =====================================================
def bandpass_filter(signal, strength):
    if strength == 0 or len(signal) < 10:
        return np.array(signal)

    strength = np.sqrt(strength)
    low = 0.02
    high = min(0.45, 0.1 + strength * 0.03)

    b, a = butter(2, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)

# =====================================================
# NORMS (REALISTIC)
# =====================================================
def normal_curve(points, values, length):
    x = np.linspace(0, 100, length)
    curve = np.interp(x, points, values)
    return gaussian_filter1d(curve, 2)

def normal_hip(l): return normal_curve([0, 30, 60, 100], [0, 25, 10, 0], l)
def normal_knee(l): return normal_curve([0, 15, 60, 100], [0, 20, 60, 0], l)
def normal_ankle(l): return normal_curve([0, 10, 50, 100], [0, -10, 10, 0], l)
def normal_pelvis(l): return 5 * np.sin(2 * np.pi * np.linspace(0, 1, l))

# =====================================================
# VIDEO PROCESS
# =====================================================
def process_video(path, joints):
    cap = cv2.VideoCapture(path)
    results = {k: [] for k in [
        "Hanche G","Hanche D","Genou G","Genou D",
        "Cheville G","Cheville D","Pelvis","Dos"
    ]}
    heel_y = []
    best_frame, best_score = None, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)
        score = np.mean(kp[:, 2])
        if score > best_score:
            best_score = score
            best_frame = frame.copy()

        heel_y.append(kp[joints["Cheville D"], 1])

        results["Hanche G"].append(angle(kp[joints["Epaule G"], :2], kp[joints["Hanche G"], :2], kp[joints["Genou G"], :2]))
        results["Hanche D"].append(angle(kp[joints["Epaule D"], :2], kp[joints["Hanche D"], :2], kp[joints["Genou D"], :2]))

        results["Genou G"].append(angle(kp[joints["Hanche G"], :2], kp[joints["Genou G"], :2], kp[joints["Cheville G"], :2]))
        results["Genou D"].append(angle(kp[joints["Hanche D"], :2], kp[joints["Genou D"], :2], kp[joints["Cheville D"], :2]))

        results["Cheville G"].append(angle(kp[joints["Genou G"], :2], kp[joints["Cheville G"], :2], kp[joints["Cheville G"], :2] + [0, 1]))
        results["Cheville D"].append(angle(kp[joints["Genou D"], :2], kp[joints["Cheville D"], :2], kp[joints["Cheville D"], :2] + [0, 1]))

        pelvis = np.degrees(np.arctan2(
            kp[joints["Hanche D"], 1] - kp[joints["Hanche G"], 1],
            kp[joints["Hanche D"], 0] - kp[joints["Hanche G"], 0]
        ))
        results["Pelvis"].append(pelvis)

        results["Dos"].append(angle(
            kp[joints["Epaule G"], :2],
            (kp[joints["Hanche G"], :2] + kp[joints["Hanche D"], :2]) / 2,
            kp[joints["Epaule D"], :2]
        ))

    cap.release()
    return results, heel_y, best_frame

# =====================================================
# GAIT CYCLE (HEEL → HEEL)
# =====================================================
def detect_gait_cycle(heel_y):
    y = -np.array(heel_y)
    peaks, _ = find_peaks(y, distance=20, prominence=np.std(y) * 0.3)
    if len(peaks) >= 2:
        return peaks[0], peaks[1]
    return None, None

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")

    st.subheader("📹 Source")
    video = st.file_uploader("Vidéo", ["mp4", "avi"])
    live = st.checkbox("Caméra live")

    st.subheader("📐 Caméra")
    cam_pos = st.selectbox("Position", ["Devant", "Côté gauche", "Côté droit"])

    st.subheader("⚙️ Filtrage")
    smooth = st.slider("Force du filtrage", 0, 10, 2)

# =====================================================
# MAIN
# =====================================================
ready = False
if live:
    cam = st.camera_input("Live")
    if cam:
        video = cam
        ready = True
elif video:
    ready = True

if ready and st.button("▶ Lancer l’analyse"):
    joints = adjust_joints(cam_pos, BASE_JOINTS)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video.read())
    results, heel_y, best_frame = process_video(tmp.name, joints)
    os.unlink(tmp.name)

    cycle_start, cycle_end = detect_gait_cycle(heel_y)

    if best_frame is not None:
        img_path = os.path.join(tempfile.gettempdir(), "keyframe.png")
        cv2.imwrite(img_path, best_frame)
        st.image(img_path, caption="Image représentative")

    ARTICS = [
        ("Hanche", ("Hanche G", "Hanche D"), normal_hip),
        ("Genou", ("Genou G", "Genou D"), normal_knee),
        ("Cheville", ("Cheville G", "Cheville D"), normal_ankle),
    ]

    for name, (jg, jd), norm in ARTICS:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

        for j, c in zip([jg, jd], ["red", "blue"]):
            s = bandpass_filter(results[j], smooth)
            ax1.plot(s, label=j, color=c)

        if cycle_start:
            ax1.axvspan(cycle_start, cycle_end, color="orange", alpha=0.25)

        ax1.set_title(f"{name} réel")
        ax1.legend()

        ax2.plot(norm(len(s)), color="green")
        ax2.set_title("Norme")

        st.pyplot(fig)
        plt.close(fig)

    st.success("Analyse terminée ✔️")
