import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(page_title="GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse de la marche")

# =========================
# MOVENET
# =========================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, tf.int32)
    outputs = movenet.signatures["serving_default"](input_img)
    return outputs["output_0"].numpy()[0, 0, :, :]

# =========================
# ARTICULATIONS
# =========================
JOINTS = {
    "Epaule G": 5, "Epaule D": 6,
    "Hanche G": 11, "Hanche D": 12,
    "Genou G": 13, "Genou D": 14,
    "Cheville G": 15, "Cheville D": 16
}

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# =========================
# FILTRE BAND-PASS
# =========================
def bandpass(signal, fs=30, low=0.3, high=6):
    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)

# =========================
# TRAITEMENT VIDÉO
# =========================
def process_video(path, side):
    cap = cv2.VideoCapture(path)
    frames, heel_y = [], []

    results = {k: [] for k in [
        "Hanche", "Genou", "Cheville", "Pelvis", "Dos"
    ]}

    side_suffix = "D" if side == "Droit" else "G"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)
        frames.append(frame.copy())

        hip = kp[JOINTS[f"Hanche {side_suffix}"], :2]
        knee = kp[JOINTS[f"Genou {side_suffix}"], :2]
        ankle = kp[JOINTS[f"Cheville {side_suffix}"], :2]
        shoulder = kp[JOINTS[f"Epaule {side_suffix}"], :2]

        results["Genou"].append(angle(hip, knee, ankle))
        results["Hanche"].append(angle(shoulder, hip, knee))
        results["Cheville"].append(angle(knee, ankle, ankle + np.array([0, 1])))

        pelvis_angle = np.degrees(
            np.arctan2(
                kp[JOINTS["Hanche D"], 1] - kp[JOINTS["Hanche G"], 1],
                kp[JOINTS["Hanche D"], 0] - kp[JOINTS["Hanche G"], 0]
            )
        )
        results["Pelvis"].append(pelvis_angle)

        spine_mid = (kp[JOINTS["Epaule G"], :2] + kp[JOINTS["Epaule D"], :2]) / 2
        hip_mid = (kp[JOINTS["Hanche G"], :2] + kp[JOINTS["Hanche D"], :2]) / 2
        results["Dos"].append(angle(spine_mid, hip_mid, hip_mid + np.array([0, -1])))

        heel_y.append(ankle[1])

    cap.release()
    return results, heel_y, frames

# =========================
# DÉTECTION CYCLE TALON
# =========================
def detect_cycle(heel_y):
    signal = bandpass(np.array(heel_y))
    mins = np.where((signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:]))[0] + 1
    if len(mins) >= 2:
        return mins[0], mins[1]
    return None, None

# =========================
# COURBES NORMALES
# =========================
def norm_curve(kind, n):
    x = np.linspace(0, 1, n)
    if kind == "Genou":
        return 10 + 50 * np.sin(np.pi * x)
    if kind == "Hanche":
        return 30 - 40 * x
    if kind == "Cheville":
        return -10 + 25 * np.sin(2 * np.pi * x)
    if kind == "Pelvis":
        return 5 * np.sin(2 * np.pi * x)
    if kind == "Dos":
        return 5 + 5 * np.sin(2 * np.pi * x)

# =========================
# PDF
# =========================
def generate_pdf(nom, prenom, image_path, figs):
    pdf_path = os.path.join(tempfile.gettempdir(), "rapport_gaitscan.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elems = []

    elems.append(Paragraph("<b>Analyse de la marche</b>", styles["Title"]))
    elems.append(Paragraph(f"{prenom} {nom}", styles["Normal"]))
    elems.append(Paragraph(datetime.now().strftime("%d/%m/%Y"), styles["Normal"]))
    elems.append(Spacer(1, 12))

    if image_path:
        elems.append(Paragraph("Posture représentative (milieu du cycle)", styles["Heading2"]))
        elems.append(PDFImage(image_path, width=14*cm, height=7*cm))
        elems.append(Spacer(1, 12))

    for f in figs:
        elems.append(PDFImage(f, width=15*cm, height=6*cm))
        elems.append(Spacer(1, 12))

    doc.build(elems)
    return pdf_path

# =========================
# INTERFACE
# =========================
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    side = st.selectbox("Côté filmé", ["Droit", "Gauche"])
    video = st.file_uploader("Vidéo", ["mp4", "avi", "mov"])

if video and st.button("Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video.read())
    tmp.close()

    results, heel_y, frames = process_video(tmp.name, side)
    start, end = detect_cycle(heel_y)

    pdf_figs = []

    if start and end:
        mid = (start + end) // 2
        img_path = os.path.join(tempfile.gettempdir(), "posture.png")
        cv2.imwrite(img_path, frames[mid])
        st.image(img_path, caption="Milieu du cycle")

    for joint, values in results.items():
        filt = bandpass(np.array(values))
        fig, ax = plt.subplots()
        ax.plot(filt, label="Mesure")
        ax.plot(norm_curve(joint, len(filt)), "--", label="Norme")
        if start and end:
            ax.axvspan(start, end, color="yellow", alpha=0.3)
        ax.set_title(joint)
        ax.legend()
        st.pyplot(fig)

        path = os.path.join(tempfile.gettempdir(), f"{joint}.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        pdf_figs.append(path)

    pdf = generate_pdf(nom, prenom, img_path if start else None, pdf_figs)
    with open(pdf, "rb") as f:
        st.download_button("📄 Télécharger le PDF", f, "rapport.pdf")
