import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# ======================================================
# CONFIG
# ======================================================
st.set_page_config("GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
st.subheader("Analyse 2D – marche / course")

FPS = 30

# ======================================================
# MOVENET
# ======================================================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(img[None], 192, 192)
    out = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
    return out["output_0"].numpy()[0, 0]

# ======================================================
# ARTICULATIONS
# ======================================================
JOINTS = {
    "Epaule G": 5, "Epaule D": 6,
    "Hanche G": 11, "Hanche D": 12,
    "Genou G": 13, "Genou D": 14,
    "Cheville G": 15, "Cheville D": 16,
}

def angle(a, b, c):
    ba, bc = a-b, c-b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# ======================================================
# BAND-PASS DOUX
# ======================================================
def bandpass(signal, level, fs=FPS):
    low = 0.3 + level*0.02
    high = 6.0 - level*0.25
    high = max(high, low + 0.4)

    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b, a, signal)

# ======================================================
# TRAITEMENT VIDEO
# ======================================================
def process_video(path, side):
    cap = cv2.VideoCapture(path)

    data = {k: [] for k in [
        "Hanche","Genou","Cheville","Pelvis","Dos"
    ]}
    heel_y, frames = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)
        frames.append(frame.copy())

        G = "G" if side == "Gauche" else "D"

        H = kp[JOINTS[f"Hanche {G}"], :2]
        K = kp[JOINTS[f"Genou {G}"], :2]
        A = kp[JOINTS[f"Cheville {G}"], :2]
        S = kp[JOINTS[f"Epaule {G}"], :2]

        data["Hanche"].append(angle(S, H, K))
        data["Genou"].append(angle(H, K, A))
        data["Cheville"].append(angle(K, A, A+np.array([0,1])))

        pelvis = kp[JOINTS["Hanche D"], :2] - kp[JOINTS["Hanche G"], :2]
        data["Pelvis"].append(np.degrees(np.arctan2(pelvis[1], pelvis[0])))

        mid_hip = (kp[11,:2]+kp[12,:2])/2
        mid_sh = (kp[5,:2]+kp[6,:2])/2
        data["Dos"].append(angle(mid_sh, mid_hip, mid_hip+np.array([0,-1])))

        heel_y.append(A[1])

    cap.release()
    return data, heel_y, frames

# ======================================================
# CYCLE TALON → TALON
# ======================================================
def detect_cycle(heel_y):
    inv = -np.array(heel_y)
    peaks, _ = find_peaks(inv, distance=FPS//2, prominence=np.std(inv)*0.3)
    if len(peaks) >= 2:
        return peaks[0], peaks[1]
    return 0, len(heel_y)-1

# ======================================================
# IMAGE STABLE (MILIEU DE CYCLE)
# ======================================================
def extract_keyframe(frames, start, end):
    idx = (start + end)//2
    path = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(path, frames[idx])
    return path

# ======================================================
# NORMES ARTICULAIRES (PHYSIO)
# ======================================================
def norm_curve(joint, n):
    x = np.linspace(0,100,n)
    if joint=="Genou":
        y = np.interp(x,[0,15,40,60,80,100],[5,15,5,40,60,5])
    elif joint=="Hanche":
        y = np.interp(x,[0,30,60,100],[30,0,-10,30])
    elif joint=="Cheville":
        y = np.interp(x,[0,10,50,70,100],[0,-5,10,-15,0])
    else:
        y = np.zeros(n)
    return y

# ======================================================
# PDF
# ======================================================
def export_pdf(info, figs, table):
    path = os.path.join(tempfile.gettempdir(), "rapport.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Analyse Cinématique</b>", styles["Title"]),
        Paragraph(f"Patient : {info['nom']} {info['prenom']}", styles["Normal"]),
        Paragraph(datetime.now().strftime("%d/%m/%Y"), styles["Normal"]),
        Spacer(1,1*cm),
        PDFImage(info["image"], width=15*cm, height=8*cm),
        Spacer(1,0.5*cm)
    ]

    for title, img in figs.items():
        story += [
            Paragraph(f"<b>{title}</b>", styles["Heading2"]),
            PDFImage(img, width=16*cm, height=6*cm),
            Spacer(1,0.4*cm)
        ]

    table = Table([["Articulation","Min","Moy","Max"]]+table)
    table.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),1,colors.black),
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey)
    ]))
    story.append(table)
    doc.build(story)
    return path

# ======================================================
# INTERFACE
# ======================================================
with st.sidebar:
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")
    side = st.selectbox("Côté filmé",["Gauche","Droit"])
    smooth = st.slider("Lissage (band-pass)",0,10,3)
    src = st.radio("Source",["Vidéo","Caméra"])

video = st.file_uploader("Vidéo",["mp4","avi","mov"]) if src=="Vidéo" else st.camera_input("Caméra")

# ======================================================
# ANALYSE
# ======================================================
if video and st.button("▶ Lancer l'analyse"):
    with st.spinner("Analyse en cours..."):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.write(video.read())
        tmp.close()

        data, heel_y, frames = process_video(tmp.name, side)
        os.unlink(tmp.name)

        heel_f = bandpass(np.array(heel_y), smooth)
        c0, c1 = detect_cycle(heel_f)
        key_img = extract_keyframe(frames, c0, c1)

        figs, table = {}, []

        for joint in ["Hanche","Genou","Cheville"]:
            sig = bandpass(np.array(data[joint]), smooth)
            nrm = norm_curve(joint, len(sig))

            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(sig, lw=2, label="Réel")
            ax.plot(nrm, lw=1.5, linestyle="--", label="Norme")
            ax.axvspan(c0, c1, color="orange", alpha=0.2)
            ax.set_title(joint)
            ax.legend()
            st.pyplot(fig)

            img = os.path.join(tempfile.gettempdir(), f"{joint}.png")
            fig.savefig(img, bbox_inches="tight")
            plt.close(fig)
            figs[joint] = img

            table.append([
                joint,
                f"{sig.min():.1f}",
                f"{sig.mean():.1f}",
                f"{sig.max():.1f}"
            ])

        pdf = export_pdf(
            {"nom":nom,"prenom":prenom,"image":key_img},
            figs,
            table
        )

        with open(pdf,"rb") as f:
            st.download_button("📄 Télécharger le PDF", f, "rapport_gaitscan.pdf")
