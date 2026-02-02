import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from scipy.ndimage import gaussian_filter1d

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro - Analyse Cinématique")
st.subheader("Analyse automatique de la marche")

# ==============================
# MOVE NET
# ==============================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 192, 192)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = movenet.signatures['serving_default'](input_img)
    return outputs['output_0'].numpy()[0,0,:,:]

# ==============================
# ARTICULATIONS
# ==============================
JOINTS_IDX = {
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

# ==============================
# VIDEO
# ==============================
def process_video(video_path, frame_skip=2):
    cap = cv2.VideoCapture(video_path)
    results = {k: [] for k in ["Hanche G","Hanche D","Genou G","Genou D","Cheville G","Cheville D","Pelvis","Dos"]}
    frames = []
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip == 0:
            kp = detect_pose(frame)
            frames.append((frame.copy(), kp))

            results["Hanche G"].append(angle(kp[5,:2], kp[11,:2], kp[13,:2]))
            results["Hanche D"].append(angle(kp[6,:2], kp[12,:2], kp[14,:2]))
            results["Genou G"].append(angle(kp[11,:2], kp[13,:2], kp[15,:2]))
            results["Genou D"].append(angle(kp[12,:2], kp[14,:2], kp[16,:2]))

            vertical = kp[15,:2] + np.array([0,1])
            results["Cheville G"].append(angle(kp[13,:2], kp[15,:2], vertical))
            results["Cheville D"].append(angle(kp[14,:2], kp[16,:2], vertical))

            pelvis = np.degrees(np.arctan2(kp[12,1]-kp[11,1], kp[12,0]-kp[11,0]))
            results["Pelvis"].append(pelvis)

            dos = angle(kp[5,:2], (kp[11,:2]+kp[12,:2])/2, kp[6,:2])
            results["Dos"].append(dos)
        i += 1

    cap.release()
    return results, frames

# ==============================
# IMAGE CLÉ (GARANTIE)
# ==============================
def select_keyframe(frames):
    frame = frames[len(frames)//2][0]
    path = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(path, frame)
    return path

# ==============================
# MODÈLES NORMAUX
# ==============================
def normal_curve(points, angles):
    x = np.linspace(0, 100, len(angles))
    return gaussian_filter1d(np.interp(np.linspace(0,100,points), x, angles), 2)

def normal_hip(n): return normal_curve(n, [30, 0, -15, 20, 30])
def normal_knee(n): return normal_curve(n, [5, 18, 3, 35, 60, 5])
def normal_ankle(n): return normal_curve(n, [0, -5, 10, -17, 0, 0])

# ==============================
# GRAPHIQUE COMPARATIF
# ==============================
def plot_real_vs_normal(results, joints, normal_func, smoothing, title):
    fig = plt.figure(figsize=(10,4))
    gs = fig.add_gridspec(1,3)

    ax_real = fig.add_subplot(gs[0,:2])
    ax_norm = fig.add_subplot(gs[0,2])

    for j in joints:
        ax_real.plot(gaussian_filter1d(results[j], smoothing), lw=2, label=j)

    ax_real.set_title(f"{title} – Mesuré")
    ax_real.legend()
    ax_real.grid(alpha=0.3)

    n = len(results[joints[0]])
    ax_norm.plot(normal_func(n), color="green", lw=2)
    ax_norm.set_title("Norme")
    ax_norm.grid(alpha=0.3)

    plt.tight_layout()
    return fig

# ==============================
# PDF
# ==============================
def export_pdf(patient, keyframe, joint_imgs, table_data):
    path = os.path.join(tempfile.gettempdir(), "rapport_gaitscan.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Analyse Cinématique – GaitScan Pro</b>", styles["Title"]),
        Paragraph(f"Patient : {patient}", styles["Normal"]),
        Paragraph(datetime.now().strftime("%d/%m/%Y"), styles["Normal"]),
        Spacer(1,1*cm),
        Paragraph("<b>Image extraite de la vidéo</b>", styles["Heading2"]),
        PDFImage(keyframe, width=15*cm, height=8*cm),
        Spacer(1,1*cm)
    ]

    for name, img in joint_imgs.items():
        story.append(Paragraph(f"<b>{name}</b>", styles["Heading2"]))
        story.append(PDFImage(img, width=15*cm, height=5*cm))
        story.append(Spacer(1,0.5*cm))

    table = Table([["Articulation","Min","Moy","Max"]] + table_data)
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),1,colors.black)]))
    story.append(table)

    doc.build(story)
    return path

# ==============================
# INTERFACE
# ==============================
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    video = st.file_uploader("Vidéo", type=["mp4","avi"])
    smoothing = st.slider("Lissage", 0, 5, 2)

if video and st.button("Lancer l'analyse"):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video.read())

    results, frames = process_video(tfile.name)
    keyframe = select_keyframe(frames)

    joint_imgs = {}
    table_data = []

    articulations = [
        (["Hanche G","Hanche D"], normal_hip, "Hanche"),
        (["Genou G","Genou D"], normal_knee, "Genou"),
        (["Cheville G","Cheville D"], normal_ankle, "Cheville")
    ]

    for joints, norm, title in articulations:
        fig = plot_real_vs_normal(results, joints, norm, smoothing, title)
        st.pyplot(fig)

        img_path = os.path.join(tempfile.gettempdir(), f"{title}.png")
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        joint_imgs[title] = img_path

        for j in joints:
            table_data.append([
                j,
                f"{np.min(results[j]):.1f}",
                f"{np.mean(results[j]):.1f}",
                f"{np.max(results[j]):.1f}"
            ])

    pdf = export_pdf(f"{nom} {prenom}", keyframe, joint_imgs, table_data)
    with open(pdf, "rb") as f:
        st.download_button("📥 Télécharger le rapport PDF", f, "Analyse_GaitScan.pdf")
