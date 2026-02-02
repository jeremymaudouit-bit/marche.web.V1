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
st.subheader("Flexion/extension des membres et posture du dos")

# ==============================
# CHARGEMENT MOVE NET
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
    keypoints = outputs['output_0'].numpy()
    return keypoints[0,0,:,:]

# ==============================
# ARTICULATIONS
# ==============================
JOINTS_IDX = {
    "Hanche G": 11, "Genou G": 13, "Cheville G": 15,
    "Hanche D": 12, "Genou D": 14, "Cheville D": 16,
    "Epaule G": 5, "Epaule D": 6
}

def angle(a, b, c):
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# ==============================
# TRAITEMENT VIDEO / CAMERA
# ==============================
def process_video(video_file, frame_skip=2):
    cap = cv2.VideoCapture(video_file)
    results = {joint: [] for joint in ["Hanche G","Genou G","Cheville G","Hanche D","Genou D","Cheville D","Pelvis","Dos"]}
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_skip == 0:
            kp = detect_pose(frame)
            # Hanche
            results["Hanche G"].append(angle(kp[JOINTS_IDX["Epaule G"],:2], kp[JOINTS_IDX["Hanche G"],:2], kp[JOINTS_IDX["Genou G"],:2]))
            results["Hanche D"].append(angle(kp[JOINTS_IDX["Epaule D"],:2], kp[JOINTS_IDX["Hanche D"],:2], kp[JOINTS_IDX["Genou D"],:2]))
            # Genou
            results["Genou G"].append(angle(kp[JOINTS_IDX["Hanche G"],:2], kp[JOINTS_IDX["Genou G"],:2], kp[JOINTS_IDX["Cheville G"],:2]))
            results["Genou D"].append(angle(kp[JOINTS_IDX["Hanche D"],:2], kp[JOINTS_IDX["Genou D"],:2], kp[JOINTS_IDX["Cheville D"],:2]))
            # Cheville
            results["Cheville G"].append(angle(kp[JOINTS_IDX["Genou G"],:2], kp[JOINTS_IDX["Cheville G"],:2], kp[JOINTS_IDX["Cheville G"],:2]+np.array([0,1])))
            results["Cheville D"].append(angle(kp[JOINTS_IDX["Genou D"],:2], kp[JOINTS_IDX["Cheville D"],:2], kp[JOINTS_IDX["Cheville D"],:2]+np.array([0,1])))
            # Dos
            results["Dos"].append(angle(kp[JOINTS_IDX["Epaule G"],:2], (kp[JOINTS_IDX["Hanche G"],:2]+kp[JOINTS_IDX["Hanche D"],:2])/2, kp[JOINTS_IDX["Epaule D"],:2]))
            # Pelvis (bascule approximative selon ligne Hanche G-D)
            pelvis_angle = np.degrees(np.arctan2(kp[JOINTS_IDX["Hanche D"],1]-kp[JOINTS_IDX["Hanche G"],1],
                                                kp[JOINTS_IDX["Hanche D"],0]-kp[JOINTS_IDX["Hanche G"],0]))
            results["Pelvis"].append(pelvis_angle)
        frame_idx +=1
    cap.release()
    return results

# ==============================
# MODÈLE NORMAL LISSE
# ==============================
def normal_ankle(length=100, sigma=2):
    cycle_percent = np.array([0, 10, 40, 60, 80, 100])
    angles = np.array([0, -5, 10, -17.5, 0, 0])
    x = np.linspace(0, 100, length)
    curve = np.interp(x, cycle_percent, angles)
    return gaussian_filter1d(curve, sigma=sigma)

def normal_knee(length=100, sigma=2):
    cycle_percent = np.array([0, 15, 40, 60, 75, 100])
    angles = np.array([5, 18, 3, 35, 60, 5])
    x = np.linspace(0, 100, length)
    curve = np.interp(x, cycle_percent, angles)
    return gaussian_filter1d(curve, sigma=sigma)

def normal_hip(length=100, sigma=2):
    cycle_percent = np.array([0, 30, 55, 85, 100])
    angles = np.array([30, 0, -15, 20, 30])
    x = np.linspace(0, 100, length)
    curve = np.interp(x, cycle_percent, angles)
    return gaussian_filter1d(curve, sigma=sigma)

def normal_pelvis(length=100, sigma=2):
    t = np.linspace(0, 1, length)
    curve = 5*np.sin(2*np.pi*t)
    return gaussian_filter1d(curve, sigma=sigma)

# ==============================
# EXPORT PDF
# ==============================
def export_pdf(patient_info, joint_images, summary_table):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "rapport_analyse.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Bilan Analyse Cinématique</b>", styles['Title']),
        Paragraph(f"Patient : {patient_info['nom']} {patient_info['prenom']}", styles['Normal']),
        Paragraph(f"Date : {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']),
        Spacer(1,1*cm)
    ]
    for joint, img_path in joint_images.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles['Heading2']))
        story.append(PDFImage(img_path, width=15*cm, height=6*cm))
        story.append(Spacer(1,0.5*cm))
    story.append(Paragraph("<b>Résumé des angles (°)</b>", styles['Heading2']))
    table_data = [["Articulation", "Min", "Moyenne", "Max"]] + summary_table
    table = Table(table_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(table)
    doc.build(story)
    return path

# ==============================
# INTERFACE
# ==============================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    
    st.subheader("📹 Source")
    video_file = st.file_uploader("Charger une vidéo", type=["mp4","mov","avi"])
    live_cam = st.checkbox("Ou utiliser la caméra live")
    
    st.subheader("⚙️ Paramètres")
    smoothing = st.slider("Lissage des courbes", 0, 10, 2)
    show_normal = st.checkbox("Afficher modèle normal à côté", value=True)

    st.subheader("📐 Position caméra")
    cam_position = st.selectbox("Position de la caméra par rapport au patient", ["Devant", "Côté gauche", "Côté droit"])

# ==============================
# Ajustement G/D selon caméra
# ==============================
def adjust_joints_for_camera(cam_position, joints_idx):
    if cam_position in ["Côté gauche", "Côté droit"]:
        # Inverse G/D
        joints_idx = joints_idx.copy()
        joints_idx["Hanche G"], joints_idx["Hanche D"] = joints_idx["Hanche D"], joints_idx["Hanche G"]
        joints_idx["Genou G"], joints_idx["Genou D"] = joints_idx["Genou D"], joints_idx["Genou G"]
        joints_idx["Cheville G"], joints_idx["Cheville D"] = joints_idx["Cheville D"], joints_idx["Cheville G"]
        joints_idx["Epaule G"], joints_idx["Epaule D"] = joints_idx["Epaule D"], joints_idx["Epaule G"]
    return joints_idx

JOINTS_IDX = adjust_joints_for_camera(cam_position, JOINTS_IDX)

# ==============================
# ANALYSE
# ==============================
video_ready = False
if live_cam:
    cam_file = st.camera_input("🎥 Caméra")
    if cam_file:
        video_file = cam_file
        video_ready = True
elif video_file:
    video_ready = True

if video_ready and st.button("⚙️ Lancer l'analyse"):
    with st.spinner("Analyse en cours..."):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        results = process_video(tfile.name, frame_skip=2)
        os.unlink(tfile.name)

        joint_imgs = {}
        summary_table = []

        articulation_pairs = [("Hanche G","Hanche D"), ("Genou G","Genou D"), ("Cheville G","Cheville D")]
        normal_funcs = [normal_hip, normal_knee, normal_ankle]

        for (joint_pair, normal_func) in zip(articulation_pairs, normal_funcs):
            col1, col2 = st.columns(2)

            # Colonne 1 : réel
            fig, ax = plt.subplots(figsize=(6,4))
            for joint, color in zip(joint_pair, ['red','blue']):
                angles_smooth = gaussian_filter1d(results[joint], sigma=smoothing)
                ax.plot(angles_smooth, lw=2, color=color, label=joint)
                summary_table.append([joint, f"{np.min(results[joint]):.1f}", f"{np.mean(results[joint]):.1f}", f"{np.max(results[joint]):.1f}"])
            ax.set_title(f"{joint_pair[0].split()[0]} : Réel")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Angle (°)")
            ax.legend()
            col1.pyplot(fig)
            img_path = os.path.join(tempfile.gettempdir(), f"{joint_pair[0]}_reel.png")
            fig.savefig(img_path, bbox_inches='tight')
            plt.close(fig)
            joint_imgs[f"{joint_pair[0]} & {joint_pair[1]} Réel"] = img_path

            # Colonne 2 : modèle normal
            if show_normal:
                fig2, ax2 = plt.subplots(figsize=(6,4))
                length = len(results[joint_pair[0]])
                normal_curve = normal_func(length)
                ax2.plot(normal_curve, lw=2, color='green', label="Modèle normal")
                ax2.set_title(f"{joint_pair[0].split()[0]} : Modèle normal")
                ax2.set_xlabel("Frame")
                ax2.set_ylabel("Angle (°)")
                ax2.legend()
                col2.pyplot(fig2)
                img_path2 = os.path.join(tempfile.gettempdir(), f"{joint_pair[0]}_normal.png")
                fig2.savefig(img_path2, bbox_inches='tight')
                plt.close(fig2)
                joint_imgs[f"{joint_pair[0]} & {joint_pair[1]} Normal"] = img_path2

        # Pelvis
        angles_smooth = gaussian_filter1d(results["Pelvis"], sigma=smoothing)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(angles_smooth, lw=2, color='purple', label="Pelvis réel")
        if show_normal:
            normal_curve = normal_pelvis(len(angles_smooth))
            ax.plot(normal_curve, lw=2, color='green', linestyle='--', label="Pelvis modèle")
        ax.set_title("Bascule Pelvis")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        st.pyplot(fig)
        img_path = os.path.join(tempfile.gettempdir(), "Pelvis.png")
        fig.savefig(img_path, bbox_inches='tight')
        plt.close(fig)
        joint_imgs["Pelvis"] = img_path
        summary_table.append(["Pelvis", f"{np.min(results['Pelvis']):.1f}", f"{np.mean(results['Pelvis']):.1f}", f"{np.max(results['Pelvis']):.1f}"])

        # Dos
        angles_smooth = gaussian_filter1d(results["Dos"], sigma=smoothing)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(angles_smooth, lw=2, color='green', label="Dos")
        ax.set_title("Dos")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
        ax.legend()
        st.pyplot(fig)
        img_path = os.path.join(tempfile.gettempdir(), f"Dos.png")
        fig.savefig(img_path, bbox_inches='tight')
        plt.close(fig)
        joint_imgs["Dos"] = img_path
        summary_table.append(["Dos", f"{np.min(results['Dos']):.1f}", f"{np.mean(results['Dos']):.1f}", f"{np.max(results['Dos']):.1f}"])

        # Export PDF
        pdf_path = export_pdf({"nom": nom, "prenom": prenom}, joint_imgs, summary_table)
        with open(pdf_path, "rb") as f:
            st.download_button("📥 Télécharger le rapport PDF", f, f"Analyse_{nom}.pdf")
