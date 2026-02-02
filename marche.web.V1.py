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
# CONFIG STREAMLIT
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
    keypoints = outputs['output_0'].numpy()  # [1,1,17,3]
    return keypoints[0,0,:,:]  # [17,3]

# ==============================
# ARTICULATIONS
# ==============================
JOINTS_IDX = {
    "Hanche G": 11,
    "Genou G": 13,
    "Cheville G": 15,
    "Hanche D": 12,
    "Genou D": 14,
    "Cheville D": 16,
    "Epaule G": 5,
    "Epaule D": 6
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
    results = {
        "Hanche G": [], "Genou G": [], "Cheville G": [],
        "Hanche D": [], "Genou D": [], "Cheville D": [],
        "Dos": []
    }
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            kp = detect_pose(frame)
            # Hanche
            results["Hanche G"].append(angle(kp[JOINTS_IDX["Epaule G"],:2],
                                             kp[JOINTS_IDX["Hanche G"],:2],
                                             kp[JOINTS_IDX["Genou G"],:2]))
            results["Hanche D"].append(angle(kp[JOINTS_IDX["Epaule D"],:2],
                                             kp[JOINTS_IDX["Hanche D"],:2],
                                             kp[JOINTS_IDX["Genou D"],:2]))
            # Genou
            results["Genou G"].append(angle(kp[JOINTS_IDX["Hanche G"],:2],
                                            kp[JOINTS_IDX["Genou G"],:2],
                                            kp[JOINTS_IDX["Cheville G"],:2]))
            results["Genou D"].append(angle(kp[JOINTS_IDX["Hanche D"],:2],
                                            kp[JOINTS_IDX["Genou D"],:2],
                                            kp[JOINTS_IDX["Cheville D"],:2]))
            # Cheville
            results["Cheville G"].append(angle(kp[JOINTS_IDX["Genou G"],:2],
                                               kp[JOINTS_IDX["Cheville G"],:2],
                                               kp[JOINTS_IDX["Cheville G"],:2]+np.array([0,1])))
            results["Cheville D"].append(angle(kp[JOINTS_IDX["Genou D"],:2],
                                               kp[JOINTS_IDX["Cheville D"],:2],
                                               kp[JOINTS_IDX["Cheville D"],:2]+np.array([0,1])))
            # Dos
            results["Dos"].append(angle(kp[JOINTS_IDX["Epaule G"],:2],
                                        (kp[JOINTS_IDX["Hanche G"],:2]+kp[JOINTS_IDX["Hanche D"],:2])/2,
                                        kp[JOINTS_IDX["Epaule D"],:2]))
        frame_idx +=1
    cap.release()
    return results

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
# INTERFACE STREAMLIT
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
        
        # Courbes superposées gauche/droite
        for joint_pair, color_pair in zip(
            [("Hanche G","Hanche D"),("Genou G","Genou D"),("Cheville G","Cheville D")],
            [('red','blue'),('red','blue'),('red','blue')]
        ):
            fig, ax = plt.subplots(figsize=(10,4))
            for joint, color in zip(joint_pair, color_pair):
                angles_smooth = gaussian_filter1d(results[joint], sigma=smoothing)
                ax.plot(angles_smooth, lw=2, color=color, label=joint)
                # résumé
                summary_table.append([joint, f"{np.min(results[joint]):.1f}", f"{np.mean(results[joint]):.1f}", f"{np.max(results[joint]):.1f}"])
            ax.set_title(f"{joint_pair[0].split()[0]} : Flexion/Extension Gauche/Droite")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Angle (°)")
            ax.legend()
            st.pyplot(fig)
            img_path = os.path.join(tempfile.gettempdir(), f"{joint_pair[0]}_{joint_pair[1]}.png")
            fig.savefig(img_path, bbox_inches='tight')
            plt.close(fig)
            joint_imgs[f"{joint_pair[0]} & {joint_pair[1]}"] = img_path
        
        # Dos
        angles_smooth = gaussian_filter1d(results["Dos"], sigma=smoothing)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(angles_smooth, lw=2, color='green')
        ax.set_title("Dos")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Angle (°)")
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
