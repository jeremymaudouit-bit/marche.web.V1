import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tempfile, os
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.pagesizes import A4
from scipy.ndimage import gaussian_filter1d

# ==============================
# CONFIG STREAMLIT
# ==============================
st.set_page_config(page_title="GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro - Analyse cinématique")
st.subheader("Analyse flexion/extension des membres et posture du dos")

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
    """Calcul de l'angle abc en degrés"""
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

# ==============================
# TRAITEMENT VIDÉO
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
            # Flexion/extension angles
            # Hanche: epaule - hanche - genou
            results["Hanche G"].append(angle(kp[JOINTS_IDX["Epaule G"],:2], 
                                            kp[JOINTS_IDX["Hanche G"],:2], 
                                            kp[JOINTS_IDX["Genou G"],:2]))
            results["Hanche D"].append(angle(kp[JOINTS_IDX["Epaule D"],:2], 
                                            kp[JOINTS_IDX["Hanche D"],:2], 
                                            kp[JOINTS_IDX["Genou D"],:2]))
            # Genou: hanche - genou - cheville
            results["Genou G"].append(angle(kp[JOINTS_IDX["Hanche G"],:2],
                                           kp[JOINTS_IDX["Genou G"],:2],
                                           kp[JOINTS_IDX["Cheville G"],:2]))
            results["Genou D"].append(angle(kp[JOINTS_IDX["Hanche D"],:2],
                                           kp[JOINTS_IDX["Genou D"],:2],
                                           kp[JOINTS_IDX["Cheville D"],:2]))
            # Cheville: genou - cheville - pied (approximé par cheville + vecteur vertical)
            results["Cheville G"].append(angle(kp[JOINTS_IDX["Genou G"],:2],
                                             kp[JOINTS_IDX["Cheville G"],:2],
                                             kp[JOINTS_IDX["Cheville G"],:2]+np.array([0,1])))
            results["Cheville D"].append(angle(kp[JOINTS_IDX["Genou D"],:2],
                                             kp[JOINTS_IDX["Cheville D"],:2],
                                             kp[JOINTS_IDX["Cheville D"],:2]+np.array([0,1])))
            # Position du dos: angle epaule - hanche - hanche opposée
            results["Dos"].append(angle(kp[JOINTS_IDX["Epaule G"],:2],
                                       (kp[JOINTS_IDX["Hanche G"],:2]+kp[JOINTS_IDX["Hanche D"],:2])/2,
                                       kp[JOINTS_IDX["Epaule D"],:2]))
        frame_idx += 1
    cap.release()
    return results

# ==============================
# EXPORT PDF
# ==============================
def export_pdf(patient_info, joint_images):
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, "bilan_analyse.pdf")
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
        story.append(PDFImage(img_path, width=15*cm, height=8*cm))
        story.append(Spacer(1,0.5*cm))
    doc.build(story)
    return path

# ==============================
# INTERFACE STREAMLIT
# ==============================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    st.subheader("📹 Vidéo")
    video_file = st.file_uploader("Charger une vidéo (gauche ou droite)", type=["mp4","mov","avi"])
    st.subheader("⚙️ Paramètres")
    smoothing = st.slider("Lissage des courbes", 0, 10, 2)

if video_file:
    if st.button("⚙️ Lancer l'analyse"):
        with st.spinner("Analyse en cours..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            results = process_video(tfile.name, frame_skip=2)
            os.unlink(tfile.name)

            joint_imgs = {}
            for joint, angles in results.items():
                angles_smooth = gaussian_filter1d(angles, sigma=smoothing)
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(angles_smooth, lw=2)
                ax.set_title(f"{joint} (flexion/extension ou posture dos)")
                ax.set_xlabel("Frame")
                ax.set_ylabel("Angle (°)")
                st.pyplot(fig)
                img_path = os.path.join(tempfile.gettempdir(), f"{joint}.png")
                fig.savefig(img_path, bbox_inches='tight')
                plt.close(fig)
                joint_imgs[joint] = img_path

            # Export PDF
            pdf_path = export_pdf({"nom": nom, "prenom": prenom}, joint_imgs)
            with open(pdf_path, "rb") as f:
                st.download_button("📥 Télécharger le rapport PDF", f, f"Analyse_{nom}.pdf")
