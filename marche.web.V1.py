import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as PDFImage, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# =====================================================
# CONFIG STREAMLIT
# =====================================================
st.set_page_config(page_title="GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
st.subheader("Analyse articulaire et posture")

# =====================================================
# LOAD MOVENET
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
        # Inverser gauche/droite
        for k in ["Epaule","Hanche","Genou","Cheville"]:
            j[f"{k} G"], j[f"{k} D"] = j[f"{k} D"], j[f"{k} G"]
    return j

def angle(a, b, c):
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

# =====================================================
# FILTRE BANDE-PASSANTE
# =====================================================
def bandpass_filter(signal, slider_value):
    signal = np.array(signal)
    if slider_value == 0 or len(signal) < 5:
        return signal
    
    # Transformation douce du slider
    strength = np.sqrt(slider_value)
    
    # Définir fréquence normalisée (0-1, 1=Nyquist)
    low = 0.01  # passe-haut fixe
    high = min(0.5, 0.05*strength + 0.1)  # passe-bas varie avec slider

    b, a = butter(N=2, Wn=[low, high], btype='bandpass')
    filtered = filtfilt(b, a, signal)
    return filtered

# =====================================================
# MODÈLES NORMAUX
# =====================================================
def normal_curve(points, values, length):
    x = np.linspace(0, 100, length)
    curve = np.interp(x, points, values)
    return gaussian_filter1d(curve, sigma=2)

def normal_hip(l):    return normal_curve([0,30,55,85,100],[30,0,-15,20,30],l)
def normal_knee(l):   return normal_curve([0,15,40,60,75,100],[5,18,3,35,60,5],l)
def normal_ankle(l):  return normal_curve([0,10,40,60,80,100],[0,-5,10,-17.5,0,0],l)
def normal_pelvis(l): return 5*np.sin(2*np.pi*np.linspace(0,1,l))

# =====================================================
# VIDEO PROCESSING
# =====================================================
def process_video(path, joints):
    cap = cv2.VideoCapture(path)
    res = {k: [] for k in ["Hanche G","Hanche D","Genou G","Genou D",
                           "Cheville G","Cheville D","Pelvis","Dos"]}
    best_frame, best_score = None, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        kp = detect_pose(frame)
        # Sélection frame représentative
        score = np.mean(kp[:,2])
        if score > best_score:
            best_score, best_frame = score, frame.copy()

        # angles
        res["Hanche G"].append(angle(kp[joints["Epaule G"],:2], kp[joints["Hanche G"],:2], kp[joints["Genou G"],:2]))
        res["Hanche D"].append(angle(kp[joints["Epaule D"],:2], kp[joints["Hanche D"],:2], kp[joints["Genou D"],:2]))

        res["Genou G"].append(angle(kp[joints["Hanche G"],:2], kp[joints["Genou G"],:2], kp[joints["Cheville G"],:2]))
        res["Genou D"].append(angle(kp[joints["Hanche D"],:2], kp[joints["Genou D"],:2], kp[joints["Cheville D"],:2]))

        res["Cheville G"].append(angle(kp[joints["Genou G"],:2], kp[joints["Cheville G"],:2], kp[joints["Cheville G"],:2]+[0,1]))
        res["Cheville D"].append(angle(kp[joints["Genou D"],:2], kp[joints["Cheville D"],:2], kp[joints["Cheville D"],:2]+[0,1]))

        pelvis = np.degrees(np.arctan2(
            kp[joints["Hanche D"],1]-kp[joints["Hanche G"],1],
            kp[joints["Hanche D"],0]-kp[joints["Hanche G"],0]))
        res["Pelvis"].append(pelvis)

        dos = angle(kp[joints["Epaule G"],:2],
                    (kp[joints["Hanche G"],:2]+kp[joints["Hanche D"],:2])/2,
                    kp[joints["Epaule D"],:2])
        res["Dos"].append(dos)

    cap.release()
    return res, best_frame

# =====================================================
# PDF EXPORT
# =====================================================
def export_pdf(info, images, table, best_img):
    path = os.path.join(tempfile.gettempdir(),"rapport_gaitscan.pdf")
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("<b>Bilan Analyse Cinématique</b>", styles["Title"]),
        Paragraph(f"Patient : {info['nom']} {info['prenom']}", styles["Normal"]),
        Paragraph(datetime.now().strftime("Date : %d/%m/%Y"), styles["Normal"]),
        Spacer(1,1*cm)
    ]

    if best_img:
        story.append(Paragraph("<b>Image de référence</b>", styles["Heading2"]))
        story.append(PDFImage(best_img, width=12*cm, height=7*cm))
        story.append(Spacer(1,0.5*cm))

    for k,v in images.items():
        story.append(Paragraph(f"<b>{k}</b>", styles["Heading3"]))
        story.append(PDFImage(v, width=16*cm, height=6*cm))
        story.append(Spacer(1,0.4*cm))

    story.append(Paragraph("<b>Résumé articulaire (°)</b>", styles["Heading2"]))
    t = Table([["Articulation","Min","Moy","Max"]] + table)
    t.setStyle(TableStyle([("GRID",(0,0),(-1,-1),1,colors.black)]))
    story.append(t)

    doc.build(story)
    return path

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("👤 Patient")
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")

    st.subheader("📹 Source")
    video = st.file_uploader("Vidéo",["mp4","avi"])
    live = st.checkbox("Caméra live")

    st.subheader("📐 Caméra")
    cam_pos = st.selectbox("Position",["Devant","Côté gauche","Côté droit"])

    st.subheader("⚙️ Paramètres")
    smoothing = st.slider("Filtrage", 0, 10, 2)
    show_norm = st.checkbox("Afficher norme",True)

# =====================================================
# MAIN
# =====================================================
ready = False
if live:
    cam = st.camera_input("Live")
    if cam: video, ready = cam, True
elif video:
    ready = True

if ready and st.button("▶ Lancer l'analyse"):
    joints = adjust_joints(cam_pos, BASE_JOINTS)

    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
    tmp.write(video.read())
    results, best_frame = process_video(tmp.name, joints)
    os.unlink(tmp.name)

    best_img_path = None
    if best_frame is not None:
        best_img_path = os.path.join(tempfile.gettempdir(),"best_frame.png")
        cv2.imwrite(best_img_path, best_frame)

    images, summary = {}, []

    ARTICS = [
        ("Hanche", ("Hanche G","Hanche D"), normal_hip),
        ("Genou", ("Genou G","Genou D"), normal_knee),
        ("Cheville", ("Cheville G","Cheville D"), normal_ankle)
    ]

    for name,(jg,jd),norm in ARTICS:
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[2,1]})
        for j,c in zip([jg,jd],["red","blue"]):
            s = bandpass_filter(results[j],smoothing)
            ax1.plot(s,label=j,color=c)
            summary.append([j,f"{min(results[j]):.1f}",f"{np.mean(results[j]):.1f}",f"{max(results[j]):.1f}"])
        ax1.set_title(f"{name} réel")
        ax1.legend()

        if show_norm:
            l = len(s)
            ax2.plot(norm(l),color="green")
            ax2.set_title("Norme")

        st.pyplot(fig)
        p=os.path.join(tempfile.gettempdir(),f"{name}.png")
        fig.savefig(p,bbox_inches="tight")
        images[name]=p
        plt.close(fig)

    # Pelvis
    fig, (ax1,ax2)=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[2,1]})
    s = bandpass_filter(results["Pelvis"], smoothing)
    ax1.plot(s, color="purple", label="Pelvis réel")
    summary.append(["Pelvis",f"{min(results['Pelvis']):.1f}",f"{np.mean(results['Pelvis']):.1f}",f"{max(results['Pelvis']):.1f}"])
    ax1.set_title("Pelvis réel")
    if show_norm:
        ax2.plot(normal_pelvis(len(s)), color="green")
        ax2.set_title("Norme")
    st.pyplot(fig)
    p=os.path.join(tempfile.gettempdir(),"Pelvis.png")
    fig.savefig(p,bbox_inches="tight")
    images["Pelvis"]=p
    plt.close(fig)

    # Dos
    fig = plt.figure(figsize=(10,4))
    s = bandpass_filter(results["Dos"], smoothing)
    plt.plot(s, color="green", label="Dos")
    summary.append(["Dos",f"{min(results['Dos']):.1f}",f"{np.mean(results['Dos']):.1f}",f"{max(results['Dos']):.1f}"])
    plt.title("Dos")
    plt.xlabel("Frame")
    plt.ylabel("Angle (°)")
    plt.legend()
    st.pyplot(fig)
    p=os.path.join(tempfile.gettempdir(),"Dos.png")
    fig.savefig(p,bbox_inches="tight")
    images["Dos"]=p
    plt.close(fig)

    # PDF
    pdf = export_pdf({"nom":nom,"prenom":prenom},images,summary,best_img_path)
    with open(pdf,"rb") as f:
        st.download_button("📥 Télécharger le PDF",f,"rapport_gaitscan.pdf")
