# ==============================
# IMPORTS
# ==============================
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
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

# ==============================
# CONFIG
# ==============================
st.set_page_config("GaitScan Pro", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique")
FPS = 30

# ==============================
# MOVENET
# ==============================
@st.cache_resource
def load_movenet():
    return hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

movenet = load_movenet()

def detect_pose(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tf.image.resize_with_pad(img[None], 192, 192)
    out = movenet.signatures["serving_default"](tf.cast(img, tf.int32))
    return out["output_0"].numpy()[0, 0]

# ==============================
# JOINTS
# ==============================
J = {
    "Epaule G":5, "Epaule D":6,
    "Hanche G":11, "Hanche D":12,
    "Genou G":13, "Genou D":14,
    "Cheville G":15, "Cheville D":16
}

def angle(a,b,c):
    ba, bc = a-b, c-b
    return np.degrees(
        np.arccos(
            np.clip(
                np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6),
                -1,1
            )
        )
    )

# ==============================
# BANDPASS
# ==============================
def bandpass(sig, level, fs=FPS):
    low = 0.3 + level*0.02
    high = max(6.0 - level*0.25, low+0.4)
    b,a = butter(2, [low/(fs/2), high/(fs/2)], btype="band")
    return filtfilt(b,a,sig)

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path):
    cap = cv2.VideoCapture(path)

    res = {
        "Hanche G":[], "Hanche D":[],
        "Genou G":[], "Genou D":[],
        "Cheville G":[], "Cheville D":[],
        "Pelvis":[], "Dos":[]
    }
    heel_y_D, frames = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        kp = detect_pose(frame)
        frames.append(frame.copy())

        # ANGLES
        res["Hanche G"].append(angle(kp[J["Epaule G"],:2], kp[J["Hanche G"],:2], kp[J["Genou G"],:2]))
        res["Hanche D"].append(angle(kp[J["Epaule D"],:2], kp[J["Hanche D"],:2], kp[J["Genou D"],:2]))

        res["Genou G"].append(angle(kp[J["Hanche G"],:2], kp[J["Genou G"],:2], kp[J["Cheville G"],:2]))
        res["Genou D"].append(angle(kp[J["Hanche D"],:2], kp[J["Genou D"],:2], kp[J["Cheville D"],:2]))

        res["Cheville G"].append(angle(kp[J["Genou G"],:2], kp[J["Cheville G"],:2], kp[J["Cheville G"],:2]+[0,1]))
        res["Cheville D"].append(angle(kp[J["Genou D"],:2], kp[J["Cheville D"],:2], kp[J["Cheville D"],:2]+[0,1]))

        pelvis = kp[J["Hanche D"],:2] - kp[J["Hanche G"],:2]
        res["Pelvis"].append(np.degrees(np.arctan2(pelvis[1], pelvis[0])))

        mid_hip = (kp[11,:2]+kp[12,:2])/2
        mid_sh = (kp[5,:2]+kp[6,:2])/2
        res["Dos"].append(angle(mid_sh, mid_hip, mid_hip+[0,-1]))

        heel_y_D.append(kp[J["Cheville D"],1])

    cap.release()
    return res, heel_y_D, frames

# ==============================
# CYCLE DETECTION
# ==============================
def detect_cycle(heel_y):
    inv = -np.array(heel_y)
    peaks,_ = find_peaks(inv, distance=FPS//2, prominence=np.std(inv)*0.3)
    if len(peaks)>=2:
        return peaks[0], peaks[1]
    return 0, len(heel_y)-1

# ==============================
# NORMES
# ==============================
def norm_curve(joint,n):
    x = np.linspace(0,100,n)
    if joint=="Genou":
        return np.interp(x,[0,15,40,60,80,100],[5,15,5,40,60,5])
    if joint=="Hanche":
        return np.interp(x,[0,30,60,100],[30,0,-10,30])
    if joint=="Cheville":
        return np.interp(x,[0,10,50,70,100],[0,-5,10,-15,0])
    return np.zeros(n)

# ==============================
# INTERFACE
# ==============================
with st.sidebar:
    nom = st.text_input("Nom","DURAND")
    prenom = st.text_input("Prénom","Jean")
    smooth = st.slider("Lissage band-pass",0,10,3)
    src = st.radio("Source",["Vidéo","Caméra"])

video = st.file_uploader("Vidéo",["mp4","avi","mov"]) if src=="Vidéo" else st.camera_input("Caméra")

# ==============================
# ANALYSE
# ==============================
if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heel_y, frames = process_video(tmp.name)
    os.unlink(tmp.name)

    heel_f = bandpass(np.array(heel_y), smooth)
    c0, c1 = detect_cycle(heel_f)

    key_img = os.path.join(tempfile.gettempdir(),"keyframe.png")
    cv2.imwrite(key_img, frames[(c0+c1)//2])

    figs, table = {}, []

    for joint in ["Hanche","Genou","Cheville"]:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4),gridspec_kw={"width_ratios":[2,1]})

        g = bandpass(np.array(data[f"{joint} G"]), smooth)
        d = bandpass(np.array(data[f"{joint} D"]), smooth)
        n = norm_curve(joint,len(g))

        ax1.plot(g,label="Gauche",color="red")
        ax1.plot(d,label="Droite",color="blue")
        ax1.axvspan(c0,c1,color="orange",alpha=0.2)
        ax1.set_title(f"{joint} – Analyse")
        ax1.legend()

        ax2.plot(n,color="green")
        ax2.set_title("Norme")

        st.pyplot(fig)

        img = os.path.join(tempfile.gettempdir(),f"{joint}.png")
        fig.savefig(img,bbox_inches="tight")
        plt.close(fig)
        figs[joint]=img

        table.append([joint,
            f"{min(g.min(),d.min()):.1f}",
            f"{(g.mean()+d.mean())/2:.1f}",
            f"{max(g.max(),d.max()):.1f}"
        ])

    # PDF (inchangé)
    # → je peux te le réinjecter tel quel au prochain message
