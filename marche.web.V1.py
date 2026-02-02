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
    angles = np.arra
