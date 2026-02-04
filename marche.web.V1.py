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
    """
    Retourne kp (17,3) en (x, y, score) normalisé [0..1] SUR L'IMAGE ORIGINALE (corrigé padding).
    """
    h, w = frame.shape[:2]

    # BGR -> RGB, uint8 [0..255]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tf = tf.convert_to_tensor(img, dtype=tf.uint8)

    input_size = 192

    # resize_with_pad -> float32 dans beaucoup de versions TF, donc on cast ensuite
    img_resized = tf.image.resize_with_pad(img_tf[None], input_size, input_size)
    img_resized = tf.cast(img_resized, tf.int32)  # ✅ IMPORTANT pour MoveNet Hub

    # Appel MoveNet
    outputs = movenet.signatures["serving_default"](img_resized)
    kp = outputs["output_0"].numpy()[0, 0]  # (17,3) [y, x, score] sur 192 paddé

    # Géométrie resize_with_pad
    scale = min(input_size / w, input_size / h)
    new_w = w * scale
    new_h = h * scale
    pad_x = (input_size - new_w) / 2.0
    pad_y = (input_size - new_h) / 2.0

    # coords en pixels 192
    y_192 = kp[:, 0] * input_size
    x_192 = kp[:, 1] * input_size
    score = kp[:, 2]

    # Dépad + renormalisation sur image originale
    x = (x_192 - pad_x) / (new_w + 1e-6)
    y = (y_192 - pad_y) / (new_h + 1e-6)

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)

    # On renvoie (x,y,score) pour rester cohérent avec ton code
    kp_xy = np.stack([x, y, score], axis=1)
    return kp_xy


# ==============================
# JOINTS
# ==============================
J = {
    "Epaule G": 5, "Epaule D": 6,
    "Hanche G": 11, "Hanche D": 12,
    "Genou G": 13, "Genou D": 14,
    "Cheville G": 15, "Cheville D": 16
}

def angle(a, b, c, joint_type="Hanche/Cheville"):
    ba = a - b
    bc = c - b
    # coord image -> y vers le bas, on inverse y pour angle "math"
    ba[1] *= -1
    bc[1] *= -1
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    ang = np.clip(np.degrees(np.arccos(cos_theta)), 0, 180)

    if joint_type == "Hanche":
        return 180 - ang
    elif joint_type == "Cheville":
        return 90 - (ang - 90)
    else:
        return ang

def angle_genou(a, b, c):
    ba = a - b
    bc = c - b
    ba[1] *= -1
    bc[1] *= -1
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    ang = np.clip(np.degrees(np.arccos(cos_theta)), 0, 180)
    return 180 - ang

# ==============================
# BANDPASS
# ==============================
def bandpass(sig, level, fs=FPS):
    low = 0.3 + level * 0.02
    high = max(6.0 - level * 0.25, low + 0.4)
    b, a = butter(2, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

# ==============================
# VIDEO PROCESS
# ==============================
def process_video(path):
    cap = cv2.VideoCapture(path)

    res = {
        "Hanche G": [], "Hanche D": [],
        "Genou G": [], "Genou D": [],
        "Cheville G": [], "Cheville D": [],
        "Pelvis": [], "Dos": []
    }
    heel_y_D, frames = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        kp = detect_pose(frame)  # kp en (x,y,score) normalisé sur l'image originale
        frames.append(frame.copy())

        # angles (on utilise seulement x,y)
        res["Hanche G"].append(angle(kp[J["Epaule G"], :2], kp[J["Hanche G"], :2], kp[J["Genou G"], :2], "Hanche"))
        res["Hanche D"].append(angle(kp[J["Epaule D"], :2], kp[J["Hanche D"], :2], kp[J["Genou D"], :2], "Hanche"))

        res["Genou G"].append(angle_genou(kp[J["Hanche G"], :2], kp[J["Genou G"], :2], kp[J["Cheville G"], :2]))
        res["Genou D"].append(angle_genou(kp[J["Hanche D"], :2], kp[J["Genou D"], :2], kp[J["Cheville D"], :2]))

        # cheville: point vertical artificiel "vers le bas"
        res["Cheville G"].append(angle(kp[J["Genou G"], :2], kp[J["Cheville G"], :2], kp[J["Cheville G"], :2] + [0, 0.12], "Cheville"))
        res["Cheville D"].append(angle(kp[J["Genou D"], :2], kp[J["Cheville D"], :2], kp[J["Cheville D"], :2] + [0, 0.12], "Cheville"))

        pelvis = kp[J["Hanche D"], :2] - kp[J["Hanche G"], :2]
        res["Pelvis"].append(np.degrees(np.arctan2(pelvis[1], pelvis[0])))

        mid_hip = (kp[11, :2] + kp[12, :2]) / 2
        mid_sh = (kp[5, :2] + kp[6, :2]) / 2
        res["Dos"].append(angle(mid_sh, mid_hip, mid_hip + [0, -0.12]))

        heel_y_D.append(kp[J["Cheville D"], 1])  # y
    cap.release()
    return res, heel_y_D, frames

# ==============================
# CYCLE DETECTION
# ==============================
def detect_cycle(heel_y):
    inv = -np.array(heel_y)
    peaks, _ = find_peaks(inv, distance=FPS // 2, prominence=np.std(inv) * 0.3)
    if len(peaks) >= 2:
        return peaks[0], peaks[1]
    return 0, len(heel_y) - 1

# ==============================
# NORMES
# ==============================
def norm_curve(joint, n):
    x = np.linspace(0, 100, n)
    if joint == "Genou":
        return np.interp(x, [0, 15, 40, 60, 80, 100], [5, 15, 5, 40, 60, 5])
    if joint == "Hanche":
        return np.interp(x, [0, 30, 60, 100], [30, 0, -10, 30])
    if joint == "Cheville":
        return np.interp(x, [0, 10, 50, 70, 100], [0, -5, 10, -15, 0])
    return np.zeros(n)

# ==============================
# PHOTO ANNOTATION
# ==============================
def angle_for_annotation(a, b, c):
    ba = a - b
    bc = c - b
    # inverse y pour angle math
    ba[1] *= -1
    bc[1] *= -1
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return int(np.clip(np.degrees(np.arccos(cos_theta)), 0, 180))

def annotate_frame(frame, kp, conf=0.30):
    """
    kp (17,3) en (x,y,score) normalisé sur image originale.
    Dessine seulement si les 3 points du joint ont score >= conf.
    Angles + gros et lisibles.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # style "plus gros"
    line_th = 4
    circle_r = 7
    text_scale = 1.1
    text_th = 3

    joints = {
        "Hanche G": (5, 11, 13),
        "Hanche D": (6, 12, 14),
        "Genou G":  (11, 13, 15),
        "Genou D":  (12, 14, 16),
        # cheville: on fabrique le 3e point vertical
        "Cheville G": (13, 15, None),
        "Cheville D": (14, 16, None),
    }

    for name, (a_idx, b_idx, c_idx) in joints.items():
        a = kp[a_idx, :2]
        b = kp[b_idx, :2]
        sa = kp[a_idx, 2]
        sb = kp[b_idx, 2]

        if c_idx is None:
            c = b + np.array([0.0, 0.12], dtype=np.float32)
            sc = sb
        else:
            c = kp[c_idx, :2]
            sc = kp[c_idx, 2]

        # seuil confiance
        if min(sa, sb, sc) < conf:
            continue

        ang = angle_for_annotation(a, b, c)

        a_px = (int(a[0] * w), int(a[1] * h))
        b_px = (int(b[0] * w), int(b[1] * h))
        c_px = (int(c[0] * w), int(c[1] * h))

        cv2.line(annotated, a_px, b_px, (0, 255, 0), line_th)
        cv2.line(annotated, c_px, b_px, (0, 255, 0), line_th)
        cv2.circle(annotated, b_px, circle_r, (0, 0, 255), -1)

        label = f"{ang}°"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_th)

        tx, ty = b_px[0] + 10, b_px[1] - 10
        cv2.rectangle(annotated, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
        cv2.putText(
            annotated, label, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_th, cv2.LINE_AA
        )

    return annotated

# ==============================
# PDF EXPORT
# ==============================
def export_pdf(patient, keyframe, figures, table_data, annotated_images=None):
    path = os.path.join(tempfile.gettempdir(), "rapport_gaitscan.pdf")
    doc = SimpleDocTemplate(
        path, pagesize=A4, rightMargin=2 * cm,
        leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique</b>", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Caméra :</b> {patient.get('camera', 'N/A')}", styles["Normal"]
    ))
    story.append(Paragraph(
        f"<b>Phase du pas basée sur :</b> {patient.get('phase_cote', 'N/A')}", styles["Normal"]
    ))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Image représentative du cycle</b>", styles["Heading2"]))
    story.append(PDFImage(keyframe, width=16 * cm, height=8 * cm))
    story.append(Spacer(1, 0.6 * cm))

    story.append(Paragraph("<b>Analyse articulaire</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))
    for joint, img_path in figures.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles["Heading3"]))
        story.append(PDFImage(img_path, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("<b>Synthèse des angles (°) – Gauche / Droite</b>", styles["Heading2"]))
    table = Table([["Articulation", "Min", "Moyenne", "Max"]] + table_data,
                  colWidths=[5 * cm, 3 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))
    story.append(table)

    if annotated_images:
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("<b>Photos annotées avec angles</b>", styles["Heading2"]))
        for img_path in annotated_images:
            story.append(PDFImage(img_path, width=16 * cm, height=8 * cm))
            story.append(Spacer(1, 0.3 * cm))

    doc.build(story)
    return path

# ==============================
# STREAMLIT INTERFACE
# ==============================
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    smooth = st.slider("Lissage band-pass", 0, 10, 3)
    src = st.radio("Source", ["Vidéo", "Caméra"])
    camera_pos = st.selectbox("Position de la caméra", ["Devant", "Droite", "Gauche"])
    phase_cote = st.selectbox("Phase du pas basée sur :", ["Droite", "Gauche", "Les deux"])
    conf = st.slider("Seuil confiance keypoints (affichage angles)", 0.05, 0.80, 0.30, 0.05)

video = st.file_uploader("Vidéo", ["mp4", "avi", "mov"]) if src == "Vidéo" else st.camera_input("Caméra")

# ==============================
# ANALYSE
# ==============================
annotated_images = []

if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heel_y, frames = process_video(tmp.name)
    os.unlink(tmp.name)

    # Phase du pas
    if phase_cote == "Droite":
        heel_f_ref = bandpass(np.array(data["Cheville D"]), smooth)
        c0, c1 = detect_cycle(heel_f_ref)
        phase_colors = [(c0, c1, "blue")]
    elif phase_cote == "Gauche":
        heel_f_ref = bandpass(np.array(data["Cheville G"]), smooth)
        c0, c1 = detect_cycle(heel_f_ref)
        phase_colors = [(c0, c1, "orange")]
    else:
        heel_f_D = bandpass(np.array(data["Cheville D"]), smooth)
        heel_f_G = bandpass(np.array(data["Cheville G"]), smooth)
        c0_D, c1_D = detect_cycle(heel_f_D)
        c0_G, c1_G = detect_cycle(heel_f_G)
        phase_colors = [(c0_D, c1_D, "blue"), (c0_G, c1_G, "orange")]

    # Keyframe milieu de vidéo (image brute)
    key_img = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(key_img, frames[len(frames) // 2])

    figs, table_data = {}, []

    for joint in ["Hanche", "Genou", "Cheville"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
        g = bandpass(np.array(data[f"{joint} G"]), smooth)
        d = bandpass(np.array(data[f"{joint} D"]), smooth)
        n = norm_curve(joint, len(g))

        ax1.plot(g, label="Gauche", color="red")
        ax1.plot(d, label="Droite", color="blue")
        for cc0, cc1, color in phase_colors:
            ax1.axvspan(cc0, cc1, color=color, alpha=0.3)
        ax1.set_title(f"{joint} – Analyse")
        ax1.legend()

        ax2.plot(n, color="green")
        ax2.set_title("Norme")

        st.pyplot(fig)
        img = os.path.join(tempfile.gettempdir(), f"{joint}.png")
        fig.savefig(img, bbox_inches="tight")
        plt.close(fig)
        figs[joint] = img

        table_data.append([joint + " Gauche", f"{g.min():.1f}", f"{g.mean():.1f}", f"{g.max():.1f}"])
        table_data.append([joint + " Droite", f"{d.min():.1f}", f"{d.mean():.1f}", f"{d.max():.1f}"])

    # ==============================
    # CAPTURER QUELQUES PHOTOS AVEC ANGLES
    # ==============================
    num_photos = st.slider("Nombre de photos à capturer depuis la vidéo", 1, 10, 3)
    total_frames = len(frames)
    frames_to_capture = np.linspace(0, total_frames - 1, num_photos, dtype=int)

    for i, f_idx in enumerate(frames_to_capture):
        frame = frames[f_idx]
        kp = detect_pose(frame)
        annotated = annotate_frame(frame, kp, conf=conf)
        path = os.path.join(tempfile.gettempdir(), f"annotated_{i}.png")
        cv2.imwrite(path, annotated)
        annotated_images.append(path)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Image annotée {i + 1}")

    # ==============================
    # PDF
    # ==============================
    pdf_path = export_pdf(
        patient={
            "nom": nom,
            "prenom": prenom,
            "camera": camera_pos,
            "phase_cote": phase_cote
        },
        keyframe=key_img,
        figures=figs,
        table_data=table_data,
        annotated_images=annotated_images
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📄 Télécharger le rapport PDF",
            f,
            file_name=f"GaitScan_{nom}_{prenom}.pdf",
            mime="application/pdf"
        )


