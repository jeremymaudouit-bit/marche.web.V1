# ==============================
# IMPORTS
# ==============================
import streamlit as st
import cv2, os, tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, find_peaks

import mediapipe as mp

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
st.set_page_config("GaitScan Pro (MediaPipe)", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique (MediaPipe Pose)")
FPS = 30

# ==============================
# MEDIAPIPE POSE
# ==============================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    # model_complexity 1 = bon compromis perf/qualité
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

def detect_pose(frame_bgr):
    """
    Retourne un dict {name: (x,y,vis)} en coords normalisées [0..1].
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(landmark_enum):
        p = lm[int(landmark_enum)]
        return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)

    out = {}
    out["Epaule G"], out["Epaule G vis"] = pt(L.LEFT_SHOULDER)
    out["Epaule D"], out["Epaule D vis"] = pt(L.RIGHT_SHOULDER)

    out["Hanche G"], out["Hanche G vis"] = pt(L.LEFT_HIP)
    out["Hanche D"], out["Hanche D vis"] = pt(L.RIGHT_HIP)

    out["Genou G"], out["Genou G vis"] = pt(L.LEFT_KNEE)
    out["Genou D"], out["Genou D vis"] = pt(L.RIGHT_KNEE)

    out["Cheville G"], out["Cheville G vis"] = pt(L.LEFT_ANKLE)
    out["Cheville D"], out["Cheville D vis"] = pt(L.RIGHT_ANKLE)

    # Points du pied (clé pour "vrai" angle tibia–pied)
    out["Orteil G"], out["Orteil G vis"] = pt(L.LEFT_FOOT_INDEX)
    out["Orteil D"], out["Orteil D vis"] = pt(L.RIGHT_FOOT_INDEX)

    out["Talon G"], out["Talon G vis"] = pt(L.LEFT_HEEL)
    out["Talon D"], out["Talon D vis"] = pt(L.RIGHT_HEEL)

    return out

# ==============================
# ANGLES
# ==============================
def angle_between_points(a, b, c):
    """
    Angle ABC (degrés). a,b,c en (x,y) normalisé.
    """
    ba = a - b
    bc = c - b
    # inversion y (repère image -> repère math)
    ba[1] *= -1
    bc[1] *= -1
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cos_theta = np.clip(cos_theta, -1, 1)
    ang = np.degrees(np.arccos(cos_theta))
    return float(np.clip(ang, 0, 180))

def angle_hanche(epaule, hanche, genou):
    # convention "flexion hanche" proche de ton ancien code : 180 - angle
    return 180.0 - angle_between_points(epaule, hanche, genou)

def angle_genou(hanche, genou, cheville):
    return 180.0 - angle_between_points(hanche, genou, cheville)

def angle_cheville(genou, cheville, orteil):
    """
    "Vrai" angle tibia–pied : angle entre (genou->cheville) et (orteil->cheville).
    Ici c = orteil, b = cheville, a = genou
    """
    return angle_between_points(genou, cheville, orteil)

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
def process_video(path, conf=0.30):
    cap = cv2.VideoCapture(path)

    res = {
        "Hanche G": [], "Hanche D": [],
        "Genou G": [], "Genou D": [],
        "Cheville G": [], "Cheville D": [],
        "Pelvis": [], "Dos": []
    }
    heel_y_D = []
    frames = []
    pose_cache = []  # garde les landmarks pour annotation sans rerun

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame.copy())

        kp = detect_pose(frame)
        pose_cache.append(kp)

        if kp is None:
            # ajoute des NaN pour garder l'alignement temporel
            for k in ["Hanche G","Hanche D","Genou G","Genou D","Cheville G","Cheville D","Pelvis","Dos"]:
                res[k].append(np.nan)
            heel_y_D.append(np.nan)
            continue

        # petit helper pour tester visibilité
        def ok(name):
            return kp.get(f"{name} vis", 0.0) >= conf

        # angles principaux
        if ok("Epaule G") and ok("Hanche G") and ok("Genou G"):
            res["Hanche G"].append(angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"]))
        else:
            res["Hanche G"].append(np.nan)

        if ok("Epaule D") and ok("Hanche D") and ok("Genou D"):
            res["Hanche D"].append(angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"]))
        else:
            res["Hanche D"].append(np.nan)

        if ok("Hanche G") and ok("Genou G") and ok("Cheville G"):
            res["Genou G"].append(angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"]))
        else:
            res["Genou G"].append(np.nan)

        if ok("Hanche D") and ok("Genou D") and ok("Cheville D"):
            res["Genou D"].append(angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"]))
        else:
            res["Genou D"].append(np.nan)

        # cheville (tibia–pied) : genou, cheville, orteil (foot index)
        if ok("Genou G") and ok("Cheville G") and ok("Orteil G"):
            res["Cheville G"].append(angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Orteil G"]))
        else:
            res["Cheville G"].append(np.nan)

        if ok("Genou D") and ok("Cheville D") and ok("Orteil D"):
            res["Cheville D"].append(angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Orteil D"]))
        else:
            res["Cheville D"].append(np.nan)

        # pelvis angle (ligne hanche G->D)
        if ok("Hanche G") and ok("Hanche D"):
            pelvis = kp["Hanche D"] - kp["Hanche G"]
            # repère math
            pelvis_m = np.array([pelvis[0], -pelvis[1]], dtype=np.float32)
            res["Pelvis"].append(float(np.degrees(np.arctan2(pelvis_m[1], pelvis_m[0]))))
        else:
            res["Pelvis"].append(np.nan)

        # dos : angle entre (mid_shoulder -> mid_hip) et verticale
        if ok("Hanche G") and ok("Hanche D") and ok("Epaule G") and ok("Epaule D"):
            mid_hip = (kp["Hanche G"] + kp["Hanche D"]) / 2
            mid_sh = (kp["Epaule G"] + kp["Epaule D"]) / 2
            # verticale artificielle
            vertical = mid_hip + np.array([0.0, -0.12], dtype=np.float32)
            res["Dos"].append(angle_between_points(mid_sh, mid_hip, vertical))
        else:
            res["Dos"].append(np.nan)

        # heel y pour cycle (utilise talon droit)
        if ok("Talon D"):
            heel_y_D.append(float(kp["Talon D"][1]))
        else:
            heel_y_D.append(np.nan)

    cap.release()
    return res, heel_y_D, frames, pose_cache

# ==============================
# CYCLE DETECTION
# ==============================
def detect_cycle(heel_y):
    y = np.array(heel_y, dtype=np.float32)
    # remplace NaN par interpolation simple
    if np.isnan(y).any():
        idx = np.arange(len(y))
        good = ~np.isnan(y)
        if good.sum() >= 2:
            y = np.interp(idx, idx[good], y[good])
        else:
            return 0, len(y) - 1

    inv = -y
    peaks, _ = find_peaks(inv, distance=FPS // 2, prominence=np.std(inv) * 0.3)
    if len(peaks) >= 2:
        return int(peaks[0]), int(peaks[1])
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
        return np.interp(x, [0, 10, 50, 70, 100], [5, 10, 25, 10, 5])
    return np.zeros(n)

# ==============================
# ANNOTATION (plus gros + lisible)
# ==============================
def draw_angle(frame, p1, p2, p3, ang, text_scale=1.1, text_th=3, line_th=4, circle_r=7):
    h, w = frame.shape[:2]
    a_px = (int(p1[0] * w), int(p1[1] * h))
    b_px = (int(p2[0] * w), int(p2[1] * h))
    c_px = (int(p3[0] * w), int(p3[1] * h))

    cv2.line(frame, a_px, b_px, (0, 255, 0), line_th)
    cv2.line(frame, c_px, b_px, (0, 255, 0), line_th)
    cv2.circle(frame, b_px, circle_r, (0, 0, 255), -1)

    label = f"{int(round(ang))}°"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_th)
    tx, ty = b_px[0] + 10, b_px[1] - 10
    cv2.rectangle(frame, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_th, cv2.LINE_AA)

def annotate_frame(frame, kp, conf=0.30):
    if kp is None:
        return frame

    out = frame.copy()
    # style
    text_scale = 1.1
    text_th = 3
    line_th = 4
    circle_r = 7

    def vis(name): return kp.get(f"{name} vis", 0.0)
    def ok(*names): return min(vis(n) for n in names) >= conf

    # Hanche
    if ok("Epaule G","Hanche G","Genou G"):
        ang = angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"])
        draw_angle(out, kp["Epaule G"], kp["Hanche G"], kp["Genou G"], ang, text_scale, text_th, line_th, circle_r)

    if ok("Epaule D","Hanche D","Genou D"):
        ang = angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"])
        draw_angle(out, kp["Epaule D"], kp["Hanche D"], kp["Genou D"], ang, text_scale, text_th, line_th, circle_r)

    # Genou
    if ok("Hanche G","Genou G","Cheville G"):
        ang = angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"])
        draw_angle(out, kp["Hanche G"], kp["Genou G"], kp["Cheville G"], ang, text_scale, text_th, line_th, circle_r)

    if ok("Hanche D","Genou D","Cheville D"):
        ang = angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"])
        draw_angle(out, kp["Hanche D"], kp["Genou D"], kp["Cheville D"], ang, text_scale, text_th, line_th, circle_r)

    # Cheville tibia–pied (genou-cheville-orteil)
    if ok("Genou G","Cheville G","Orteil G"):
        ang = angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Orteil G"])
        draw_angle(out, kp["Genou G"], kp["Cheville G"], kp["Orteil G"], ang, text_scale, text_th, line_th, circle_r)

    if ok("Genou D","Cheville D","Orteil D"):
        ang = angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Orteil D"])
        draw_angle(out, kp["Genou D"], kp["Cheville D"], kp["Orteil D"], ang, text_scale, text_th, line_th, circle_r)

    return out

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

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique (MediaPipe)</b>", styles["Title"]))
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
    conf = st.slider("Seuil confiance landmarks (affichage/mesure)", 0.05, 0.95, 0.30, 0.05)

video = st.file_uploader("Vidéo", ["mp4", "avi", "mov"]) if src == "Vidéo" else st.camera_input("Caméra")

# ==============================
# ANALYSE
# ==============================
annotated_images = []

if video and st.button("▶ Lancer l'analyse"):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp.write(video.read())
    tmp.close()

    data, heel_y, frames, pose_cache = process_video(tmp.name, conf=conf)
    os.unlink(tmp.name)

    # Phase du pas : basé sur talon droit (plus logique qu'avant)
    c0, c1 = detect_cycle(heel_y)
    phase_colors = [(c0, c1, "blue")] if phase_cote != "Gauche" else [(c0, c1, "orange")]

    key_img = os.path.join(tempfile.gettempdir(), "keyframe.png")
    cv2.imwrite(key_img, frames[len(frames) // 2])

    figs, table_data = {}, []

    for joint in ["Hanche", "Genou", "Cheville"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

        g = np.array(data[f"{joint} G"], dtype=np.float32)
        d = np.array(data[f"{joint} D"], dtype=np.float32)

        # Interp NaN avant bandpass
        def nan_interp(x):
            x = x.copy()
            idx = np.arange(len(x))
            good = ~np.isnan(x)
            if good.sum() >= 2:
                return np.interp(idx, idx[good], x[good])
            return np.nan_to_num(x, nan=0.0)

        g_f = bandpass(nan_interp(g), smooth)
        d_f = bandpass(nan_interp(d), smooth)
        n = norm_curve(joint, len(g_f))

        ax1.plot(g_f, label="Gauche", color="red")
        ax1.plot(d_f, label="Droite", color="blue")
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

        table_data.append([joint + " Gauche", f"{np.nanmin(g_f):.1f}", f"{np.nanmean(g_f):.1f}", f"{np.nanmax(g_f):.1f}"])
        table_data.append([joint + " Droite", f"{np.nanmin(d_f):.1f}", f"{np.nanmean(d_f):.1f}", f"{np.nanmax(d_f):.1f}"])

    # ==============================
    # CAPTURER QUELQUES PHOTOS AVEC ANGLES
    # ==============================
    num_photos = st.slider("Nombre de photos à capturer depuis la vidéo", 1, 10, 3)
    total_frames = len(frames)
    frames_to_capture = np.linspace(0, total_frames - 1, num_photos, dtype=int)

    for i, f_idx in enumerate(frames_to_capture):
        frame = frames[f_idx]
        kp = pose_cache[f_idx]
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
