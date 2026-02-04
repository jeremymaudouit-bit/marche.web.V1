import streamlit as st
st.set_page_config(page_title="Analyseur Postural Pro (MediaPipe)", layout="wide")

import os
import tempfile
import numpy as np
import cv2
from PIL import Image
import math
from fpdf import FPDF
from datetime import datetime

import mediapipe as mp

st.title("🧍 Analyseur Postural Pro (MediaPipe)")
st.markdown("---")

# ================= 1. CHARGEMENT MEDIAPIPE =================
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=True,     # image fixe
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()

# ================= 2. OUTILS =================
def rotate_if_landscape(img_np):
    if img_np.shape[1] > img_np.shape[0]:
        img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
    return img_np

def calculate_angle(p1, p2, p3):
    """
    Angle au point p2 entre segments p2->p1 et p2->p3
    """
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]], dtype=float)
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]], dtype=float)
    dot = float(np.dot(v1, v2))
    mag = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if mag == 0:
        return 0.0
    return math.degrees(math.acos(np.clip(dot / mag, -1, 1)))

def femur_tibia_knee_angle(hip, knee, ankle):
    """
    Genou = angle fémur–tibia => Hanche–Genou–Cheville
    (Angle au genou)
    """
    return calculate_angle(hip, knee, ankle)

def tibia_rearfoot_ankle_angle(knee, ankle, heel):
    """
    Cheville = angle tibia – arrière-pied
    Tibia: segment cheville->genou
    Arrière-pied: segment cheville->talon
    => angle au niveau de la cheville: Genou–Cheville–Talon
    """
    return calculate_angle(knee, ankle, heel)

def generate_pdf(data, img_np):
    pdf = FPDF()
    pdf.add_page()

    # En-tête
    pdf.set_fill_color(31, 73, 125)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 24)
    pdf.cell(0, 20, "BILAN POSTURAL IA", ln=True, align="C")

    # Infos Patient
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(25)
    pdf.cell(100, 10, f"Patient : {data['Nom']}", ln=0)
    pdf.set_font("Arial", '', 11)
    pdf.cell(90, 10, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1, align="R")
    pdf.line(10, 68, 200, 68)
    pdf.ln(5)

    # Image temporaire
    img_pil = Image.fromarray(img_np)
    tmp_img = os.path.join(tempfile.gettempdir(), "temp_analysis.png")
    img_pil.save(tmp_img)
    pdf.image(tmp_img, x=60, w=90)
    pdf.ln(5)

    # Tableau
    pdf.set_font("Arial", 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(110, 10, "Indicateur de Mesure", 1, 0, 'L', True)
    pdf.cell(80, 10, "Valeur", 1, 1, 'C', True)

    pdf.set_font("Arial", '', 11)
    for k, v in data.items():
        if k != "Nom":
            pdf.cell(110, 9, f" {k}", 1, 0, 'L')
            pdf.cell(80, 9, f" {v}", 1, 1, 'C')

    # Footer
    pdf.set_y(-25)
    pdf.set_font("Arial", 'I', 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 10, "Document généré par Analyseur Postural Pro - Usage indicatif uniquement.", align="C")

    filename = f"Bilan_{data['Nom'].replace(' ', '_')}.pdf"
    pdf.output(filename)

    # Nettoyage
    if os.path.exists(tmp_img):
        os.remove(tmp_img)

    return filename

def safe_point(lm, landmark_enum, w, h):
    p = lm[landmark_enum.value]
    return np.array([p.x * w, p.y * h], dtype=np.float32), float(p.visibility)

# ================= 3. UI =================
with st.sidebar:
    st.header("👤 Dossier Patient")
    nom = st.text_input("Nom complet", value="Anonyme")
    taille_cm = st.number_input("Taille (cm)", min_value=100, max_value=220, value=170)

    # ✅ Choix manuel (pas d'automatique)
    vue = st.selectbox("Vue de la photo", ["Face", "Dos"], index=0)

    st.divider()
    source = st.radio("Source de l'image", ["📷 Caméra", "📁 Téléverser une photo"])

col_input, col_result = st.columns([1, 1])

image_data = None
with col_input:
    if source == "📷 Caméra":
        st.write("La caméra par défaut du navigateur sera utilisée.")
        image_data = st.camera_input("Capturez la posture")
    else:
        image_data = st.file_uploader("Format JPG/PNG", type=["jpg", "png", "jpeg"])

# ================= 4. ANALYSE =================
if image_data:
    if isinstance(image_data, Image.Image):
        img = image_data.convert("RGB")
        img_np = np.array(img)
    else:
        img = Image.open(image_data).convert("RGB")
        img_np = np.array(img)

    img_np = rotate_if_landscape(img_np)
    h, w, _ = img_np.shape

    if st.button("⚙️ LANCER L'ANALYSE BIOMÉCANIQUE", use_container_width=True):
        with st.spinner("Détection de la posture (MediaPipe)..."):
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            res = pose.process(img_rgb)

            if not res.pose_landmarks:
                st.error("Aucune pose détectée. Photo plus nette, en pied, bien centrée.")
            else:
                lm = res.pose_landmarks.landmark
                L = mp_pose.PoseLandmark

                # Points + visibilité
                LS, vis_LS = safe_point(lm, L.LEFT_SHOULDER, w, h)
                RS, vis_RS = safe_point(lm, L.RIGHT_SHOULDER, w, h)
                LH, vis_LH = safe_point(lm, L.LEFT_HIP, w, h)
                RH, vis_RH = safe_point(lm, L.RIGHT_HIP, w, h)
                LK, vis_LK = safe_point(lm, L.LEFT_KNEE, w, h)
                RK, vis_RK = safe_point(lm, L.RIGHT_KNEE, w, h)
                LA, vis_LA = safe_point(lm, L.LEFT_ANKLE, w, h)
                RA, vis_RA = safe_point(lm, L.RIGHT_ANKLE, w, h)
                LHE, vis_LHE = safe_point(lm, L.LEFT_HEEL, w, h)
                RHE, vis_RHE = safe_point(lm, L.RIGHT_HEEL, w, h)

                # --- Inclinaison épaules / bassin (horizontale)
                raw_shoulder_angle = math.degrees(math.atan2(LS[1]-RS[1], LS[0]-RS[0]))
                shoulder_angle = abs(raw_shoulder_angle)
                if shoulder_angle > 90:
                    shoulder_angle = abs(shoulder_angle - 180)

                raw_hip_angle = math.degrees(math.atan2(LH[1]-RH[1], LH[0]-RH[0]))
                hip_angle = abs(raw_hip_angle)
                if hip_angle > 90:
                    hip_angle = abs(hip_angle - 180)

                # --- Genou : fémur-tibia (Hanche-Genou-Cheville)
                knee_l = femur_tibia_knee_angle(LH, LK, LA)
                knee_r = femur_tibia_knee_angle(RH, RK, RA)

                # --- Cheville : tibia / arrière-pied (Genou-Cheville-Talon)
                ankle_l = tibia_rearfoot_ankle_angle(LK, LA, LHE)
                ankle_r = tibia_rearfoot_ankle_angle(RK, RA, RHE)

                # --- Échelle mm/pixel (approx) (épaules -> chevilles)
                px_height = max(LA[1], RA[1]) - min(LS[1], RS[1])
                mm_per_px = (float(taille_cm) * 10.0) / px_height if px_height > 0 else 0.0
                diff_shoulders_mm = abs(LS[1] - RS[1]) * mm_per_px
                diff_hips_mm = abs(LH[1] - RH[1]) * mm_per_px

                # --- côté le plus bas (y plus grand = plus bas)
                shoulder_lower = "Gauche" if LS[1] > RS[1] else "Droite"
                hip_lower = "Gauche" if LH[1] > RH[1] else "Droite"

                # Inverser si photo de DOS (comme ton ancien code)
                if vue == "Dos":
                    shoulder_lower = "Droite" if shoulder_lower == "Gauche" else "Gauche"
                    hip_lower = "Droite" if hip_lower == "Gauche" else "Gauche"

                # ================= ANNOTATION IMAGE =================
                annotated = img_np.copy()

                # Points (visuels)
                pts = [LS, RS, LH, RH, LK, RK, LA, RA, LHE, RHE]
                for p in pts:
                    cv2.circle(annotated, tuple(p.astype(int)), 7, (0, 255, 0), -1)

                # Lignes épaules/bassin
                cv2.line(annotated, tuple(LS.astype(int)), tuple(RS.astype(int)), (255, 0, 0), 3)
                cv2.line(annotated, tuple(LH.astype(int)), tuple(RH.astype(int)), (255, 0, 0), 3)

                # Lignes segments genou/cheville pour “lecture” angles
                cv2.line(annotated, tuple(LH.astype(int)), tuple(LK.astype(int)), (0, 255, 255), 2)
                cv2.line(annotated, tuple(LK.astype(int)), tuple(LA.astype(int)), (0, 255, 255), 2)
                cv2.line(annotated, tuple(LA.astype(int)), tuple(LHE.astype(int)), (0, 255, 255), 2)

                cv2.line(annotated, tuple(RH.astype(int)), tuple(RK.astype(int)), (0, 255, 255), 2)
                cv2.line(annotated, tuple(RK.astype(int)), tuple(RA.astype(int)), (0, 255, 255), 2)
                cv2.line(annotated, tuple(RA.astype(int)), tuple(RHE.astype(int)), (0, 255, 255), 2)

                # Texte
                cv2.putText(annotated, f"Vue choisie : {vue}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                cv2.putText(annotated, f"Epaules: {shoulder_lower} plus basse ({diff_shoulders_mm:.1f} mm)",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                cv2.putText(annotated, f"Bassin: {hip_lower} plus bas ({diff_hips_mm:.1f} mm)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # ================= RESULTS =================
                results = {
                    "Nom": nom,
                    "Vue (choisie)": vue,
                    "Inclinaison Épaules (Horizon = 0°)": f"{shoulder_angle:.1f}°",
                    "Épaule la plus basse": shoulder_lower,
                    "Dénivelé Épaules (mm)": f"{diff_shoulders_mm:.1f} mm",
                    "Inclinaison Bassin (Horizon = 0°)": f"{hip_angle:.1f}°",
                    "Bassin le plus bas": hip_lower,
                    "Dénivelé Bassin (mm)": f"{diff_hips_mm:.1f} mm",
                    "Angle Genou Gauche (fémur–tibia)": f"{knee_l:.1f}°",
                    "Angle Genou Droit (fémur–tibia)": f"{knee_r:.1f}°",
                    "Cheville G (tibia–arrière-pied)": f"{ankle_l:.1f}°",
                    "Cheville D (tibia–arrière-pied)": f"{ankle_r:.1f}°",
                }

                with col_result:
                    st.subheader("Résultats de l'analyse")
                    st.image(annotated, use_container_width=True)
                    st.table(results)

                    pdf_path = generate_pdf(results, annotated)
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="📥 Télécharger le Bilan PDF",
                            data=f,
                            file_name=pdf_path,
                            mime="application/pdf",
                            use_container_width=True
                        )

                    # Optionnel : supprimer le PDF après download
                    # os.remove(pdf_path)
