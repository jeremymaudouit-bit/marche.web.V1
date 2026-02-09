import streamlit as st
import cv2, os, tempfile, base64
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from scipy.signal import butter, filtfilt, find_peaks

import mediapipe as mp

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
import streamlit.components.v1 as components


# ============================== #
# CONFIG
# ============================== #
st.set_page_config("GaitScan Pro (1 vidéo – Qualité+)", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique (1 vidéo – Qualité+)")

# FPS: on va lire le FPS réel de la vidéo si possible
DEFAULT_FPS = 30


# ============================== #
# MEDIAPIPE
# ============================== #
mp_pose = mp.solutions.pose

@st.cache_resource
def load_pose():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,          # + robuste que 1 (un peu + lent)
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

pose = load_pose()


# ============================== #
# NORMES (demo)
# ============================== #
def norm_curve(joint, n):
    x = np.linspace(0, 100, n)
    if joint == "Genou":
        return np.interp(x, [0, 15, 40, 60, 80, 100], [5, 15, 5, 40, 60, 5])
    if joint == "Hanche":
        return np.interp(x, [0, 30, 60, 100], [30, 0, -10, 30])
    if joint == "Cheville":
        return np.interp(x, [0, 10, 50, 70, 100], [5, 10, 25, 10, 5])
    return np.zeros(n)

def smooth_ma(y, win=7):
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(ypad, kernel, mode="valid")


# ============================== #
# GEOM / ANGLES
# ============================== #
def angle(a, b, c):
    # angle ABC in image coordinates; we invert y to treat "up" positive
    ba = a - b
    bc = c - b
    ba[1] *= -1
    bc[1] *= -1
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosv = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(np.clip(cosv, -1, 1))))

def angle_hanche(e, h, g):  # hip flexion approx
    return 180.0 - angle(e, h, g)

def angle_genou(h, g, c):   # knee flexion approx
    return 180.0 - angle(h, g, c)

def angle_cheville(g, c, o):  # tibia–foot
    return angle(g, c, o)

def dist2(a, b):
    d = a - b
    return float(np.sqrt(d[0]*d[0] + d[1]*d[1] + 1e-12))


# ============================== #
# SIGNAL PROCESS
# ============================== #
def lowpass(sig, cutoff_hz, fs, order=4):
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 10:
        return sig
    nyq = fs / 2.0
    c = min(max(cutoff_hz, 0.1), nyq * 0.95)
    b, a = butter(order, c / nyq, btype="low")
    return filtfilt(b, a, sig)

def nan_interp(x):
    x = np.asarray(x, dtype=float)
    idx = np.arange(len(x))
    ok = np.isfinite(x)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], x[ok])
    return None

def detect_cycle(y, fs):
    y = np.asarray(y, dtype=float)
    y = nan_interp(y)
    if y is None:
        return None
    inv = -y
    dist = max(1, int(fs * 0.5))
    prom = np.std(inv) * 0.3 if np.std(inv) > 1e-9 else 0.0
    p, _ = find_peaks(inv, distance=dist, prominence=prom)
    return (int(p[0]), int(p[1])) if len(p) >= 2 else None

def asym_percent(left, right):
    if left is None or right is None:
        return None
    denom = (left + right) / 2.0
    if abs(denom) < 1e-6:
        return None
    return 100.0 * abs(right - left) / abs(denom)


# ============================== #
# POSE DETECTION
# ============================== #
def detect_pose(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark
    L = mp_pose.PoseLandmark

    def pt(l):
        p = lm[int(l)]
        return np.array([p.x, p.y], dtype=np.float32), float(p.visibility)

    kp = {}
    for side, suf in [("LEFT", "G"), ("RIGHT", "D")]:
        kp[f"Epaule {suf}"], kp[f"Epaule {suf} vis"] = pt(getattr(L, f"{side}_SHOULDER"))
        kp[f"Hanche {suf}"], kp[f"Hanche {suf} vis"] = pt(getattr(L, f"{side}_HIP"))
        kp[f"Genou {suf}"], kp[f"Genou {suf} vis"] = pt(getattr(L, f"{side}_KNEE"))
        kp[f"Cheville {suf}"], kp[f"Cheville {suf} vis"] = pt(getattr(L, f"{side}_ANKLE"))
        kp[f"Talon {suf}"], kp[f"Talon {suf} vis"] = pt(getattr(L, f"{side}_HEEL"))
        kp[f"Orteil {suf}"], kp[f"Orteil {suf} vis"] = pt(getattr(L, f"{side}_FOOT_INDEX"))
    return kp


# ============================== #
# QUALITY GATING (important)
# ============================== #
def frame_quality_score(kp, conf, side="G"):
    """
    Score in [0..1] based on:
      - visibility for key joints
      - segment-length consistency (thigh/shank) in normalized coords
    Returns: score, seg_lengths dict or None
    """
    if kp is None:
        return 0.0, None

    def ok(name):
        return kp.get(f"{name} {side} vis", 0.0) >= conf

    needed = ["Hanche", "Genou", "Cheville", "Talon", "Orteil"]
    vis_vals = [kp.get(f"{n} {side} vis", 0.0) for n in needed]
    vis_ok = [v for v in vis_vals if v is not None]
    vis_score = float(np.clip(np.mean(vis_ok), 0.0, 1.0)) if vis_ok else 0.0

    if not (ok("Hanche") and ok("Genou") and ok("Cheville")):
        return 0.15 * vis_score, None

    hip = kp[f"Hanche {side}"]
    knee = kp[f"Genou {side}"]
    ank = kp[f"Cheville {side}"]

    thigh = dist2(hip, knee)
    shank = dist2(knee, ank)

    # baseline plausibility: avoid degenerate skeleton
    if thigh < 1e-4 or shank < 1e-4:
        return 0.0, None

    # combine: visibility + geometry
    score = 0.55 * vis_score + 0.45 * 1.0
    return float(np.clip(score, 0.0, 1.0)), {"thigh": thigh, "shank": shank}


def build_outlier_mask(seg_series, z_thresh=3.0):
    """
    seg_series: list of segment length dicts or None
    Returns mask_good (bool array)
    """
    n = len(seg_series)
    thigh = np.array([s["thigh"] if s is not None else np.nan for s in seg_series], dtype=float)
    shank = np.array([s["shank"] if s is not None else np.nan for s in seg_series], dtype=float)

    def robust_good(x):
        ok = np.isfinite(x)
        if ok.sum() < max(10, n // 5):
            return np.ones(n, dtype=bool)
        xv = x[ok]
        med = np.median(xv)
        mad = np.median(np.abs(xv - med)) + 1e-12
        z = np.zeros(n, dtype=float)
        z[ok] = 0.6745 * (x[ok] - med) / mad
        good = np.ones(n, dtype=bool)
        good[ok] = np.abs(z[ok]) <= z_thresh
        return good

    g1 = robust_good(thigh)
    g2 = robust_good(shank)
    return (g1 & g2)


# ============================== #
# VIDEO PROCESS (quality+)
# ============================== #
def process_video_quality(path, conf, side_pref="auto"):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 5 else float(DEFAULT_FPS)

    frames = []
    kp_list = []
    qG, qD = [], []
    segG, segD = [], []

    while cap.isOpened():
        r, f = cap.read()
        if not r:
            break
        frames.append(f.copy())
        kp = detect_pose(f)
        kp_list.append(kp)

        sG, gG = frame_quality_score(kp, conf, side="G")
        sD, gD = frame_quality_score(kp, conf, side="D")
        qG.append(sG); qD.append(sD)
        segG.append(gG); segD.append(gD)

    cap.release()

    if len(frames) == 0:
        raise ValueError("Vidéo vide ou illisible.")

    qG = np.array(qG, dtype=float)
    qD = np.array(qD, dtype=float)

    if side_pref == "auto":
        side = "G" if np.nanmean(qG) >= np.nanmean(qD) else "D"
    else:
        side = "G" if side_pref == "Gauche" else "D"

    # Outlier masks on segment lengths for the chosen side
    seg_series = segG if side == "G" else segD
    mask_geom = build_outlier_mask(seg_series, z_thresh=3.0)

    # Combine with quality threshold
    qual = qG if side == "G" else qD
    mask_qual = qual >= 0.35
    mask_good = mask_geom & mask_qual

    # Prepare angle arrays (both sides for asymmetry)
    out = {k: np.full(len(frames), np.nan, dtype=float)
           for k in ["Hanche G", "Hanche D", "Genou G", "Genou D", "Cheville G", "Cheville D"]}
    heelG = np.full(len(frames), np.nan, dtype=float)
    heelD = np.full(len(frames), np.nan, dtype=float)

    for i, kp in enumerate(kp_list):
        if kp is None:
            continue

        def ok(n, suf):
            return kp.get(f"{n} {suf} vis", 0.0) >= conf

        # angles: compute but later we will reject outliers for chosen side by mask_good
        if ok("Epaule", "G") and ok("Hanche", "G") and ok("Genou", "G"):
            out["Hanche G"][i] = angle_hanche(kp["Epaule G"], kp["Hanche G"], kp["Genou G"])
        if ok("Epaule", "D") and ok("Hanche", "D") and ok("Genou", "D"):
            out["Hanche D"][i] = angle_hanche(kp["Epaule D"], kp["Hanche D"], kp["Genou D"])

        if ok("Hanche", "G") and ok("Genou", "G") and ok("Cheville", "G"):
            out["Genou G"][i] = angle_genou(kp["Hanche G"], kp["Genou G"], kp["Cheville G"])
        if ok("Hanche", "D") and ok("Genou", "D") and ok("Cheville", "D"):
            out["Genou D"][i] = angle_genou(kp["Hanche D"], kp["Genou D"], kp["Cheville D"])

        if ok("Genou", "G") and ok("Cheville", "G") and ok("Orteil", "G"):
            out["Cheville G"][i] = angle_cheville(kp["Genou G"], kp["Cheville G"], kp["Orteil G"])
        if ok("Genou", "D") and ok("Cheville", "D") and ok("Orteil", "D"):
            out["Cheville D"][i] = angle_cheville(kp["Genou D"], kp["Cheville D"], kp["Orteil D"])

        if ok("Talon", "G"):
            heelG[i] = float(kp["Talon G"][1])
        if ok("Talon", "D"):
            heelD[i] = float(kp["Talon D"][1])

    # Reject frames on chosen side (set NaN) to force interpolation later
    if side == "G":
        bad = ~mask_good
        for k in ["Hanche G", "Genou G", "Cheville G"]:
            out[k][bad] = np.nan
    else:
        bad = ~mask_good
        for k in ["Hanche D", "Genou D", "Cheville D"]:
            out[k][bad] = np.nan

    info = {
        "fps": fps,
        "side_used": side,
        "qG_mean": float(np.nanmean(qG)),
        "qD_mean": float(np.nanmean(qD)),
        "good_ratio": float(np.mean(mask_good)),
    }
    return out, heelG, heelD, frames, info, mask_good


# ============================== #
# STEP LENGTH (2D approximate, like your original but gated)
# ============================== #
def compute_step_length_cm(heelG, heelD, taille_cm, fs):
    hG = nan_interp(heelG)
    hD = nan_interp(heelD)
    if hG is None or hD is None:
        return None

    cG = detect_cycle(hG, fs)
    cD = detect_cycle(hD, fs)

    stepG_norm = None
    stepD_norm = None

    if cG:
        i0, i1 = cG
        stepG_norm = abs(hG[i1] - hD[i0])
    if cD:
        i0, i1 = cD
        stepD_norm = abs(hD[i1] - hG[i0])

    steps_norm = [v for v in [stepG_norm, stepD_norm] if v is not None and np.isfinite(v)]
    if not steps_norm:
        return None

    scale = float(taille_cm) / 0.53  # heuristic scale in your original
    stepG_cm = stepG_norm * scale if stepG_norm is not None else None
    stepD_cm = stepD_norm * scale if stepD_norm is not None else None

    valid = [v for v in [stepG_cm, stepD_cm] if v is not None]
    mean_cm = float(np.mean(valid))
    std_cm = float(np.std(valid))
    asym = asym_percent(stepG_cm, stepD_cm)
    return {"mean": mean_cm, "std": std_cm, "G": stepG_cm, "D": stepD_cm, "asym": asym}


# ============================== #
# PDF EXPORT
# ============================== #
def export_pdf(patient, keyframe_path, figures, table_data, step_info=None, asym_table=None):
    out_path = os.path.join(
        tempfile.gettempdir(),
        f"GaitScan_Qualite_{patient['nom']}_{patient['prenom']}.pdf"
    )

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique (1 vidéo – Qualité+)</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Angle de film :</b> {patient.get('camera','N/A')}<br/>"
        f"<b>Côté privilégié :</b> {patient.get('side_used','N/A')}<br/>"
        f"<b>Norme affichée :</b> {'Oui' if patient.get('show_norm', True) else 'Non'}<br/>"
        f"<b>Taille :</b> {patient.get('taille_cm','N/A')} cm",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.3 * cm))

    if step_info is not None:
        story.append(Paragraph("<b>Paramètres spatio-temporels (estimation)</b>", styles["Heading2"]))
        txt = (
            f"<b>Longueur de pas moyenne :</b> {step_info['mean']:.1f} cm<br/>"
            f"<b>Variabilité :</b> ± {step_info['std']:.1f} cm<br/>"
        )
        if step_info.get("G") is not None and step_info.get("D") is not None:
            txt += f"<b>Pas G :</b> {step_info['G']:.1f} cm &nbsp;&nbsp; <b>Pas D :</b> {step_info['D']:.1f} cm<br/>"
        if step_info.get("asym") is not None:
            txt += f"<b>Asymétrie pas (G/D) :</b> {step_info['asym']:.1f} %<br/>"
        txt += "<i>Mesure monocaméra 2D sans calibration métrique : valeurs estimées.</i>"
        story.append(Paragraph(txt, styles["Normal"]))
        story.append(Spacer(1, 0.25 * cm))

    if asym_table:
        story.append(Paragraph("<b>Asymétries droite/gauche (angles)</b>", styles["Heading2"]))
        t = Table([["Mesure", "Moy G", "Moy D", "Asym %"]] + asym_table,
                  colWidths=[6 * cm, 3 * cm, 3 * cm, 3 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("ALIGN", (1, 1), (-1, -1), "CENTER")
        ]))
        story.append(t)
        story.append(Spacer(1, 0.35 * cm))

    if keyframe_path:
        story.append(Paragraph("<b>Image clé</b>", styles["Heading2"]))
        story.append(PDFImage(keyframe_path, width=16 * cm, height=8 * cm))
        story.append(Spacer(1, 0.35 * cm))

    story.append(Paragraph("<b>Analyse articulaire</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * cm))
    for joint, figpath in figures.items():
        story.append(Paragraph(f"<b>{joint}</b>", styles["Heading3"]))
        story.append(PDFImage(figpath, width=16 * cm, height=6 * cm))
        story.append(Spacer(1, 0.3 * cm))

    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("<b>Synthèse (°)</b>", styles["Heading2"]))
    table = Table([["Mesure", "Min", "Moyenne", "Max"]] + table_data,
                  colWidths=[7 * cm, 3 * cm, 3 * cm, 3 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.7, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER")
    ]))
    story.append(table)

    doc.build(story)
    return out_path


# ============================== #
# PDF VIEWER (optional)
# ============================== #
def pdf_viewer_with_print(pdf_bytes: bytes, height=800):
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f"""
    <div style="display:flex; gap:12px; align-items:center; margin: 6px 0 10px 0;">
      <button onclick="printPdf()" style="padding:10px 14px; font-size:16px; cursor:pointer;">
        🖨️ Imprimer le rapport
      </button>
      <span style="opacity:0.7;">(ouvre la boîte d’impression du navigateur)</span>
    </div>
    <iframe id="pdfFrame" src="data:application/pdf;base64,{b64}" width="100%" height="{height}px"
            style="border:1px solid #ddd; border-radius:8px;"></iframe>
    <script>
      function printPdf() {{
        const iframe = document.getElementById('pdfFrame');
        iframe.contentWindow.focus();
        iframe.contentWindow.print();
      }}
    </script>
    """
    components.html(html, height=height + 80, scrolling=True)


# ============================== #
# UI
# ============================== #
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")
    camera_pos = st.selectbox("Angle de film", ["Profil (sagittal)", "Face (frontal)", "Oblique"])

    side_pref = st.selectbox("Côté privilégié", ["auto", "Gauche", "Droite"])
    conf = st.slider("Seuil confiance (visibility)", 0.1, 0.9, 0.35, 0.05)

    # filtrage
    cutoff_hz = st.slider("Filtre low-pass (Hz)", 3.0, 10.0, 6.0, 0.5,
                          help="6 Hz = classique gait. Monte si la marche est très rapide.")
    show_norm = st.checkbox("Afficher la norme", value=True)
    norm_smooth_win = st.slider("Lissage norme", 1, 21, 7, 2)

    taille_cm = st.number_input("Taille du patient (cm)", min_value=80, max_value=230, value=170, step=1)

video = st.file_uploader("Vidéo (mp4/avi/mov)", ["mp4", "avi", "mov"])


# ============================== #
# ANALYSE
# ============================== #
if video and st.button("▶ Lancer l'analyse (Qualité+)"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(video.read())
    tmp.close()

    data, heelG, heelD, frames, info, mask_good = process_video_quality(
        tmp.name, conf=conf, side_pref=side_pref
    )
    os.unlink(tmp.name)

    fps = info["fps"]
    st.subheader("✅ Qualité de tracking")
    st.write(f"- **FPS estimé :** {fps:.1f}")
    st.write(f"- **Côté privilégié :** {'Gauche' if info['side_used']=='G' else 'Droite'}")
    st.write(f"- **Qualité moyenne** (vis+cohérence) : G={info['qG_mean']:.2f} | D={info['qD_mean']:.2f}")
    st.write(f"- **Frames conservées (côté choisi)** : {100*info['good_ratio']:.1f}%")
    if info["good_ratio"] < 0.5:
        st.warning("Beaucoup de frames rejetées (occlusion / cadrage / lumière). Les courbes peuvent rester bruitées.")

    # Step length (2D estimate)
    step_info = compute_step_length_cm(heelG, heelD, float(taille_cm), fs=fps)
    st.subheader("📏 Paramètres spatio-temporels")
    if step_info is not None:
        st.write(f"**Longueur de pas moyenne :** {step_info['mean']:.1f} cm")
        st.write(f"**Variabilité (±1σ) :** {step_info['std']:.1f} cm")
        if step_info.get("G") is not None and step_info.get("D") is not None:
            st.write(f"**Pas G :** {step_info['G']:.1f} cm — **Pas D :** {step_info['D']:.1f} cm")
        if step_info.get("asym") is not None:
            st.write(f"**Asymétrie pas (G/D) :** {step_info['asym']:.1f} %")
        st.caption("Estimation monocaméra 2D sans calibration métrique.")
    else:
        st.warning("Longueur de pas non calculable (talons insuffisamment détectés).")

    # keyframe
    keyframe_path = os.path.join(tempfile.gettempdir(), "keyframe_gaitscan.png")
    cv2.imwrite(keyframe_path, frames[len(frames)//2])

    # Graphs + PDF figures
    figures = {}
    table_data = []
    asym_rows = []

    for joint in ["Hanche", "Genou", "Cheville"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

        g_raw = np.array(data[f"{joint} G"], dtype=float)
        d_raw = np.array(data[f"{joint} D"], dtype=float)

        # Interp then filter
        g_i = nan_interp(g_raw)
        d_i = nan_interp(d_raw)

        if g_i is None:
            g_f = np.full_like(g_raw, np.nan)
        else:
            g_f = lowpass(g_i, cutoff_hz=cutoff_hz, fs=fps, order=4)

        if d_i is None:
            d_f = np.full_like(d_raw, np.nan)
        else:
            d_f = lowpass(d_i, cutoff_hz=cutoff_hz, fs=fps, order=4)

        ax1.plot(g_f, label="Gauche")
        ax1.plot(d_f, label="Droite")
        ax1.set_title(f"{joint} – Courbes filtrées (Qualité+)")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Angle (°)")
        ax1.legend()

        if show_norm:
            norm = norm_curve(joint, len(g_f))
            norm = smooth_ma(norm, win=norm_smooth_win)
            ax2.plot(norm)
            ax2.set_title("Norme (lissée)" if norm_smooth_win and norm_smooth_win > 1 else "Norme")
            ax2.set_xlabel("% cycle (approx)")
            ax2.set_ylabel("Angle (°)")
        else:
            ax2.axis("off")

        st.pyplot(fig)

        fig_path = os.path.join(tempfile.gettempdir(), f"{joint}_plot_quality.png")
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        figures[joint] = fig_path

        def stats(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) == 0:
                return np.nan, np.nan, np.nan, None
            return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr)), float(np.mean(arr))

        gmin, gmean, gmax, gmean_only = stats(g_f)
        dmin, dmean, dmax, dmean_only = stats(d_f)

        table_data.append([f"{joint} Gauche", f"{gmin:.1f}", f"{gmean:.1f}", f"{gmax:.1f}"])
        table_data.append([f"{joint} Droite", f"{dmin:.1f}", f"{dmean:.1f}", f"{dmax:.1f}"])

        a = asym_percent(gmean_only, dmean_only)
        if a is None:
            asym_rows.append([joint,
                              f"{gmean_only:.1f}" if gmean_only is not None else "NA",
                              f"{dmean_only:.1f}" if dmean_only is not None else "NA",
                              "NA"])
        else:
            asym_rows.append([joint, f"{gmean_only:.1f}", f"{dmean_only:.1f}", f"{a:.1f}"])

    st.subheader("↔️ Asymétries droite/gauche (angles)")
    for row in asym_rows:
        st.write(f"**{row[0]}** — Moy G: {row[1]}° | Moy D: {row[2]}° | Asym: {row[3]}%")

    # PDF
    pdf_path = export_pdf(
        patient={
            "nom": nom,
            "prenom": prenom,
            "camera": camera_pos,
            "taille_cm": int(taille_cm),
            "show_norm": bool(show_norm),
            "side_used": "Gauche" if info["side_used"] == "G" else "Droite",
        },
        keyframe_path=keyframe_path,
        figures=figures,
        table_data=table_data,
        step_info=step_info,
        asym_table=asym_rows
    )

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    st.success("✅ Rapport généré")
    st.download_button(
        "📄 Télécharger le rapport PDF",
        data=pdf_bytes,
        file_name=f"GaitScan_Qualite_{nom}_{prenom}.pdf",
        mime="application/pdf"
    )

    # Option preview + print
    with st.expander("👁️ Aperçu du PDF + impression"):
        pdf_viewer_with_print(pdf_bytes, height=800)
