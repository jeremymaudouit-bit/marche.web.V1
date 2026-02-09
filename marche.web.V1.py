# ============================== #
# GAITSCAN PRO – OPENCAP EDITION #
# ============================== #
# Streamlit app that reads OpenCap exports (.mot for joint angles, optional .trc for markers)
# and generates plots, asymmetry metrics, step-length estimate, and a PDF report.

import os
import re
import io
import zipfile
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, find_peaks

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as PDFImage,
    Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors

# ============================== #
# CONFIG
# ============================== #
st.set_page_config("GaitScan Pro (OpenCap)", layout="wide")
st.title("🏃 GaitScan Pro – Analyse Cinématique (OpenCap)")

# ============================== #
# NORMES (demo curves – adjust to your reference if needed)
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
    """Simple moving-average smoothing (odd window recommended)."""
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
# SIGNAL HELPERS
# ============================== #
def bandpass(sig, lvl, fs):
    """
    Gentle bandpass similar to your original: low rises with lvl, high drops with lvl.
    lvl in [0..10] typical.
    """
    sig = np.asarray(sig, dtype=float)
    if len(sig) < 10:
        return sig

    low = 0.3 + float(lvl) * 0.02
    high = max(6.0 - float(lvl) * 0.25, low + 0.4)

    nyq = fs / 2.0
    if high >= nyq * 0.95:
        high = nyq * 0.95
    if low <= 0:
        low = 0.05

    if high <= low:
        return sig

    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig)

def nan_interp(x):
    x = np.array(x, dtype=float)
    idx = np.arange(len(x))
    ok = ~np.isnan(x)
    if ok.sum() >= 2:
        return np.interp(idx, idx[ok], x[ok])
    return None

def detect_cycle(y, fs):
    """
    Detect two heel-strikes via minima on vertical heel trajectory:
    peaks on inverted signal.
    """
    y = np.array(y, dtype=float)
    if np.isnan(y).any():
        idx = np.arange(len(y))
        ok = ~np.isnan(y)
        if ok.sum() >= 2:
            y = np.interp(idx, idx[ok], y[ok])
        else:
            return None

    inv = -y
    # distance about half a second
    dist = max(1, int(fs * 0.5))
    prom = np.std(inv) * 0.3 if np.std(inv) > 1e-9 else 0.0
    p, _ = find_peaks(inv, distance=dist, prominence=prom)
    return (int(p[0]), int(p[1])) if len(p) >= 2 else None

def asym_percent(left, right):
    # Asym (%) = 100 * |R - L| / ((R + L)/2)
    if left is None or right is None:
        return None
    denom = (left + right) / 2.0
    if abs(denom) < 1e-9:
        return None
    return 100.0 * abs(right - left) / abs(denom)

# ============================== #
# OPENSIM FILE READERS (.mot/.sto/.trc)
# ============================== #
def read_opensim_table(path: str) -> pd.DataFrame:
    """
    Read OpenSim .mot/.sto into a DataFrame. Handles 'endheader' section.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    end_idx = None
    for i, line in enumerate(lines):
        if "endheader" in line.lower():
            end_idx = i
            break

    if end_idx is None:
        # fallback: find first line containing 'time' as a column header
        for i, line in enumerate(lines[:80]):
            toks = line.strip().split()
            if len(toks) >= 2 and any(t.lower() == "time" for t in toks):
                end_idx = i - 1
                break

    data_str = "".join(lines[(end_idx + 1) if end_idx is not None else 0 :])
    df = pd.read_csv(io.StringIO(data_str), sep=r"\s+|\t+|,", engine="python")
    if "Time" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"Time": "time"})
    return df

def read_trc(path: str):
    """
    Read OpenSim TRC marker file. Returns:
      df columns: frame, time, <marker>_X, <marker>_Y, <marker>_Z
      marker_names list
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_line_idx = None
    for i, line in enumerate(lines[:30]):
        low = line.lower()
        if ("frame" in low or "frame#" in low) and "time" in low:
            header_line_idx = i
            break
    if header_line_idx is None:
        raise ValueError("TRC: impossible de trouver la ligne 'Frame# Time ...'")

    marker_line = lines[header_line_idx].strip().split()
    # Next line describes X/Y/Z indices, ignore content
    data_start = header_line_idx + 2

    raw = pd.read_csv(
        io.StringIO("".join(lines[data_start:])),
        sep=r"\s+|\t+|,", engine="python", header=None
    )

    ncols = raw.shape[1]
    cols = ["frame", "time"]

    n_markers = (ncols - 2) // 3
    base = marker_line[2:]
    if len(base) >= 3 * n_markers:
        base_names = [base[i * 3] for i in range(n_markers)]
    else:
        base_names = [f"M{i+1}" for i in range(n_markers)]

    for m in base_names:
        cols += [f"{m}_X", f"{m}_Y", f"{m}_Z"]

    raw.columns = cols[:ncols]
    return raw, base_names

def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def infer_sampling_rate(time_vec):
    time_vec = np.asarray(time_vec, dtype=float)
    if len(time_vec) < 3:
        return 30.0
    dt = np.diff(time_vec)
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0:
        return 30.0
    med = float(np.median(dt))
    if med <= 0:
        return 30.0
    return 1.0 / med

def find_marker_name(markers, substrings):
    for m in markers:
        low = m.lower()
        if any(s in low for s in substrings):
            return m
    return None

def infer_vertical_axis(trc_df, heel_marker):
    """
    Try to infer which axis is vertical by picking the one with most periodic variation.
    Falls back to Y.
    """
    axes = ["X", "Y", "Z"]
    best_axis = "Y"
    best_score = -1.0
    for ax in axes:
        col = f"{heel_marker}_{ax}"
        if col not in trc_df.columns:
            continue
        v = trc_df[col].to_numpy(dtype=float)
        v = nan_interp(v)
        if v is None:
            continue
        # score: std * (some peaks)
        score = float(np.std(v))
        if score > best_score:
            best_score = score
            best_axis = ax
    return best_axis

# ============================== #
# OPENCAP PROCESSING
# ============================== #
def process_opencap(mot_path: str, trc_path: str | None):
    """
    Returns:
      angles_dict: keys like your original res dict (Hanche/Genou/Cheville G/D)
      heelG_vert, heelD_vert: vertical trajectories if trc provided (else None)
      fs: sampling frequency from .mot time
      time vector
      trc_df (optional), marker_names (optional)
    """
    kin = read_opensim_table(mot_path)
    if "time" not in kin.columns:
        raise ValueError("Le fichier .mot/.sto ne contient pas de colonne 'time'.")

    t = kin["time"].to_numpy(dtype=float)
    fs = infer_sampling_rate(t)

    # Column mappings: adapt if your model uses different names
    col_hL = pick_col(kin, ["hip_flexion_l", "hip_flex_l", "hip_flexion_left"])
    col_hR = pick_col(kin, ["hip_flexion_r", "hip_flex_r", "hip_flexion_right"])
    col_kL = pick_col(kin, ["knee_angle_l", "knee_flexion_l", "knee_flex_l"])
    col_kR = pick_col(kin, ["knee_angle_r", "knee_flexion_r", "knee_flex_r"])
    col_aL = pick_col(kin, ["ankle_angle_l", "ankle_flexion_l", "ankle_flex_l"])
    col_aR = pick_col(kin, ["ankle_angle_r", "ankle_flexion_r", "ankle_flex_r"])

    missing = [name for name, col in [
        ("Hanche L", col_hL), ("Hanche R", col_hR),
        ("Genou L", col_kL), ("Genou R", col_kR),
        ("Cheville L", col_aL), ("Cheville R", col_aR),
    ] if col is None]

    if missing:
        preview_cols = list(kin.columns)[:60]
        raise ValueError(
            f"Colonnes cinématiques manquantes dans le .mot : {missing}\n"
            f"Colonnes dispo (aperçu): {preview_cols}"
        )

    angles = {
        "Hanche G": kin[col_hL].to_numpy(dtype=float),
        "Hanche D": kin[col_hR].to_numpy(dtype=float),
        "Genou G": kin[col_kL].to_numpy(dtype=float),
        "Genou D": kin[col_kR].to_numpy(dtype=float),
        "Cheville G": kin[col_aL].to_numpy(dtype=float),
        "Cheville D": kin[col_aR].to_numpy(dtype=float),
    }

    heelG = heelD = None
    trc_df = None
    markers = None
    vertical_axis = "Y"

    if trc_path:
        trc_df, markers = read_trc(trc_path)

        # Try common naming
        mL = find_marker_name(markers, ["lheel", "heel_l", "left_heel", "lhe", "calc_l", "calcn_l"])
        mR = find_marker_name(markers, ["rheel", "heel_r", "right_heel", "rhe", "calc_r", "calcn_r"])

        # If not found, allow user to pick later by returning df+markers
        if mL:
            vertical_axis = infer_vertical_axis(trc_df, mL)
            col = f"{mL}_{vertical_axis}"
            if col in trc_df.columns:
                heelG = trc_df[col].to_numpy(dtype=float)
        if mR:
            # keep same axis for both
            col = f"{mR}_{vertical_axis}"
            if col in trc_df.columns:
                heelD = trc_df[col].to_numpy(dtype=float)

    return angles, heelG, heelD, fs, t, trc_df, markers

# ============================== #
# STEP LENGTH (3D if possible)
# ============================== #
def compute_step_length_cm_from_trc(trc_df, markers, fs, prefer_axis="Y"):
    """
    Estimate step length using heel marker trajectory in 3D.
    Approach:
      - detect two heel-strikes for each side using vertical axis minima (peaks on inverted)
      - compute forward displacement between successive heel-strikes for the SAME heel (step-like stride proxy)
    Notes:
      - assumes marker coordinates are in meters (typical OpenSim). Converts to cm.
      - forward axis is inferred as the axis with the largest range among X/Z (excluding vertical).
    """
    if trc_df is None or markers is None:
        return None

    mL = find_marker_name(markers, ["lheel", "heel_l", "left_heel", "lhe", "calc_l", "calcn_l"])
    mR = find_marker_name(markers, ["rheel", "heel_r", "right_heel", "rhe", "calc_r", "calcn_r"])
    if not (mL and mR):
        return None

    # Determine vertical axis
    vert = prefer_axis
    if f"{mL}_{vert}" not in trc_df.columns:
        vert = infer_vertical_axis(trc_df, mL)

    # Determine forward axis as the non-vertical axis with greatest range (use X/Z preference)
    axes = ["X", "Y", "Z"]
    non_vert = [a for a in axes if a != vert]
    # Use data ranges on left heel
    ranges = {}
    for a in non_vert:
        col = f"{mL}_{a}"
        if col in trc_df.columns:
            v = trc_df[col].to_numpy(dtype=float)
            v = nan_interp(v)
            if v is not None:
                ranges[a] = float(np.nanmax(v) - np.nanmin(v))
    if not ranges:
        return None
    fwd = max(ranges, key=ranges.get)

    # signals
    Lvert = trc_df[f"{mL}_{vert}"].to_numpy(dtype=float)
    Rvert = trc_df[f"{mR}_{vert}"].to_numpy(dtype=float)

    Lfwd = trc_df[f"{mL}_{fwd}"].to_numpy(dtype=float)
    Rfwd = trc_df[f"{mR}_{fwd}"].to_numpy(dtype=float)

    # cycles
    cL = detect_cycle(Lvert, fs)
    cR = detect_cycle(Rvert, fs)

    # step proxy: forward displacement between two heel-strikes for same foot (stride-like),
    # but we report it as "longueur de pas estimée" to stay consistent with your UI.
    stepL = stepR = None
    if cL:
        i0, i1 = cL
        stepL = abs(Lfwd[i1] - Lfwd[i0])
    if cR:
        i0, i1 = cR
        stepR = abs(Rfwd[i1] - Rfwd[i0])

    vals = [v for v in [stepL, stepR] if v is not None and np.isfinite(v)]
    if not vals:
        return None

    # meters -> cm (common for OpenSim). If your TRC is in mm, adjust here.
    stepL_cm = stepL * 100.0 if stepL is not None else None
    stepR_cm = stepR * 100.0 if stepR is not None else None

    valid = [v for v in [stepL_cm, stepR_cm] if v is not None]
    mean_cm = float(np.mean(valid))
    std_cm = float(np.std(valid))
    asym = asym_percent(stepL_cm, stepR_cm)

    return {
        "mean": mean_cm,
        "std": std_cm,
        "G": stepL_cm,  # keep naming consistent (G=Left)
        "D": stepR_cm,
        "asym": asym,
        "meta": {"vertical_axis": vert, "forward_axis": fwd, "marker_L": mL, "marker_R": mR}
    }

# ============================== #
# PDF EXPORT
# ============================== #
def export_pdf(patient, figures, table_data, step_info=None, asym_table=None):
    out_path = os.path.join(
        tempfile.gettempdir(),
        f"GaitScan_OpenCap_{patient['nom']}_{patient['prenom']}.pdf"
    )

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=1.7 * cm, rightMargin=1.7 * cm,
        topMargin=1.7 * cm, bottomMargin=1.7 * cm
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>GaitScan Pro – Analyse Cinématique (OpenCap)</b>", styles["Title"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph(
        f"<b>Patient :</b> {patient['nom']} {patient['prenom']}<br/>"
        f"<b>Date :</b> {datetime.now().strftime('%d/%m/%Y')}<br/>"
        f"<b>Angle de film :</b> {patient.get('camera','N/A')}<br/>"
        f"<b>Affichage phases :</b> {patient.get('phase','N/A')}<br/>"
        f"<b>Norme affichée :</b> {'Oui' if patient.get('show_norm', True) else 'Non'}<br/>"
        f"<b>Source :</b> OpenCap (.mot/.trc)",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.35 * cm))

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
        if step_info.get("meta"):
            meta = step_info["meta"]
            txt += f"<i>Markers: {meta.get('marker_L','?')}/{meta.get('marker_R','?')} – axes: vert={meta.get('vertical_axis','?')}, fwd={meta.get('forward_axis','?')}</i><br/>"
        txt += "<i>Estimation basée sur les marqueurs OpenCap/OpenSim.</i>"
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
# ZIP HELPERS (optional convenience)
# ============================== #
def extract_zip_to_temp(zip_bytes: bytes) -> str:
    out_dir = tempfile.mkdtemp(prefix="opencap_")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(out_dir)
    return out_dir

def find_files(root_dir: str, exts):
    exts = tuple(e.lower() for e in exts)
    found = []
    for r, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(exts):
                found.append(os.path.join(r, f))
    return sorted(found)

# ============================== #
# UI
# ============================== #
with st.sidebar:
    nom = st.text_input("Nom", "DURAND")
    prenom = st.text_input("Prénom", "Jean")

    camera_pos = st.selectbox("Angle de film", ["Devant", "Droite", "Gauche"])
    phase_cote = st.selectbox("Phases (si .trc dispo)", ["Aucune", "Droite", "Gauche", "Les deux"])

    smooth = st.slider("Filtrage (0=off → 10=fort)", 0, 10, 3)
    show_norm = st.checkbox("Afficher la norme", value=True)
    norm_smooth_win = st.slider(
        "Lissage norme (simple)", 1, 21, 7, 2,
        help="Moyenne glissante (impair conseillé). 1 = pas de lissage."
    )

st.subheader("📥 Import OpenCap")
colA, colB = st.columns(2)
with colA:
    mot_file = st.file_uploader("Cinématique OpenSim (.mot ou .sto)", ["mot", "sto", "txt"])
with colB:
    trc_file = st.file_uploader("Marqueurs (.trc) (optionnel)", ["trc"])

st.caption(
    "Astuce : si tu as un export OpenCap en .zip, tu peux aussi l’importer ci-dessous et choisir les fichiers."
)

zip_file = st.file_uploader("OU export OpenCap (.zip) (optionnel)", ["zip"])

mot_path = None
trc_path = None

# Allow ZIP selection
if zip_file is not None:
    try:
        tmpdir = extract_zip_to_temp(zip_file.read())
        mot_candidates = find_files(tmpdir, [".mot", ".sto", ".txt"])
        trc_candidates = find_files(tmpdir, [".trc"])

        st.info(f"ZIP extrait: {os.path.basename(tmpdir)} — {len(mot_candidates)} .mot/.sto, {len(trc_candidates)} .trc trouvés.")

        if mot_candidates:
            chosen_mot = st.selectbox("Choisir le fichier cinématique (.mot/.sto)", mot_candidates)
            mot_path = chosen_mot
        if trc_candidates:
            chosen_trc = st.selectbox("Choisir le fichier marqueurs (.trc) (optionnel)", ["(Aucun)"] + trc_candidates)
            trc_path = None if chosen_trc == "(Aucun)" else chosen_trc
    except Exception as e:
        st.error(f"Erreur ZIP: {e}")

# If direct uploads provided, use them (override ZIP choice if both)
if mot_file is not None:
    mot_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{mot_file.name.split('.')[-1]}")
    mot_tmp.write(mot_file.read())
    mot_tmp.close()
    mot_path = mot_tmp.name

if trc_file is not None:
    trc_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".trc")
    trc_tmp.write(trc_file.read())
    trc_tmp.close()
    trc_path = trc_tmp.name

# ============================== #
# ANALYSE
# ============================== #
if mot_path and st.button("▶ Lancer l'analyse (OpenCap)"):
    try:
        data, heelG, heelD, fs, time_vec, trc_df, markers = process_opencap(mot_path, trc_path)

        st.success(f"✅ Données chargées. Fréquence estimée: {fs:.1f} Hz")

        # Phase shading based on heel vertical minima if available
        phases = []
        if trc_df is not None and heelG is not None and heelD is not None:
            if phase_cote in ["Gauche", "Les deux"]:
                c = detect_cycle(heelG, fs)
                if c:
                    phases.append((*c, "orange"))
            if phase_cote in ["Droite", "Les deux"]:
                c = detect_cycle(heelD, fs)
                if c:
                    phases.append((*c, "blue"))

        # ============================== #
        # STEP LENGTH (3D markers)
        # ============================== #
        step_info = None
        if trc_df is not None and markers is not None:
            step_info = compute_step_length_cm_from_trc(trc_df, markers, fs)
        st.subheader("📏 Paramètres spatio-temporels")
        if step_info is not None:
            st.write(f"**Longueur de pas moyenne :** {step_info['mean']:.1f} cm")
            st.write(f"**Variabilité (±1σ) :** {step_info['std']:.1f} cm")
            if step_info.get("G") is not None and step_info.get("D") is not None:
                st.write(f"**Pas G :** {step_info['G']:.1f} cm — **Pas D :** {step_info['D']:.1f} cm")
            if step_info.get("asym") is not None:
                st.write(f"**Asymétrie pas (G/D) :** {step_info['asym']:.1f} %")
            if step_info.get("meta"):
                st.caption(
                    f"Markers: {step_info['meta'].get('marker_L')} / {step_info['meta'].get('marker_R')} — "
                    f"axes: vert={step_info['meta'].get('vertical_axis')}, fwd={step_info['meta'].get('forward_axis')}"
                )
        else:
            st.warning("Longueur de pas non calculable (pas de .trc ou talons non trouvés).")

        # ============================== #
        # GRAPHS + PDF FIGURES
        # ============================== #
        figures = {}
        table_data = []
        asym_rows = []

        for joint in ["Hanche", "Genou", "Cheville"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})

            g_raw = np.array(data[f"{joint} G"], dtype=float)
            d_raw = np.array(data[f"{joint} D"], dtype=float)

            # Filter (no NaNs expected from .mot, but just in case)
            g_f = bandpass(np.nan_to_num(g_raw, nan=np.nanmean(g_raw) if np.isfinite(np.nanmean(g_raw)) else 0.0), smooth, fs)
            d_f = bandpass(np.nan_to_num(d_raw, nan=np.nanmean(d_raw) if np.isfinite(np.nanmean(d_raw)) else 0.0), smooth, fs)

            ax1.plot(g_f, label="Gauche")
            ax1.plot(d_f, label="Droite")
            for c0, c1, col in phases:
                ax1.axvspan(c0, c1, color=col, alpha=0.25)
            ax1.set_title(f"{joint} – Analyse (OpenCap)")
            ax1.set_xlabel("Index échantillon")
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

            fig_path = os.path.join(tempfile.gettempdir(), f"OpenCap_{joint}_plot.png")
            fig.savefig(fig_path, bbox_inches="tight")
            plt.close(fig)
            figures[joint] = fig_path

            # Stats
            def stats(arr):
                arr = np.asarray(arr, dtype=float)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    return np.nan, np.nan, np.nan
                return float(np.min(arr)), float(np.mean(arr)), float(np.max(arr))

            gmin, gmean, gmax = stats(g_f)
            dmin, dmean, dmax = stats(d_f)

            table_data.append([f"{joint} Gauche", f"{gmin:.1f}", f"{gmean:.1f}", f"{gmax:.1f}"])
            table_data.append([f"{joint} Droite", f"{dmin:.1f}", f"{dmean:.1f}", f"{dmax:.1f}"])

            a = asym_percent(gmean, dmean)
            if a is None or not np.isfinite(a):
                asym_rows.append([joint, f"{gmean:.1f}", f"{dmean:.1f}", "NA"])
            else:
                asym_rows.append([joint, f"{gmean:.1f}", f"{dmean:.1f}", f"{a:.1f}"])

        st.subheader("↔️ Asymétries droite/gauche (angles)")
        for row in asym_rows:
            st.write(f"**{row[0]}** — Moy G: {row[1]}° | Moy D: {row[2]}° | Asym: {row[3]}%")

        # ============================== #
        # PDF
        # ============================== #
        pdf_path = export_pdf(
            patient={
                "nom": nom,
                "prenom": prenom,
                "camera": camera_pos,
                "phase": phase_cote,
                "show_norm": bool(show_norm),
            },
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
            file_name=f"GaitScan_OpenCap_{nom}_{prenom}.pdf",
            mime="application/pdf"
        )

    except Exception as e:
        st.error(f"Erreur pendant l'analyse: {e}")
