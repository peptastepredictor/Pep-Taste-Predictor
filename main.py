# ==========================================================
# PepTastePredictor — app.py
# A complete end-to-end peptide analysis platform
# Light + Dark Mode Compatible | External Folding Workflow | Dynamic UI
# Cloud-compatible: no ColabFold / ESMFold / JAX / GPU required
# ==========================================================

# ==========================================================
# SECTION 1 - IMPORTS
# ==========================================================

import os
import re
import json
import time
import hashlib
import tempfile
import traceback
from datetime import date
from collections import Counter
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder, DSSP
from Bio.PDB.SASA import ShrakeRupley

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as RLImage,
    Spacer,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ==========================================================
# SECTION 2 - GLOBAL CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH     = "AIML.xlsx"
PREDICTIONS_DIR  = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

PLDDT_THRESHOLDS = {
    "Very High": (90, 100),
    "High":      (70, 90),
    "Medium":    (50, 70),
    "Low":       (0,  50),
}

TASTE_EMOJI = {
    "Bitter": "😖", "Sweet": "😋", "Salty": "🧂",
    "Sour": "😮‍💨", "Umami": "🤤",
}


# ==========================================================
# SECTION 3 - FRONTEND STYLING
# ==========================================================

st.markdown("""
<style>
.stApp p, .stApp span, .stApp label,
.stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp div { color: var(--text-color) !important; }
.stMarkdown, .stMarkdown * { color: var(--text-color) !important; }
h1, h2, h3, h4 { color: var(--text-color) !important; }
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p { color: var(--text-color) !important; }
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p { color: var(--text-color) !important; }
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input::placeholder { color: var(--text-color) !important; }
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p { color: var(--text-color) !important; }
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label p { color: var(--text-color) !important; }
details summary p, details summary span,
button[data-testid="stExpanderToggleButton"] p,
button[data-testid="stExpanderToggleButton"] span,
button[data-testid="stExpanderToggleButton"] div { color: var(--text-color) !important; }
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text-color) !important; }
.stDataFrame, .stDataFrame td, .stDataFrame th,
[data-testid="stDataFrame"] * { color: var(--text-color) !important; }
[data-testid="stMetric"] label,
[data-testid="stMetric"] div { color: var(--text-color) !important; }
.stButton button p,
div[data-testid="stDownloadButton"] button p { color: inherit !important; }
div[data-testid="stAlert"] *,
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span { color: inherit !important; }

.hero {
    background: linear-gradient(135deg, #1f3c88 0%, #0b7285 60%, #12b886 100%);
    padding: 40px 44px;
    border-radius: 20px;
    margin-bottom: 36px;
    box-shadow: 0 8px 32px rgba(31,60,136,0.18);
}
.hero h1 {
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    margin-bottom: 10px;
    color: #ffffff !important;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.08rem !important;
    line-height: 1.8;
    color: #dce8ff !important;
    margin: 0;
}

.card {
    border: 1px solid rgba(128,128,180,0.3);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 28px;
    background: rgba(128,128,180,0.05);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

.live-card {
    border: 2px solid rgba(26,143,209,0.35);
    padding: 22px 28px;
    border-radius: 14px;
    margin-bottom: 18px;
    background: rgba(26,143,209,0.05);
    transition: all 0.3s ease;
}

.ext-card {
    border: 2px solid rgba(18,184,134,0.4);
    padding: 26px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    background: rgba(18,184,134,0.05);
}

.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 14px;
}
.step-num {
    min-width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #1a8fd1;
    color: #fff !important;
    font-weight: 800;
    font-size: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.step-text {
    font-size: 15px;
    line-height: 1.6;
    color: var(--text-color) !important;
    padding-top: 4px;
}

.metric-label {
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.6;
    margin-bottom: 4px;
    margin-top: 18px;
    color: var(--text-color) !important;
}
.metric-label:first-child { margin-top: 0; }

.metric-value {
    font-size: 24px !important;
    font-weight: 800 !important;
    color: #1a8fd1 !important;
    margin-bottom: 2px;
}

.progress-bar-wrap {
    background: rgba(128,128,180,0.15);
    border-radius: 8px;
    height: 10px;
    margin: 6px 0 16px 0;
    overflow: hidden;
}
.progress-bar-fill {
    height: 10px;
    border-radius: 8px;
    transition: width 0.6s ease;
}

.seq-counter {
    font-size: 13px;
    font-weight: 600;
    opacity: 0.65;
    margin-bottom: 6px;
    color: var(--text-color) !important;
}

.aa-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin: 3px 2px;
    background: rgba(26,143,209,0.15);
    color: #1a8fd1 !important;
    border: 1px solid rgba(26,143,209,0.3);
}

.graph-caption {
    border-left: 5px solid #4a6fa5;
    border-radius: 0 10px 10px 0;
    padding: 18px 24px;
    margin-top: 12px;
    margin-bottom: 40px;
    font-size: 15px !important;
    line-height: 1.9;
    background: rgba(74,111,165,0.08);
    color: var(--text-color) !important;
}
.graph-caption strong { font-weight: 700; color: var(--text-color) !important; }
.graph-caption em { font-style: italic; color: var(--text-color) !important; opacity: 0.85; }

.section-gap { margin-top: 40px; margin-bottom: 6px; }

.footer {
    text-align: center;
    font-size: 14px !important;
    padding: 44px 20px 20px;
    margin-top: 60px;
    line-height: 2.2;
    border-top: 1px solid rgba(128,128,180,0.25);
    color: var(--text-color) !important;
    opacity: 0.7;
}

.conf-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    margin-top: 4px;
}

@keyframes pulse {
    0%   { opacity: 1; }
    50%  { opacity: 0.5; }
    100% { opacity: 1; }
}
.live-indicator {
    display: inline-block;
    width: 8px; height: 8px;
    background: #12b886;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 4 - SIDEBAR
# ==========================================================

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)

st.sidebar.markdown("### 🧬 PepTastePredictor")
st.sidebar.write("AI-driven peptide analysis platform")
st.sidebar.write("• Taste prediction")
st.sidebar.write("• Solubility prediction")
st.sidebar.write("• Docking estimation")
st.sidebar.write("• Structural bioinformatics")
st.sidebar.write("• Batch screening")

st.sidebar.markdown("---")
st.sidebar.markdown("**🌐 Structure Prediction**")
st.sidebar.markdown(
    '<div style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;'
    'border-radius:20px;font-size:13px;font-weight:600;background:rgba(18,184,134,0.12);'
    'border:1px solid rgba(18,184,134,0.35);color:#12b886;margin-bottom:6px;">🌐 ESM Atlas</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;'
    'border-radius:20px;font-size:13px;font-weight:600;background:rgba(26,143,209,0.12);'
    'border:1px solid rgba(26,143,209,0.35);color:#1a8fd1;">🔬 AlphaFold Server</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Generate structures externally, then upload the PDB for full analysis.")
st.sidebar.info("For academic & educational use only")


# ==========================================================
# SECTION 5 - SESSION STATE
# ==========================================================

defaults = {
    "initialized":    True,
    "pdb_text":       None,
    "last_prediction":{},
    "show_analytics": False,
    "pdf_figures":    [],
    "live_seq":       "",
    "live_results":   None,
    "current_mode":   None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# SECTION 6 - UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename):
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    if filename not in st.session_state.pdf_figures:
        st.session_state.pdf_figures.append(filename)


def clean_sequence(seq):
    """
    Strip whitespace/newlines/FASTA headers, uppercase, keep only valid AA letters.
    Never truncates — processes the entire input.
    """
    if not isinstance(seq, str):
        return ""
    lines = seq.splitlines()
    lines = [l for l in lines if not l.strip().startswith(">")]
    seq = "".join(lines)
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


def validate_sequence(seq: str) -> tuple:
    """Return (is_valid, message)."""
    if not seq:
        return False, "Sequence is empty."
    invalid = [c for c in seq.upper() if c not in AA]
    if invalid:
        return False, f"Invalid characters detected: {set(invalid)}"
    return True, ""


def sequence_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()[:16]


def model_features(seq):
    L = len(seq)
    features = {"length": L}

    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            features.update({
                "mw":          ana.molecular_weight(),
                "pI":          ana.isoelectric_point(),
                "aromaticity": ana.aromaticity(),
                "instability": ana.instability_index(),
                "gravy":       ana.gravy(),
                "charge":      ana.charge_at_pH(7.0),
            })
        except Exception:
            features.update({"mw": 0, "pI": 7.0, "aromaticity": 0,
                              "instability": 0, "gravy": 0, "charge": 0})
    else:
        features.update({
            "mw":          111.0,
            "pI":          7.0,
            "aromaticity": 1.0 if seq in "FWY" else 0.0,
            "instability": 0.0,
            "gravy":       KD_SCALE.get(seq, 0.0),
            "charge":      1.0 if seq in "KRH" else (-1.0 if seq in "DE" else 0.0),
        })

    for aa in AA:
        features[f"AA_{aa}"] = seq.count(aa) / L
    denom = max(L - 1, 1)
    for dp in ALL_DIPEPTIDES:
        features[f"DPC_{dp}"] = seq.count(dp) / denom
    groups = {
        "hydrophobic": "AILMFWV",
        "polar":       "STNQ",
        "charged":     "DEKRH",
        "aromatic":    "FWY",
        "tiny":        "AGSC",
    }
    for name, aas in groups.items():
        features[name] = sum(seq.count(a) for a in aas) / L
    return features


def build_feature_table(seqs):
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)


def physicochemical_features(seq):
    L = len(seq)
    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            h, t, s = ana.secondary_structure_fraction()
            return {
                "Length":                L,
                "Molecular weight (Da)": round(ana.molecular_weight(), 2),
                "Isoelectric point":     round(ana.isoelectric_point(), 2),
                "Net charge (pH 7)":     round(ana.charge_at_pH(7.0), 2),
                "Aromaticity":           round(ana.aromaticity(), 3),
                "GRAVY":                 round(ana.gravy(), 3),
                "Instability index":     round(ana.instability_index(), 2),
                "Helix fraction":        round(h, 3),
                "Turn fraction":         round(t, 3),
                "Sheet fraction":        round(s, 3),
            }
        except Exception:
            pass
    return {
        "Length":      L,
        "GRAVY":       round(KD_SCALE.get(seq, 0.0), 3),
        "Aromaticity": 1.0 if seq in "FWY" else 0.0,
        "Note":        "Extended analysis requires ≥2 amino acids",
    }


def composition_features(seq):
    c = Counter(seq)
    L = len(seq)
    return {
        "Hydrophobic (%)": round(100 * sum(c[a] for a in "AILMFWV") / L, 1),
        "Polar (%)":       round(100 * sum(c[a] for a in "STNQ") / L, 1),
        "Charged (%)":     round(100 * sum(c[a] for a in "DEKRH") / L, 1),
        "Aromatic (%)":    round(100 * sum(c[a] for a in "FWY") / L, 1),
    }


def simplify_taste(taste_series):
    counts = taste_series.value_counts()
    rare = set(counts[counts < 5].index)
    def _map(t):
        if t in rare:
            for base in ["Bitter", "Sweet", "Salty", "Sour", "Umami"]:
                if base.lower() in t.lower():
                    return base
            return "Bitter"
        return t
    return taste_series.apply(_map)


def prettify_feature(name):
    if name.startswith("DPC_"):
        return f"Dipeptide {name[4:]}"
    if name.startswith("AA_"):
        return f"Amino acid: {name[3:]}"
    return name.replace("_", " ").title()


def gravy_score(seq):
    if not seq:
        return 0.0
    return sum(KD_SCALE.get(a, 0) for a in seq) / len(seq)


def show_caption(html_text: str):
    st.markdown(f'<div class="graph-caption">{html_text}</div>', unsafe_allow_html=True)


def taste_emoji(taste: str) -> str:
    for k, v in TASTE_EMOJI.items():
        if k.lower() in taste.lower():
            return v
    return "🧬"


# ==========================================================
# SECTION 6B - DYNAMIC LIVE PREVIEW
# ==========================================================

def render_live_preview(seq: str):
    if not seq:
        return
    L = len(seq)
    gv = gravy_score(seq)
    c  = Counter(seq)

    hydro_pct  = 100 * sum(c[a] for a in "AILMFWV") / L
    charge_pct = 100 * sum(c[a] for a in "DEKRH") / L
    arom_pct   = 100 * sum(c[a] for a in "FWY") / L

    hydro_color  = "#1a8fd1" if gv > 0 else "#e67e22"
    charge_label = "Positive" if sum(c[a] for a in "KRH") > sum(c[a] for a in "DE") else "Negative"

    st.markdown(f"""
    <div class="live-card">
      <span style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;letter-spacing:0.08em;">
        <span class="live-indicator"></span>Live Sequence Analysis
      </span>
      <div style="display:flex;gap:32px;flex-wrap:wrap;margin-top:14px;">
        <div>
          <div class="metric-label" style="margin-top:0;">Length</div>
          <div class="metric-value">{L} aa</div>
        </div>
        <div>
          <div class="metric-label" style="margin-top:0;">GRAVY</div>
          <div class="metric-value" style="color:{hydro_color} !important;">{gv:+.2f}</div>
          <div style="font-size:11px;opacity:0.6;">{'Hydrophobic' if gv>0 else 'Hydrophilic'}</div>
        </div>
        <div>
          <div class="metric-label" style="margin-top:0;">Charge</div>
          <div class="metric-value">{charge_label}</div>
        </div>
        <div>
          <div class="metric-label" style="margin-top:0;">Aromatic</div>
          <div class="metric-value">{arom_pct:.0f}%</div>
        </div>
      </div>
      <div style="margin-top:14px;">
        <div class="metric-label" style="margin-top:0;">Hydrophobic residues</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar-fill" style="width:{min(hydro_pct,100):.0f}%;background:#1a8fd1;"></div>
        </div>
        <div class="metric-label">Charged residues</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar-fill" style="width:{min(charge_pct,100):.0f}%;background:#e67e22;"></div>
        </div>
      </div>
      <div style="margin-top:10px;">
        {''.join(f'<span class="aa-badge">{aa}</span>' for aa in seq[:30])}
        {'<span style="opacity:0.5;font-size:12px;"> +more</span>' if L > 30 else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# SECTION 6C - CAPTION HELPERS
# ==========================================================

def caption_distributions(df):
    seq_lengths    = [len(s) for s in df["peptide"]]
    gravy_vals     = [gravy_score(s) for s in df["peptide"]]
    mean_len       = np.mean(seq_lengths)
    min_len        = int(np.min(seq_lengths))
    max_len        = int(np.max(seq_lengths))
    dominant_taste = df["taste"].value_counts().idxmax()
    dominant_count = df["taste"].value_counts().max()
    n_classes      = df["taste"].nunique()
    mean_gravy     = np.mean(gravy_vals)
    gravy_label    = (
        "slightly hydrophobic" if mean_gravy > 0.2 else
        "slightly hydrophilic" if mean_gravy < -0.2 else
        "amphipathic (near-neutral)"
    )
    return (
        f"<strong>Length distribution (left):</strong> "
        f"Peptide lengths range from <strong>{min_len}</strong> to <strong>{max_len} aa</strong>, "
        f"with a mean of <strong>{mean_len:.1f} aa</strong>.<br><br>"
        f"<strong>Taste classes (centre):</strong> "
        f"&ldquo;{dominant_taste}&rdquo; is the most represented class "
        f"(<strong>{dominant_count}</strong> peptides out of {len(df)} total, "
        f"across {n_classes} classes).<br><br>"
        f"<strong>GRAVY distribution (right):</strong> "
        f"The dataset's mean GRAVY score is <strong>{mean_gravy:.2f}</strong>, "
        f"indicating the average peptide is <strong>{gravy_label}</strong>."
    )


def caption_pca(pca_model, class_names):
    var1  = pca_model.explained_variance_ratio_[0] * 100
    var2  = pca_model.explained_variance_ratio_[1] * 100
    total = var1 + var2
    return (
        f"Each dot represents one peptide compressed to 2 dimensions.<br><br>"
        f"<strong>PC1</strong> captures <strong>{var1:.1f}%</strong> and "
        f"<strong>PC2</strong> captures <strong>{var2:.1f}%</strong> — "
        f"together <strong>{total:.1f}%</strong> of all variation.<br><br>"
        f"Tight, well-separated clusters mean reliable class distinction; "
        f"overlapping clusters expect higher confusion matrix errors."
    )


def caption_confusion_taste(y_true, y_pred, class_names):
    acc     = accuracy_score(y_true, y_pred) * 100
    cm      = confusion_matrix(y_true, y_pred)
    cm_copy = cm.astype(float)
    np.fill_diagonal(cm_copy, 0)
    idx           = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    true_cls      = class_names[idx[0]]
    pred_cls      = class_names[idx[1]]
    worst_n       = int(cm_copy[idx])
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    best_cls      = class_names[np.argmax(per_class_acc)]
    worst_cls     = class_names[np.argmin(per_class_acc)]
    return (
        f"Taste model achieved <strong>{acc:.1f}% overall accuracy</strong> on the test set.<br><br>"
        f"<strong>Most common confusion:</strong> &ldquo;{true_cls}&rdquo; predicted as "
        f"&ldquo;{pred_cls}&rdquo; <strong>{worst_n} time(s)</strong>.<br><br>"
        f"<strong>Best class:</strong> &ldquo;{best_cls}&rdquo; &nbsp;|&nbsp; "
        f"<strong>Hardest class:</strong> &ldquo;{worst_cls}&rdquo;."
    )


def caption_confusion_sol(y_true, y_pred, class_names):
    acc     = accuracy_score(y_true, y_pred) * 100
    cm      = confusion_matrix(y_true, y_pred)
    cm_copy = cm.astype(float)
    np.fill_diagonal(cm_copy, 0)
    idx      = np.unravel_index(np.argmax(cm_copy), cm_copy.shape)
    true_cls = class_names[idx[0]]
    pred_cls = class_names[idx[1]]
    worst_n  = int(cm_copy[idx])
    return (
        f"Solubility model achieved <strong>{acc:.1f}% overall accuracy</strong>.<br><br>"
        f"<strong>Most common misclassification:</strong> "
        f"&ldquo;{true_cls}&rdquo; predicted as &ldquo;{pred_cls}&rdquo; "
        f"<strong>{worst_n} time(s)</strong>.<br><br>"
        f"Errors typically occur for peptides near the solubility decision boundary."
    )


def caption_feature_importance(model, feature_names, top_n=20):
    imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    top3        = [prettify_feature(f) for f in imp["Feature"].iloc[:3]]
    top3_scores = imp["Importance"].iloc[:3].tolist()
    n_dpc = sum(1 for f in imp["Feature"] if f.startswith("DPC_"))
    n_aa  = sum(1 for f in imp["Feature"] if f.startswith("AA_"))
    dpc_note = (
        "Dipeptide patterns dominate — <em>sequential context</em> matters more than composition alone."
        if n_dpc > n_aa else
        "Single amino acid composition is the stronger predictor here."
    )
    return (
        f"Top {top_n} features ranked by importance.<br><br>"
        f"<strong>#1 — {top3[0]}</strong> (score: {top3_scores[0]:.4f})<br>"
        f"<strong>#2 — {top3[1]}</strong> (score: {top3_scores[1]:.4f})<br>"
        f"<strong>#3 — {top3[2]}</strong> (score: {top3_scores[2]:.4f})<br><br>"
        f"{n_dpc} dipeptide (DPC) and {n_aa} single amino acid features in top {top_n}. {dpc_note}"
    )


def caption_docking(y_true, y_pred):
    r2        = r2_score(y_true, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_true, y_pred))
    min_score = y_true.min()
    max_score = y_true.max()
    quality   = "strong" if r2 >= 0.75 else ("moderate" if r2 >= 0.5 else "weak")
    return (
        f"Each point is one peptide from the test set. Red dashed = perfect prediction.<br><br>"
        f"<strong>R² = {r2:.3f}</strong> — <strong>{quality} fit</strong>. "
        f"Explains <strong>{r2*100:.1f}%</strong> of variance.<br>"
        f"<strong>RMSE = {rmse:.2f} kcal/mol</strong> — typical error per peptide.<br>"
        f"True scores range <strong>{min_score:.2f}</strong> to <strong>{max_score:.2f} kcal/mol</strong>."
    )


def caption_ramachandran(phi_psi, seq=""):
    if not phi_psi:
        return "No φ/ψ angles — peptide may be too short (fewer than 3 residues)."
    n_total  = len(phi_psi)
    n_helix  = sum(1 for p, s in phi_psi if -180<=p<=-45 and -75<=s<=-15)
    n_sheet  = sum(1 for p, s in phi_psi if -180<=p<=-45 and 90<=s<=180)
    n_dis    = n_total - n_helix - n_sheet
    dominant = "α-helix" if n_helix >= n_sheet else "β-sheet"
    return (
        f"Each dot is one backbone torsion angle pair (φ, ψ)"
        f"{f' for this <strong>{len(seq)}-residue peptide</strong>' if seq else ''}.<br><br>"
        f"<strong>α-helix region:</strong> ~{n_helix/n_total*100:.0f}% of residues<br>"
        f"<strong>β-sheet region:</strong> ~{n_sheet/n_total*100:.0f}% of residues<br>"
        f"<strong>Outside allowed:</strong> ~{n_dis/n_total*100:.0f}% of residues<br><br>"
        f"Backbone geometry primarily favours <strong>{dominant}</strong> character."
    )


def caption_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if n < 2:
        return "Distance map could not be computed — fewer than 2 Cα atoms."
    mask     = ~np.eye(n, dtype=bool)
    off_diag = dist_matrix[mask]
    max_dist = off_diag.max()
    min_dist = off_diag.min()
    long_range = sum(
        1 for i in range(n) for j in range(n)
        if abs(i-j) > 3 and dist_matrix[i,j] < 8.0
    )
    fold_note = (
        "suggesting the peptide <strong>folds back on itself</strong>"
        if long_range > 0 else
        "consistent with an <strong>extended / linear conformation</strong>"
    )
    return (
        f"Pairwise Cα–Cα distances — <strong>darker = closer in 3D</strong>.<br><br>"
        f"Nearest non-adjacent: {min_dist:.1f} Å | Furthest pair: {max_dist:.1f} Å<br>"
        f"Long-range contacts (|i−j|>3, &lt;8Å): <strong>{long_range}</strong> — {fold_note}."
    )


# ==========================================================
# SECTION 6D - MATPLOTLIB THEME
# ==========================================================

def _is_dark_mode():
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def get_plot_colors():
    dark = _is_dark_mode()
    if dark:
        return {
            "fig_bg": "#1a1d2e", "ax_bg": "#1e2140", "text": "#e8edf8",
            "grid": "#2e3560", "accent1": "#5c7cfa", "accent2": "#748ffc",
            "accent3": "#4dd0e1", "red": "#ff6b6b", "orange": "#ffa94d", "tick": "#c5cff0",
        }
    return {
        "fig_bg": "#f8f9fc", "ax_bg": "#ffffff", "text": "#1a1d2e",
        "grid": "#d0d5e8", "accent1": "#1a56db", "accent2": "#4361ee",
        "accent3": "#0b7285", "red": "#c0392b", "orange": "#e67e22", "tick": "#4a5170",
    }


def apply_plot_style(fig, axes_list):
    C = get_plot_colors()
    fig.patch.set_facecolor(C["fig_bg"])
    for ax in axes_list:
        ax.set_facecolor(C["ax_bg"])
        ax.tick_params(colors=C["tick"], labelsize=10)
        ax.xaxis.label.set_color(C["text"])
        ax.yaxis.label.set_color(C["text"])
        ax.title.set_color(C["text"])
        for spine in ax.spines.values():
            spine.set_edgecolor(C["grid"])
        ax.tick_params(axis="x", colors=C["tick"])
        ax.tick_params(axis="y", colors=C["tick"])


# ==========================================================
# SECTION 7 - PDB HELPER UTILITIES
# ==========================================================

def _write_temp_pdb(pdb_text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(pdb_text)
    tmp.close()
    return tmp.name


def _extract_plddt_from_pdb(pdb_text: str) -> list:
    """Extract per-residue pLDDT from the B-factor column of a PDB file."""
    seen = set()
    vals = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        chain   = line[21]
        res_seq = line[22:26].strip()
        key     = (chain, res_seq)
        if key not in seen:
            seen.add(key)
            try:
                vals.append(float(line[60:66].strip()))
            except ValueError:
                pass
    return vals


def plddt_label(score: float) -> tuple:
    """Return (label, hex_color) for a pLDDT score."""
    if score >= 90:
        return "Very High", "#12b886"
    if score >= 70:
        return "High", "#1a8fd1"
    if score >= 50:
        return "Medium", "#f39c12"
    return "Low", "#c0392b"


# ==========================================================
# SECTION 8 - STRUCTURE ANALYSIS
# ==========================================================

def show_structure(pdb_text: str):
    view = py3Dmol.view(width=800, height=480)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.SAS, {"opacity": 0.08})
    view.zoomTo()
    view.spin(False)
    return view


def ramachandran(pdb_text: str) -> list:
    if not pdb_text or not pdb_text.strip():
        return []
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        model     = structure[0]
        pts       = []
        for pp in PPBuilder().build_peptides(model):
            for phi, psi in pp.get_phi_psi_list():
                if phi is not None and psi is not None:
                    pts.append((np.degrees(phi), np.degrees(psi)))
        return pts
    except Exception:
        return []
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ca_distance_map(pdb_text: str) -> np.ndarray:
    if not pdb_text or not pdb_text.strip():
        return np.zeros((1, 1))
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        cas = [
            r["CA"].get_vector().get_array()
            for r in structure.get_residues()
            if "CA" in r
        ]
        if not cas:
            return np.zeros((1, 1))
        coords = np.array(cas)
        diff   = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))
    except Exception:
        return np.zeros((1, 1))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ca_rmsd(pdb_text: str):
    if not pdb_text or not pdb_text.strip():
        return None
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        cas = [
            r["CA"].get_vector()
            for r in structure.get_residues()
            if "CA" in r
        ]
        if len(cas) < 2:
            return None
        ref = cas[0]
        return float(np.sqrt(np.mean([(v - ref).norm() ** 2 for v in cas])))
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def run_dssp(pdb_text: str) -> dict:
    fallback = {"helix": 0.0, "sheet": 0.0, "coil": 0.0, "raw": []}
    if not pdb_text or not pdb_text.strip():
        return fallback
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        model     = structure[0]
        dssp      = DSSP(model, tmp_path)
        raw       = [dssp[k][2] for k in dssp.property_keys]
        total     = len(raw)
        if total == 0:
            return fallback
        helix_chars = {"H", "G", "I"}
        sheet_chars = {"E", "B"}
        helix = sum(1 for s in raw if s in helix_chars) / total * 100
        sheet = sum(1 for s in raw if s in sheet_chars) / total * 100
        coil  = 100.0 - helix - sheet
        return {"helix": round(helix, 1), "sheet": round(sheet, 1),
                "coil": round(coil, 1),   "raw": raw}
    except Exception:
        return fallback
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def radius_of_gyration(pdb_text: str):
    if not pdb_text:
        return None
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        coords    = np.array([a.get_vector().get_array()
                               for a in structure.get_atoms()])
        if len(coords) == 0:
            return None
        centroid = coords.mean(axis=0)
        return float(np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()))
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def compute_sasa(pdb_text: str):
    if not pdb_text:
        return None
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        sr        = ShrakeRupley()
        sr.compute(structure, level="S")
        return round(structure.sasa, 2)
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def count_hbonds(pdb_text: str) -> int:
    if not pdb_text:
        return 0
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        residues  = list(structure.get_residues())
        donors    = []
        acceptors = []
        for res in residues:
            if "N" in res:
                donors.append(res["N"].get_vector())
            if "O" in res:
                acceptors.append(res["O"].get_vector())
        count = 0
        for d in donors:
            for a in acceptors:
                if (d - a).norm() < 3.5:
                    count += 1
        return count
    except Exception:
        return 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def count_disulfide_bonds(pdb_text: str) -> int:
    if not pdb_text:
        return 0
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        parser    = PDBParser(QUIET=True)
        structure = parser.get_structure("x", tmp_path)
        cys_sg    = [
            res["SG"].get_vector()
            for res in structure.get_residues()
            if res.get_resname() == "CYS" and "SG" in res
        ]
        count = 0
        for i in range(len(cys_sg)):
            for j in range(i + 1, len(cys_sg)):
                if (cys_sg[i] - cys_sg[j]).norm() < 2.1:
                    count += 1
        return count
    except Exception:
        return 0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ==========================================================
# SECTION 9 - PLOT FUNCTIONS
# ==========================================================

def plot_plddt(plddt_vals: list, seq: str = ""):
    C   = get_plot_colors()
    n   = len(plddt_vals)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.18), 4))
    apply_plot_style(fig, [ax])
    colors = ["#12b886" if v >= 90 else "#1a8fd1" if v >= 70 else
              "#f39c12" if v >= 50 else "#c0392b" for v in plddt_vals]
    ax.bar(range(n), plddt_vals, color=colors, width=0.85)
    ax.axhline(90, color="#12b886", linestyle="--", lw=1.2, alpha=0.7, label="Very High (≥90)")
    ax.axhline(70, color="#1a8fd1", linestyle="--", lw=1.2, alpha=0.7, label="High (≥70)")
    ax.axhline(50, color="#f39c12", linestyle="--", lw=1.2, alpha=0.7, label="Medium (≥50)")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue Index", fontsize=11, labelpad=8)
    ax.set_ylabel("pLDDT", fontsize=11, labelpad=8)
    ax.set_title("Per-Residue pLDDT Confidence", fontsize=13, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    leg = ax.legend(fontsize=8, loc="lower right",
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_pca(X, y_labels, class_names, title="PCA"):
    C = get_plot_colors()
    pca    = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var1   = pca.explained_variance_ratio_[0] * 100
    var2   = pca.explained_variance_ratio_[1] * 100
    palette = plt.cm.get_cmap("tab20", len(class_names))
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    for i, cls in enumerate(class_names):
        mask = y_labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cls, alpha=0.75, s=35,
                   color=palette(i), edgecolors="none")
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",
                       borderaxespad=0, title="Taste class", title_fontsize=9,
                       facecolor=C["fig_bg"], edgecolor=C["grid"])
    legend.get_title().set_color(C["text"])
    for text in legend.get_texts():
        text.set_color(C["text"])
    plt.tight_layout()
    return fig, pca


def plot_confusion(y_true, y_pred, class_names, title, cmap):
    C   = get_plot_colors()
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.75), max(5, n * 0.6)))
    apply_plot_style(fig, [ax])
    annot_color = "#111122" if not _is_dark_mode() else "#ffffff"
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.4, linecolor=C["grid"],
                annot_kws={"size": 11, "color": annot_color})
    ax.set_title(f"{title}  —  Accuracy: {acc*100:.1f}%",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("True",      fontsize=12, labelpad=10)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(C["text"])
    cbar.ax.tick_params(colors=C["text"])
    plt.xticks(rotation=45, ha="right", fontsize=9, color=C["tick"])
    plt.yticks(rotation=0,  fontsize=9, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_docking(y_true, y_pred):
    C    = get_plot_colors()
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lims = [min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors="none", color=C["accent1"], s=45)
    ax.plot(lims, lims, color=C["red"], linestyle="--", lw=1.8, label="Perfect fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.annotate(f"R² = {r2:.3f}\nRMSE = {rmse:.2f} kcal/mol",
                xy=(0.05, 0.87), xycoords="axes fraction", fontsize=11,
                color=C["text"],
                bbox=dict(boxstyle="round,pad=0.5", fc=C["fig_bg"], ec=C["grid"], alpha=0.95))
    ax.set_xlabel("True Docking Score (kcal/mol)",      fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted Docking Score (kcal/mol)", fontsize=12, labelpad=10)
    ax.set_title("Docking: True vs Predicted (test set)", fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(fontsize=10, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for text in legend.get_texts():
        text.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    C = get_plot_colors()
    imp = pd.DataFrame({
        "Feature":    [prettify_feature(f) for f in feature_names],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp))[::-1])
    fig, ax = plt.subplots(figsize=(8, 7))
    apply_plot_style(fig, [ax])
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1], color=colors, edgecolor=C["grid"])
    ax.set_xlabel("Importance Score", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Features — Taste Model", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=9, colors=C["tick"])
    ax.tick_params(axis="x", labelsize=9, colors=C["tick"])
    plt.tight_layout()
    return fig


def plot_distributions(df):
    C            = get_plot_colors()
    seq_lengths  = [len(s) for s in df["peptide"]]
    taste_counts = df["taste"].value_counts()
    gravy_vals   = [gravy_score(s) for s in df["peptide"]]
    fig, axes    = plt.subplots(1, 3, figsize=(16, 5))
    apply_plot_style(fig, axes)

    mean_len = np.mean(seq_lengths)
    axes[0].hist(seq_lengths, bins=20, color=C["accent1"], edgecolor=C["grid"], alpha=0.85)
    axes[0].axvline(mean_len, color=C["red"], linestyle="--", lw=2, label=f"Mean = {mean_len:.1f} aa")
    axes[0].set_xlabel("Sequence Length (aa)", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Count", fontsize=11, labelpad=8)
    axes[0].set_title("Peptide Length Distribution", fontsize=12, fontweight="bold", pad=10)
    leg0 = axes[0].legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg0.get_texts(): t.set_color(C["text"])

    n_cls      = len(taste_counts)
    bar_colors = plt.cm.get_cmap("tab20", n_cls)(np.linspace(0, 1, n_cls))
    axes[1].barh(taste_counts.index, taste_counts.values, color=bar_colors, edgecolor=C["grid"], alpha=0.9)
    axes[1].set_xlabel("Number of Peptides", fontsize=11, labelpad=8)
    axes[1].set_title("Taste Class Distribution", fontsize=12, fontweight="bold", pad=10)
    axes[1].tick_params(axis="y", labelsize=9, colors=C["tick"])
    axes[1].tick_params(axis="x", labelsize=9, colors=C["tick"])
    for i, v in enumerate(taste_counts.values):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=9, color=C["text"])

    axes[2].hist(gravy_vals, bins=20, color=C["accent2"], edgecolor=C["grid"], alpha=0.85)
    axes[2].axvline(0, color=C["red"], linestyle="--", lw=2, label="Hydrophilic | Hydrophobic")
    axes[2].axvline(np.mean(gravy_vals), color=C["orange"], linestyle="--", lw=2,
                    label=f"Mean = {np.mean(gravy_vals):.2f}")
    axes[2].set_xlabel("GRAVY Score", fontsize=11, labelpad=8)
    axes[2].set_ylabel("Count", fontsize=11, labelpad=8)
    axes[2].set_title("GRAVY Hydrophobicity Distribution", fontsize=12, fontweight="bold", pad=10)
    leg2 = axes[2].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg2.get_texts(): t.set_color(C["text"])

    plt.tight_layout(pad=2.5)
    return fig


def plot_ramachandran(phi_psi):
    C = get_plot_colors()
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.fill([-180,-180,-45,-45,-180], [-75,-45,-45,-75,-75], color="#4CAF50", alpha=0.25, label="α-helix (allowed)")
    ax.fill([-180,-180,-90,-90,-180], [90,180,180,90,90],    color="#2196F3", alpha=0.25, label="β-sheet (allowed)")
    ax.fill([45,45,90,90,45],         [0,90,90,0,0],         color="#FF9800", alpha=0.2,  label="L-helix (allowed)")
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=50, color=C["red"], zorder=5, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.axvline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi φ (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("Psi ψ (°)", fontsize=12, labelpad=10)
    ax.set_title("Ramachandran Plot", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, loc="upper right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    return fig


def plot_distance_map(dist_matrix, seq=""):
    C = get_plot_colors()
    n = dist_matrix.shape[0]
    if seq and len(seq) == n:
        labels = [f"{aa}{i+1}" for i, aa in enumerate(seq)]
    else:
        labels = [str(i+1) for i in range(n)]
    tick_step   = max(1, n // 15)
    show_labels = [labels[i] if i % tick_step == 0 else "" for i in range(n)]
    size        = max(5, n * 0.3 + 2)
    fig, ax     = plt.subplots(figsize=(size, size))
    apply_plot_style(fig, [ax])
    sns.heatmap(dist_matrix, cmap="viridis", ax=ax,
                xticklabels=show_labels, yticklabels=show_labels,
                linewidths=0, cbar_kws={"label": "Distance (Å)"})
    ax.set_title("Cα Distance Map", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Residue", fontsize=12, labelpad=10)
    ax.set_ylabel("Residue", fontsize=12, labelpad=10)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(C["text"])
    cbar.ax.tick_params(colors=C["text"])
    plt.xticks(rotation=45, ha="right", fontsize=8, color=C["tick"])
    plt.yticks(rotation=0,  fontsize=8, color=C["tick"])
    plt.tight_layout()
    return fig


# ==========================================================
# SECTION 10 - STRUCTURAL ANALYSIS HELPER
# ==========================================================

def render_structural_analysis(
    pdb_text: str,
    prefix: str = "",
    seq: str = "",
    plddt_vals: list = None,
    pae: np.ndarray = None,
):
    """
    Full structural analysis panel:
      - DSSP secondary structure
      - Structural metrics (Rg, SASA, H-bonds, disulfides)
      - pLDDT per-residue plot (if values provided)
      - Ramachandran plot
      - Cα distance map
    """
    if not pdb_text or not pdb_text.strip():
        st.warning("No PDB data available for structural analysis.")
        return

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🔩 Secondary Structure (DSSP)")
    dssp_result = run_dssp(pdb_text)
    d1, d2, d3 = st.columns(3)
    d1.metric("α-Helix (%)",    f"{dssp_result['helix']}%")
    d2.metric("β-Sheet (%)",    f"{dssp_result['sheet']}%")
    d3.metric("Coil/Other (%)", f"{dssp_result['coil']}%")

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🔬 Structural Metrics")
    rg       = radius_of_gyration(pdb_text)
    sasa     = compute_sasa(pdb_text)
    hbonds   = count_hbonds(pdb_text)
    ss_bonds = count_disulfide_bonds(pdb_text)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Radius of Gyration", f"{rg:.2f} Å"   if rg   is not None else "N/A")
    m2.metric("SASA",               f"{sasa:.1f} Å²" if sasa is not None else "N/A")
    m3.metric("H-Bonds (est.)",     str(hbonds))
    m4.metric("Disulfide Bonds",    str(ss_bonds))

    if plddt_vals and len(plddt_vals) > 0:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Per-Residue pLDDT Confidence")
        fig_plddt = plot_plddt(plddt_vals, seq=seq)
        save_fig(fig_plddt, f"{prefix}plddt.png")
        st.pyplot(fig_plddt)
        plt.close(fig_plddt)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 📐 Ramachandran Plot")
    phi_psi = ramachandran(pdb_text)
    if not phi_psi:
        st.info("No phi/psi angles computed — peptide may be too short (needs ≥3 residues).")
    fig_rama = plot_ramachandran(phi_psi)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.pyplot(fig_rama)
    plt.close(fig_rama)
    show_caption(caption_ramachandran(phi_psi, seq=seq))

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Cα Distance Map")
    try:
        dist_map = ca_distance_map(pdb_text)
        fig_dist = plot_distance_map(dist_map, seq=seq)
        save_fig(fig_dist, f"{prefix}ca_distance_map.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        show_caption(caption_distance_map(dist_map, seq=seq))
    except Exception as e:
        st.warning(f"Distance map could not be rendered: {e}")


# ==========================================================
# SECTION 11 - MODEL TRAINING
# ==========================================================

@st.cache_data
def train_models():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found: {DATASET_PATH}")
        st.stop()

    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 1].reset_index(drop=True)
    df = df[
        df["taste"].notna()
        & df["solubility"].notna()
        & df["docking score (kcal/mol)"].notna()
    ].reset_index(drop=True)

    df["solubility"] = df["solubility"].str.strip().str.rstrip(".")
    df["taste"]      = simplify_taste(df["taste"])

    X        = build_feature_table(df["peptide"])
    le_taste = LabelEncoder()
    le_sol   = LabelEncoder()
    y_taste  = le_taste.fit_transform(df["taste"])
    y_sol    = le_sol.fit_transform(df["solubility"])
    y_dock   = df["docking score (kcal/mol)"].values

    idx            = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_taste)
    Xtr, Xte       = X.iloc[tr_idx], X.iloc[te_idx]
    yt_tr, yt_te   = y_taste[tr_idx], y_taste[te_idx]
    ys_tr, ys_te   = y_sol[tr_idx],   y_sol[te_idx]
    yd_tr, yd_te   = y_dock[tr_idx],  y_dock[te_idx]

    taste_model = ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    sol_model   = ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    dock_model  = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr, ys_tr)
    dock_model.fit(Xtr, yd_tr)

    metrics = {
        "Taste accuracy":      accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1":            f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1":       f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE":        np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R2":          r2_score(yd_te, dock_model.predict(Xte)),
    }

    return (
        df, X, Xte, yt_te, ys_te, yd_te,
        taste_model, sol_model, dock_model,
        le_taste, le_sol, metrics,
    )


# ==========================================================
# SECTION 12 - LOAD MODELS
# ==========================================================

(
    df_all, X_all, X_test, yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics,
) = train_models()


# ==========================================================
# SECTION 13 - PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics, prediction, image_paths):
    file_name = "PepTastePredictor_Full_Report.pdf"
    styles    = getSampleStyleSheet()
    doc       = SimpleDocTemplate(file_name, pagesize=A4)
    story     = []
    story.append(Paragraph("<b>PepTastePredictor</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking, and structural analysis platform. "
        "Structure prediction via external services (ESM Atlas / AlphaFold Server).",
        styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {round(v, 4)}", styles["Normal"]))
    story.append(Spacer(1, 12))
    if prediction:
        story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
        for k, v in prediction.items():
            story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
        story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))
    story.append(Spacer(1, 12))
    for img in image_paths:
        if os.path.exists(img):
            story.append(RLImage(img, width=450, height=300))
            story.append(Spacer(1, 18))
    doc.build(story)
    return file_name


# ==========================================================
# SECTION 14 - HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🧬 PepTastePredictor</h1>
<p>
An integrated machine learning &amp; structural bioinformatics platform for peptide
taste, solubility, docking, and 3D structure analysis.
Generate structures via <strong>ESM Atlas</strong> or <strong>AlphaFold Server</strong>,
then upload the PDB for complete analysis.
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 15 - MODE SELECTION
# ==========================================================

st.markdown("## 🔧 Prediction & Analysis Mode")
mode = st.radio(
    "Choose the analysis mode",
    ["Single Peptide Prediction", "Batch Peptide Prediction", "PDB Upload & Structural Analysis"],
    horizontal=True,
)

if "current_mode" not in st.session_state or st.session_state.current_mode != mode:
    st.session_state.pdf_figures     = []
    st.session_state.show_analytics  = False
    st.session_state.last_prediction = {}
    st.session_state.live_results    = None
    st.session_state.current_mode    = mode


# ==========================================================
# SECTION 16 - SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("## 🔬 Single Peptide Prediction")

    seq_raw = st.text_area(
        "Enter peptide sequence (FASTA or plain single-letter code)",
        help="Accepts 1–2500 amino acids. FASTA headers are stripped automatically.",
        placeholder="Paste sequence or FASTA here…",
        key="single_seq_input",
        height=120,
    )
    seq_clean = clean_sequence(seq_raw)

    # ── Sequence counter ───────────────────────────────────
    if seq_raw:
        valid        = len(seq_clean)
        raw_stripped = re.sub(r">[^\n]*\n?", "", seq_raw)
        raw_stripped = raw_stripped.replace(" ", "").replace("\n", "").replace("\t", "").upper()
        invalid      = len([c for c in raw_stripped if c not in AA])
        badge_color  = "#12b886" if valid > 0 else "#c0392b"
        inv_note     = (
            f" &nbsp; <span style='color:#c0392b;'>({invalid} invalid character(s) removed)</span>"
            if invalid else ""
        )
        st.markdown(
            f'<div class="seq-counter">'
            f'Valid amino acids detected: '
            f'<span style="color:{badge_color};font-weight:800;">{valid}</span>'
            f'{inv_note}</div>',
            unsafe_allow_html=True,
        )

    # ── Live dynamic preview ───────────────────────────────
    if seq_clean:
        render_live_preview(seq_clean)

    # ── Quick ML predictions ──────────────────────────────
    ml_seq = seq_clean[:100] if len(seq_clean) > 100 else seq_clean
    if ml_seq:
        Xp    = pd.DataFrame([model_features(ml_seq)])
        taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
        sol   = le_sol.inverse_transform(sol_model.predict(Xp))[0]
        dock  = dock_model.predict(Xp)[0]
        emoji = taste_emoji(taste)

        sol_color  = "#12b886" if "soluble" in sol.lower() else "#e67e22"
        dock_color = "#12b886" if dock < -6 else ("#f39c12" if dock < -4 else "#c0392b")

        st.markdown(f"""
        <div class="card">
          <div style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:16px;">
            <span class="live-indicator"></span>ML Prediction Results
          </div>
          <div style="display:flex;gap:40px;flex-wrap:wrap;align-items:flex-start;">
            <div>
              <div class="metric-label" style="margin-top:0;">Taste</div>
              <div class="metric-value">{emoji} {taste}</div>
            </div>
            <div>
              <div class="metric-label" style="margin-top:0;">Solubility</div>
              <div class="metric-value" style="color:{sol_color} !important;">{sol}</div>
            </div>
            <div>
              <div class="metric-label" style="margin-top:0;">Docking Score</div>
              <div class="metric-value" style="color:{dock_color} !important;">{dock:.3f} kcal/mol</div>
              <div style="font-size:11px;opacity:0.6;">{'Strong binder' if dock<-6 else 'Moderate' if dock<-4 else 'Weak binder'}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if len(seq_clean) > 100:
            st.info(
                f"ℹ️ ML predictions use the first 100 aa of your {len(seq_clean)}-aa sequence. "
                "Submit the full sequence to an external folding server below."
            )

        st.session_state.last_prediction = {
            "Sequence (first 60 aa)":   seq_clean[:60] + ("…" if len(seq_clean) > 60 else ""),
            "Full sequence length":     len(seq_clean),
            "Predicted taste":          taste,
            "Predicted solubility":     sol,
            "Docking score (kcal/mol)": round(dock, 3),
        }
        st.session_state.show_analytics = True

    # ── Physicochemical Properties ─────────────────────────
    if seq_clean:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📌 Physicochemical Properties")
        phys = physicochemical_features(seq_clean[:100] if len(seq_clean) > 100 else seq_clean)
        cols = st.columns(min(len(phys), 4))
        for i, (k, v) in enumerate(phys.items()):
            cols[i % len(cols)].metric(k, v)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🧪 Amino Acid Composition")
        comp      = composition_features(seq_clean)
        comp_cols = st.columns(len(comp))
        for i, (k, v) in enumerate(comp.items()):
            comp_cols[i].metric(k, f"{v}%")
            comp_cols[i].markdown(
                f'<div class="progress-bar-wrap"><div class="progress-bar-fill" '
                f'style="width:{v}%;background:#1a8fd1;"></div></div>',
                unsafe_allow_html=True,
            )

    # ==========================================================
    # SECTION 16B - EXTERNAL STRUCTURE PREDICTION WORKFLOW
    # ==========================================================

    if seq_clean:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("## 🌐 External Structure Prediction")

        # Sequence copy box
        st.markdown("**Your peptide sequence (copy this):**")
        st.code(seq_clean, language=None)

        col_copy1, col_copy2 = st.columns(2)
        col_copy1.download_button(
            "⬇️ Download as FASTA",
            f">peptide\n{seq_clean}\n",
            file_name="peptide.fasta",
            mime="text/plain",
        )
        col_copy2.download_button(
            "⬇️ Download sequence (.txt)",
            seq_clean,
            file_name="peptide_sequence.txt",
            mime="text/plain",
        )

        # External server links
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("**Open an external folding server:**")
        btn_col1, btn_col2 = st.columns(2)
        btn_col1.markdown(
            '<a href="https://esmatlas.com/resources?action=fold" target="_blank">'
            '<button style="width:100%;padding:14px 0;font-size:15px;font-weight:700;'
            'border-radius:10px;border:2px solid #12b886;background:rgba(18,184,134,0.12);'
            'color:#12b886;cursor:pointer;">🌿 Open ESM Atlas</button></a>',
            unsafe_allow_html=True,
        )
        btn_col2.markdown(
            '<a href="https://alphafoldserver.com" target="_blank">'
            '<button style="width:100%;padding:14px 0;font-size:15px;font-weight:700;'
            'border-radius:10px;border:2px solid #1a8fd1;background:rgba(26,143,209,0.12);'
            'color:#1a8fd1;cursor:pointer;">🔬 Open AlphaFold Server</button></a>',
            unsafe_allow_html=True,
        )

        # Step-by-step instructions
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="ext-card">
          <div style="font-size:13px;font-weight:700;opacity:0.55;text-transform:uppercase;
                      letter-spacing:0.08em;margin-bottom:18px;">How to get your structure</div>
          <div class="step-row">
            <div class="step-num">1</div>
            <div class="step-text"><strong>Copy</strong> the peptide sequence shown above (use the download buttons or select all from the code box).</div>
          </div>
          <div class="step-row">
            <div class="step-num">2</div>
            <div class="step-text"><strong>Open</strong> ESM Atlas or AlphaFold Server using the buttons above.</div>
          </div>
          <div class="step-row">
            <div class="step-num">3</div>
            <div class="step-text"><strong>Paste</strong> your sequence into the server's input field and run the prediction.</div>
          </div>
          <div class="step-row">
            <div class="step-num">4</div>
            <div class="step-text"><strong>Download</strong> the resulting PDB file from the server's results page.</div>
          </div>
          <div class="step-row">
            <div class="step-num">5</div>
            <div class="step-text"><strong>Upload</strong> the PDB file below to run the full structural analysis.</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # PDB uploader
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        uploaded_pdb = st.file_uploader(
            "📂 Upload generated PDB structure",
            type=["pdb"],
            key="single_pdb_upload",
            help="Upload the PDB file downloaded from ESM Atlas or AlphaFold Server.",
        )

        if uploaded_pdb is not None:
            try:
                pdb_text = uploaded_pdb.read().decode("utf-8")
            except Exception as e:
                st.error(f"Could not read the uploaded PDB file: {e}")
                pdb_text = ""

            if pdb_text and pdb_text.strip():
                st.session_state.pdb_text = pdb_text

                n_atoms    = sum(1 for l in pdb_text.splitlines() if l.startswith("ATOM"))
                n_residues = len({l[22:26].strip() for l in pdb_text.splitlines() if l.startswith("ATOM")})
                plddt_vals = _extract_plddt_from_pdb(pdb_text)

                # Check if pLDDT values are meaningful (AlphaFold stores them in B-factor)
                has_plddt = len(plddt_vals) > 0 and max(plddt_vals) > 1.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ATOM records", n_atoms)
                c2.metric("Residues",     n_residues)
                c3.metric("File size",    f"{max(1, len(pdb_text) // 1024)} KB")
                if has_plddt:
                    mean_pl = float(np.mean(plddt_vals))
                    lbl, lcol = plddt_label(mean_pl)
                    c4.metric("Mean pLDDT", f"{mean_pl:.1f}")

                # Downloads
                dl1, dl2 = st.columns(2)
                dl1.download_button("⬇️ Re-download PDB", pdb_text, file_name="structure.pdb")
                dl2.download_button(
                    "⬇️ FASTA",
                    f">uploaded_peptide\n{seq_clean}\n",
                    file_name="sequence.fasta",
                )

                # 3D viewer
                st.markdown("### 🧬 3D Structure Viewer")
                try:
                    st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)
                except Exception as e:
                    st.warning(f"3D viewer could not render: {e}")

                rmsd_val = ca_rmsd(pdb_text)
                if rmsd_val is not None:
                    st.success(f"Cα RMSD from first residue: **{rmsd_val:.3f} Å**")

                render_structural_analysis(
                    pdb_text,
                    prefix="single_",
                    seq=seq_clean,
                    plddt_vals=plddt_vals if has_plddt else None,
                )
            else:
                st.error("Uploaded PDB file is empty or could not be decoded.")


# ==========================================================
# SECTION 17 - BATCH PEPTIDE PREDICTION MODE
# ==========================================================

elif mode == "Batch Peptide Prediction":

    st.markdown("## 📦 Batch Peptide Prediction")
    batch_file = st.file_uploader("Upload CSV file with a column named 'peptide'", type=["csv"])

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        if "peptide" not in batch_df.columns:
            st.error("CSV must contain a column named 'peptide'.")
        else:
            batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
            batch_df = batch_df[batch_df["peptide"].str.len() >= 1].reset_index(drop=True)

            if batch_df.empty:
                st.error("No valid peptide sequences found in the uploaded file.")
            else:
                progress = st.progress(0, text="Processing peptides…")
                total    = len(batch_df)
                chunk    = max(1, total // 20)

                tastes, sols, docks = [], [], []
                for i, row in batch_df.iterrows():
                    try:
                        ml_seq = row["peptide"][:100]
                        Xr = pd.DataFrame([model_features(ml_seq)])
                        tastes.append(le_taste.inverse_transform(taste_model.predict(Xr))[0])
                        sols.append(le_sol.inverse_transform(sol_model.predict(Xr))[0])
                        docks.append(round(dock_model.predict(Xr)[0], 3))
                    except Exception:
                        tastes.append("Error")
                        sols.append("Error")
                        docks.append(None)
                    if i % chunk == 0:
                        progress.progress(min(int((i + 1) / total * 100), 100),
                                          text=f"Processing {i+1}/{total} peptides…")

                progress.progress(100, text="Done!")
                batch_df["Predicted Taste"]         = tastes
                batch_df["Predicted Solubility"]    = sols
                batch_df["Predicted Docking Score"] = docks

                st.markdown("### 📊 Batch Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Peptides", total)
                s2.metric("Unique Tastes", batch_df["Predicted Taste"].nunique())
                s3.metric(
                    "Soluble (%)",
                    f"{100 * batch_df['Predicted Solubility'].str.contains('oluble', case=False, na=False).mean():.1f}%",
                )
                valid_docks = batch_df["Predicted Docking Score"].dropna()
                s4.metric(
                    "Avg Docking",
                    f"{valid_docks.mean():.2f} kcal/mol" if len(valid_docks) else "N/A",
                )

                fig_b, ax_b = plt.subplots(figsize=(8, 3))
                C = get_plot_colors()
                apply_plot_style(fig_b, [ax_b])
                tc       = batch_df["Predicted Taste"].value_counts()
                colors_b = plt.cm.get_cmap("tab20", len(tc))(np.linspace(0, 1, len(tc)))
                ax_b.barh(tc.index, tc.values, color=colors_b)
                ax_b.set_xlabel("Count", color=C["text"])
                ax_b.set_title("Batch Taste Distribution", color=C["text"], fontweight="bold")
                st.pyplot(fig_b)
                plt.close(fig_b)

                st.markdown("### ✅ Batch Results")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Batch Predictions",
                    batch_df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                )
                st.session_state.show_analytics = True


# ==========================================================
# SECTION 18 - PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

elif mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyze PDB Structure")
    st.info(
        "Generate your structure using [ESM Atlas](https://esmatlas.com/resources?action=fold) "
        "or [AlphaFold Server](https://alphafoldserver.com), then upload the PDB file here.",
        icon="🌐",
    )
    uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])

    if uploaded_pdb is not None:
        try:
            pdb_text = uploaded_pdb.read().decode("utf-8")
        except Exception as e:
            st.error(f"Could not read the uploaded PDB file: {e}")
            pdb_text = ""

        if pdb_text and pdb_text.strip():
            st.session_state.pdb_text       = pdb_text
            st.session_state.show_analytics = True

            n_atoms    = sum(1 for l in pdb_text.splitlines() if l.startswith("ATOM"))
            n_residues = len({l[22:26].strip() for l in pdb_text.splitlines() if l.startswith("ATOM")})
            plddt_vals = _extract_plddt_from_pdb(pdb_text)
            has_plddt  = len(plddt_vals) > 0 and max(plddt_vals) > 1.0

            c1, c2, c3 = st.columns(3)
            c1.metric("ATOM records", n_atoms)
            c2.metric("Residues",     n_residues)
            c3.metric("File size",    f"{max(1, len(pdb_text) // 1024)} KB")

            st.markdown("### 🧬 3D Structure Viewer")
            try:
                st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)
            except Exception as e:
                st.warning(f"3D viewer could not render: {e}")

            rmsd_val = ca_rmsd(pdb_text)
            if rmsd_val is not None:
                st.success(f"Cα RMSD: **{rmsd_val:.3f} Å**")

            render_structural_analysis(
                pdb_text, prefix="pdb_",
                plddt_vals=plddt_vals if has_plddt else None,
            )
        else:
            st.error("Uploaded PDB file is empty or could not be decoded.")


# ==========================================================
# SECTION 19 - MODEL & DATASET ANALYTICS
# ==========================================================

if st.session_state.show_analytics:

    st.markdown("---")

    with st.expander("📊 Model Performance & Dataset Analytics", expanded=False):

        st.markdown("### 📈 Model Performance")
        mc = st.columns(3)
        mc[0].metric("Taste Accuracy",      f"{metrics['Taste accuracy']*100:.1f}%")
        mc[0].metric("Taste F1",            f"{metrics['Taste F1']:.3f}")
        mc[1].metric("Solubility Accuracy", f"{metrics['Solubility accuracy']*100:.1f}%")
        mc[1].metric("Solubility F1",       f"{metrics['Solubility F1']:.3f}")
        mc[2].metric("Docking R²",          f"{metrics['Docking R2']:.3f}")
        mc[2].metric("Docking RMSE",        f"{metrics['Docking RMSE']:.3f} kcal/mol")

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Distributions")
        fig_dist = plot_distributions(df_all)
        save_fig(fig_dist, "distributions.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        show_caption(caption_distributions(df_all))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 PCA: Peptide Feature Space")
        fig_pca, pca_model = plot_pca(
            X_all, le_taste.transform(df_all["taste"]), le_taste.classes_,
            title="PCA — Peptide Feature Space (coloured by taste class)",
        )
        save_fig(fig_pca, "pca_overall.png")
        st.pyplot(fig_pca)
        plt.close(fig_pca)
        show_caption(caption_pca(pca_model, le_taste.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Confusion Matrix — Taste")
        taste_preds  = taste_model.predict(X_test)
        fig_cm_taste = plot_confusion(yt_test, taste_preds, le_taste.classes_,
                                      title="Taste Confusion Matrix", cmap="Blues")
        save_fig(fig_cm_taste, "confusion_taste.png")
        st.pyplot(fig_cm_taste)
        plt.close(fig_cm_taste)
        show_caption(caption_confusion_taste(yt_test, taste_preds, le_taste.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Confusion Matrix — Solubility")
        sol_preds  = sol_model.predict(X_test)
        fig_cm_sol = plot_confusion(ys_test, sol_preds, le_sol.classes_,
                                    title="Solubility Confusion Matrix", cmap="Greens")
        save_fig(fig_cm_sol, "confusion_solubility.png")
        st.pyplot(fig_cm_sol)
        plt.close(fig_cm_sol)
        show_caption(caption_confusion_sol(ys_test, sol_preds, le_sol.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Feature Importance — Taste Model")
        fig_imp = plot_feature_importance(taste_model, X_all.columns, top_n=20)
        save_fig(fig_imp, "feature_importance_taste.png")
        st.pyplot(fig_imp)
        plt.close(fig_imp)
        show_caption(caption_feature_importance(taste_model, X_all.columns, top_n=20))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Docking Score: True vs Predicted")
        dock_preds = dock_model.predict(X_test)
        fig_dock   = plot_docking(yd_test, dock_preds)
        save_fig(fig_dock, "docking_scatter.png")
        st.pyplot(fig_dock)
        plt.close(fig_dock)
        show_caption(caption_docking(yd_test, dock_preds))


# ==========================================================
# SECTION 20 - PDF DOWNLOAD
# ==========================================================

if st.session_state.show_analytics and len(st.session_state.pdf_figures) > 0:
    st.markdown("## 📄 Download Complete PDF Report")
    pdf_path = generate_pdf(
        metrics, st.session_state.last_prediction, st.session_state.pdf_figures)
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            st.download_button(
                "📥 Download Full Analytics PDF", f,
                file_name="PepTastePredictor_Full_Report.pdf",
                mime="application/pdf",
            )


# ==========================================================
# SECTION 21 - FOOTER
# ==========================================================

st.markdown(f"""
<div class="footer">
&copy; {date.today().year} &nbsp; <b>PepTastePredictor</b><br>
AI + Structural Bioinformatics platform for peptide analysis<br>
External Structure Prediction via ESM Atlas &amp; AlphaFold Server &nbsp;|&nbsp;
For academic, educational, and research use
</div>
""", unsafe_allow_html=True)
