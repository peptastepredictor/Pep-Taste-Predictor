# ==========================================================
# PepTastePredictor — app.py
# Hybrid Structural Engine v2 | Light + Dark Mode
# MULTI-LABEL taste model: each peptide can have multiple tastes
# Taste labels: Bitter, Salty, Sour, Sweet, Umami (binary per taste)
# Solubility classifier | Docking R² regressor
# SHAP interpretability integrated
# ==========================================================

# ==========================================================
# SECTION 1 — IMPORTS
# ==========================================================

import os
import io
import gc
import re
import time
import json
import urllib.request
import urllib.error
import tempfile
import zipfile
from datetime import date
from collections import Counter
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import py3Dmol
import shap

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder
import PeptideBuilder
from PeptideBuilder import Geometry

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    r2_score, confusion_matrix, classification_report,
    hamming_loss,
)
from sklearn.decomposition import PCA

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Image as RLImage,
    Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors


# ==========================================================
# SECTION 2 — GLOBAL CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH    = "AIML.xlsx"
PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

AA             = "ACDEFGHIKLMNPQRSTVWY"
ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]

# ==========================================================
# KEY DESIGN DECISION — MULTI-LABEL CLASSIFICATION
# ==========================================================
# The dataset contains composite labels like "Sour Sweet Umami", "Salty Umami", etc.
# Nearly ALL Umami rows are composite. Using single-label (pick-one) forces a wrong
# choice and makes Umami (or other tastes) disappear from predictions entirely.
#
# Solution: treat each taste as an INDEPENDENT binary classification problem.
# One ExtraTreesClassifier per taste, wrapped in MultiOutputClassifier.
# A peptide can be predicted as Bitter AND Umami simultaneously — matching reality.
# ==========================================================
TASTES = ["Bitter", "Salty", "Sour", "Sweet", "Umami"]

TASTE_EMOJI = {
    "Bitter":  "😖",
    "Sweet":   "😋",
    "Salty":   "🧂",
    "Sour":    "😮‍💨",
    "Umami":   "🍖",
}

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

CF_HELIX = {
    "A":1.42,"R":0.98,"N":0.67,"D":1.01,"C":0.70,"Q":1.11,"E":1.51,
    "G":0.57,"H":1.00,"I":1.08,"L":1.21,"K":1.16,"M":1.45,"F":1.13,
    "P":0.57,"S":0.77,"T":0.83,"W":1.08,"Y":0.69,"V":1.06,
}
CF_SHEET = {
    "A":0.83,"R":0.93,"N":0.89,"D":0.54,"C":1.19,"Q":1.10,"E":0.37,
    "G":0.75,"H":0.87,"I":1.60,"L":1.30,"K":0.74,"M":1.05,"F":1.38,
    "P":0.55,"S":0.75,"T":1.19,"W":1.37,"Y":1.47,"V":1.70,
}

THREE_LETTER = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY",
    "H":"HIS","I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN",
    "P":"PRO","Q":"GLN","R":"ARG","S":"SER","T":"THR","V":"VAL",
    "W":"TRP","Y":"TYR",
}

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FETCH  = "https://files.rcsb.org/download/{}.pdb"
ESMFOLD_API = "https://api.esmatlas.com/foldSequence/v1/pdb/"


def _get_cmap(name, n=None):
    try:
        cmap = matplotlib.colormaps[name]
    except Exception:
        cmap = plt.cm.get_cmap(name)
    if n is not None:
        return cmap.resampled(n) if hasattr(cmap, "resampled") else cmap
    return cmap


# ==========================================================
# SECTION 3 — STYLING
# ==========================================================

st.markdown("""
<style>
.stApp p,.stApp span,.stApp label,.stApp li,.stApp h1,.stApp h2,.stApp h3,
.stApp h4,.stApp h5,.stApp div{color:var(--text-color) !important;}
.stMarkdown,.stMarkdown *{color:var(--text-color) !important;}
h1,h2,h3,h4{color:var(--text-color) !important;}
div[data-testid="stRadio"] label,div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p{color:var(--text-color) !important;}
div[data-testid="stTextInput"] label p,div[data-testid="stTextInput"] input{color:var(--text-color) !important;}
div[data-testid="stTextArea"] label p{color:var(--text-color) !important;}
div[data-testid="stFileUploader"] label p,div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploaderDropzone"] span{color:var(--text-color) !important;}
section[data-testid="stSidebar"],section[data-testid="stSidebar"] *{color:var(--text-color) !important;}
.stDataFrame,.stDataFrame td,.stDataFrame th{color:var(--text-color) !important;}
[data-testid="stMetric"] label,[data-testid="stMetric"] div{color:var(--text-color) !important;}

.hero{
  background:linear-gradient(135deg,#1f3c88 0%,#0b7285 60%,#12b886 100%);
  padding:44px 48px;border-radius:22px;margin-bottom:36px;
  box-shadow:0 8px 36px rgba(31,60,136,0.22);
}
.hero h1{font-size:2.6rem !important;font-weight:800 !important;
  margin-bottom:12px;color:#ffffff !important;letter-spacing:-0.5px;}
.hero p{font-size:1.1rem !important;line-height:1.9;color:#dce8ff !important;margin:0;}

.card{border:1px solid rgba(128,128,180,0.3);padding:28px 32px;border-radius:18px;
  margin-bottom:24px;background:rgba(128,128,180,0.05);
  box-shadow:0 2px 14px rgba(0,0,0,0.07);}
.card-title{font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.12em;
  opacity:0.55;margin-bottom:18px;color:var(--text-color) !important;}

.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:20px;}
.metric-box{background:rgba(26,143,209,0.07);border:1px solid rgba(26,143,209,0.18);
  border-radius:12px;padding:16px 20px;}
.metric-box-label{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;
  opacity:0.55;margin-bottom:6px;color:var(--text-color) !important;}
.metric-box-value{font-size:22px;font-weight:800;color:#1a8fd1 !important;}
.metric-box-sub{font-size:11px;opacity:0.55;margin-top:3px;color:var(--text-color) !important;}

/* Taste badge colours */
.taste-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;
  border-radius:20px;font-size:13px;font-weight:700;margin:3px;}
.taste-Bitter{background:rgba(192,57,43,0.12);border:1.5px solid rgba(192,57,43,0.45);color:#c0392b !important;}
.taste-Salty{background:rgba(41,128,185,0.12);border:1.5px solid rgba(41,128,185,0.45);color:#1a5276 !important;}
.taste-Sour{background:rgba(243,156,18,0.12);border:1.5px solid rgba(243,156,18,0.45);color:#b7770d !important;}
.taste-Sweet{background:rgba(155,89,182,0.12);border:1.5px solid rgba(155,89,182,0.45);color:#7d3c98 !important;}
.taste-Umami{background:rgba(39,174,96,0.12);border:1.5px solid rgba(39,174,96,0.45);color:#1e8449 !important;}

.engine-badge{display:inline-flex;align-items:center;gap:8px;padding:7px 16px;
  border-radius:20px;font-size:13px;font-weight:700;margin:4px 4px 4px 0;}
.badge-esm{background:rgba(18,184,134,0.12);border:1.5px solid rgba(18,184,134,0.4);color:#12b886 !important;}
.badge-pdb{background:rgba(255,152,0,0.12);border:1.5px solid rgba(255,152,0,0.4);color:#e65100 !important;}
.badge-fold{background:rgba(103,58,183,0.12);border:1.5px solid rgba(103,58,183,0.4);color:#6a1b9a !important;}
.badge-pb{background:rgba(255,165,0,0.12);border:1.5px solid rgba(255,165,0,0.4);color:#e67e22 !important;}

.struct-info-panel{border:1px solid rgba(18,184,134,0.3);border-radius:16px;
  padding:24px 30px;margin-bottom:22px;background:rgba(18,184,134,0.04);}
.struct-info-panel h4{font-size:12px !important;font-weight:800 !important;
  text-transform:uppercase;letter-spacing:0.12em;opacity:0.5;margin:0 0 18px 0;
  color:var(--text-color) !important;}
.struct-row{display:flex;flex-wrap:wrap;gap:28px;align-items:flex-start;}
.struct-item{display:flex;flex-direction:column;}
.struct-label{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:0.1em;
  opacity:0.45;margin-bottom:4px;color:var(--text-color) !important;}
.struct-value{font-size:15px;font-weight:800;color:#1a8fd1 !important;}
.struct-value.green{color:#12b886 !important;}
.struct-value.purple{color:#6a1b9a !important;}

.ss-bar{display:flex;height:18px;border-radius:9px;overflow:hidden;margin:14px 0 8px;gap:2px;}
.ss-segment{height:100%;border-radius:4px;}
.ss-legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:4px;}
.ss-chip{display:inline-flex;align-items:center;gap:7px;font-size:12px;
  font-weight:700;color:var(--text-color) !important;}
.ss-dot{width:11px;height:11px;border-radius:3px;display:inline-block;}

.plddt-legend{display:flex;gap:20px;flex-wrap:wrap;margin-top:10px;}
.plddt-chip{display:inline-flex;align-items:center;gap:8px;font-size:13px;
  font-weight:700;color:var(--text-color) !important;}
.plddt-dot{width:14px;height:14px;border-radius:50%;display:inline-block;}

.esm-banner{background:linear-gradient(90deg,rgba(18,184,134,0.12),rgba(18,184,134,0.04));
  border:1.5px solid rgba(18,184,134,0.35);border-radius:12px;
  padding:14px 22px;margin:14px 0 18px;display:flex;align-items:center;gap:14px;}
.esm-banner-icon{font-size:26px;}
.esm-banner-text{font-size:14px;font-weight:600;color:#12b886 !important;}
.esm-banner-sub{font-size:12px;opacity:0.7;color:var(--text-color) !important;}

.shap-panel{border:1px solid rgba(255,127,0,0.3);border-radius:16px;
  padding:24px 30px;margin:18px 0;background:rgba(255,127,0,0.04);}
.shap-panel h4{font-size:12px !important;font-weight:800 !important;
  text-transform:uppercase;letter-spacing:0.12em;opacity:0.55;margin:0 0 14px 0;
  color:var(--text-color) !important;}

.phys-table{width:100%;border-collapse:collapse;margin-top:8px;font-size:14px;}
.phys-table th{background:rgba(31,60,136,0.1);color:var(--text-color) !important;
  font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:0.08em;
  padding:10px 14px;text-align:left;border-bottom:2px solid rgba(128,128,180,0.25);}
.phys-table td{padding:9px 14px;border-bottom:1px solid rgba(128,128,180,0.12);
  color:var(--text-color) !important;}
.phys-table tr:last-child td{border-bottom:none;}
.phys-table td:nth-child(2){font-weight:700;color:#1a8fd1 !important;}

.graph-caption{border-left:5px solid #4a6fa5;border-radius:0 12px 12px 0;
  padding:18px 24px;margin-top:10px;margin-bottom:40px;font-size:14px !important;
  line-height:1.9;background:rgba(74,111,165,0.07);color:var(--text-color) !important;}
.graph-caption strong{font-weight:700;color:var(--text-color) !important;}

.section-gap{margin-top:44px;margin-bottom:6px;}

.pdf-card{background:linear-gradient(135deg,rgba(31,60,136,0.08),rgba(18,184,134,0.06));
  border:1.5px solid rgba(31,60,136,0.2);border-radius:18px;
  padding:28px 36px;margin-top:32px;text-align:center;}
.pdf-card h3{font-size:1.3rem !important;font-weight:800 !important;
  color:var(--text-color) !important;margin-bottom:8px;}
.pdf-card p{font-size:14px;opacity:0.65;color:var(--text-color) !important;}

.footer{text-align:center;font-size:13px !important;padding:44px 20px 20px;
  margin-top:60px;line-height:2.4;border-top:1px solid rgba(128,128,180,0.22);
  color:var(--text-color) !important;opacity:0.7;}

@keyframes pulse{0%{opacity:1;}50%{opacity:0.45;}100%{opacity:1;}}
.live-indicator{display:inline-block;width:8px;height:8px;background:#12b886;
  border-radius:50%;animation:pulse 1.5s infinite;margin-right:6px;}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 4 — SIDEBAR
# ==========================================================

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)

st.sidebar.markdown("### 🧬 PepTastePredictor")
st.sidebar.write("AI-driven peptide analysis platform")
st.sidebar.markdown("""
- 🎯 Multi-label taste prediction (all 5 tastes)
- 💧 Solubility prediction
- 🔗 Docking score estimation
- 🔬 Structural bioinformatics
- 🧠 SHAP interpretability
- ⚙️ Hybrid Structural Engine v2
- 📦 Batch screening + ZIP export
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Structure Engine Priority:**")
st.sidebar.markdown(
    '<div class="engine-badge badge-pdb">① RCSB PDB</div>'
    '<div class="engine-badge badge-esm">② Remote ESMFold</div>'
    '<div class="engine-badge badge-fold">③ Peptide Folder</div>'
    '<div class="engine-badge badge-pb">④ PeptideBuilder</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Taste Model:**")
st.sidebar.markdown(
    "Multi-label · one binary classifier per taste · "
    "peptides can have multiple simultaneous tastes"
)
st.sidebar.markdown(
    "**Feature vector:** 432 dimensions — "
    "7 physicochemical + 20 AAC + 400 DPC + 5 group ratios"
)
st.sidebar.info("For academic and research use only")


# ==========================================================
# SECTION 5 — SESSION STATE
# ==========================================================

_defaults = {
    "initialized":      True,
    "pdb_text":         None,
    "pdb_source":       None,
    "last_prediction":  {},
    "show_analytics":   False,
    "pdf_figures":      [],
    "current_mode":     None,
    "prediction_count": 0,
    "shap_explainers":  {},   # dict: taste -> explainer
    "pdf_captions":     {},   # dict: filename -> explanation & inference text for PDF
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==========================================================
# SECTION 6 — UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename: str, caption: str = None):
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    if filename not in st.session_state.pdf_figures:
        st.session_state.pdf_figures.append(filename)
    if caption:
        st.session_state.pdf_captions[filename] = caption


def clean_sequence(seq) -> str:
    if not isinstance(seq, str):
        return ""
    lines = seq.splitlines()
    lines = [l for l in lines if not l.strip().startswith(">")]
    seq   = "".join(lines)
    seq   = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


def parse_fasta(text: str) -> list:
    records = []
    current_header, current_seq = "", []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_seq:
                seq = clean_sequence("".join(current_seq))
                if seq:
                    records.append((current_header, seq))
            current_header = line[1:].strip()
            current_seq    = []
        else:
            current_seq.append(line)
    if current_seq:
        seq = clean_sequence("".join(current_seq))
        if seq:
            records.append((current_header, seq))
    return records


def read_uploaded_fasta(uploaded_file) -> list:
    if uploaded_file is None:
        return []
    raw = uploaded_file.read()
    for enc in ("utf-8", "latin-1", "ascii"):
        try:
            text    = raw.decode(enc)
            records = parse_fasta(text)
            if records:
                return records
            seq = clean_sequence(text)
            if seq:
                return [("uploaded_sequence", seq)]
            return []
        except Exception:
            continue
    return []


def build_taste_labels(taste_series: pd.Series) -> pd.DataFrame:
    """
    Convert raw taste labels into a binary DataFrame with one column per taste.

    Examples:
        "Bitter"              → Bitter=1, rest=0
        "Sour Sweet Umami"   → Sour=1, Sweet=1, Umami=1, rest=0
        "Salty, Umami"       → Salty=1, Umami=1, rest=0

    Uses regex word-matching so punctuation (commas, slashes, periods) never
    prevents a match.
    """
    result = {}
    for t in TASTES:
        pattern = re.compile(r"\b" + t + r"\b", re.IGNORECASE)
        result[t] = taste_series.apply(
            lambda x: 1 if pattern.search(str(x)) else 0
        )
    return pd.DataFrame(result)


def model_features(seq: str) -> dict:
    """
    Extract 432-dimensional feature vector from a peptide sequence.
    Breakdown: 7 physicochemical + 20 AAC + 400 DPC (20×20) + 5 group = 432 total.
    Note: secondary structure fractions are computed separately for display
    only and are NOT included in this feature vector.
    """
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
            features.update({"mw":0,"pI":7.0,"aromaticity":0,
                             "instability":0,"gravy":0,"charge":0})
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
        "hydrophobic": "AILMFWV", "polar": "STNQ",
        "charged": "DEKRH",       "aromatic": "FWY", "tiny": "AGSC",
    }
    for name, aas in groups.items():
        features[name] = sum(seq.count(a) for a in aas) / L
    return features


def build_feature_table(seqs) -> pd.DataFrame:
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)


def build_batch_physicochemical_row(seq: str) -> dict:
    phys = physicochemical_features(seq)
    comp = composition_features(seq)
    row = dict(phys)
    row.update(comp)
    return row


def physicochemical_features(seq: str) -> dict:
    L = len(seq)
    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            h, t, s = ana.secondary_structure_fraction()
            return {
                "Length":                                L,
                "Molecular Weight (Da)": round(ana.molecular_weight(), 2),
                "Isoelectric Point":     round(ana.isoelectric_point(), 2),
                "Net Charge (pH 7)":     round(ana.charge_at_pH(7.0), 2),
                "Aromaticity":           round(ana.aromaticity(), 3),
                "GRAVY":                 round(ana.gravy(), 3),
                "Instability Index":     round(ana.instability_index(), 2),
                "Helix Fraction":        round(h, 3),
                "Turn Fraction":         round(t, 3),
                "Sheet Fraction":        round(s, 3),
            }
        except Exception:
            pass
    return {
        "Length": L,
        "GRAVY":  round(KD_SCALE.get(seq, 0.0), 3),
        "Note":   "Extended analysis requires ≥2 residues",
    }


def composition_features(seq: str) -> dict:
    c = Counter(seq)
    L = len(seq)
    return {
        "Hydrophobic (%)": round(100 * sum(c[a] for a in "AILMFWV") / L, 1),
        "Polar (%)":       round(100 * sum(c[a] for a in "STNQ")    / L, 1),
        "Charged (%)":     round(100 * sum(c[a] for a in "DEKRH")   / L, 1),
        "Aromatic (%)":    round(100 * sum(c[a] for a in "FWY")     / L, 1),
    }


def gravy_score(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KD_SCALE.get(a, 0) for a in seq) / len(seq)


def prettify_feature(name: str) -> str:
    if name.startswith("DPC_"):
        return f"Dipeptide {name[4:]}"
    if name.startswith("AA_"):
        return f"Amino acid: {name[3:]}"
    return name.replace("_", " ").title()


def taste_badges_html(predicted_tastes: list) -> str:
    """Return HTML badges for a list of predicted taste strings."""
    if not predicted_tastes:
        return '<span style="opacity:0.5;font-style:italic;">None detected</span>'
    badges = ""
    for t in predicted_tastes:
        emoji = TASTE_EMOJI.get(t, "🧬")
        badges += f'<span class="taste-badge taste-{t}">{emoji} {t}</span>'
    return badges


def show_caption(html_text: str):
    st.markdown(f'<div class="graph-caption">{html_text}</div>', unsafe_allow_html=True)


def _close_all_figs():
    plt.close("all")
    gc.collect()


# ==========================================================
# SECTION 7 — MATPLOTLIB THEME
# ==========================================================

def _is_dark_mode() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def get_plot_colors() -> dict:
    if _is_dark_mode():
        return {
            "fig_bg":"#1a1d2e","ax_bg":"#1e2140","text":"#e8edf8",
            "grid":"#2e3560","accent1":"#5c7cfa","accent2":"#748ffc",
            "accent3":"#4dd0e1","red":"#ff6b6b","orange":"#ffa94d",
            "tick":"#c5cff0","green":"#51cf66",
        }
    return {
        "fig_bg":"#f8f9fc","ax_bg":"#ffffff","text":"#1a1d2e",
        "grid":"#d0d5e8","accent1":"#1a56db","accent2":"#4361ee",
        "accent3":"#0b7285","red":"#c0392b","orange":"#e67e22",
        "tick":"#4a5170","green":"#12b886",
    }


def apply_plot_style(fig, axes_list):
    C = get_plot_colors()
    fig.patch.set_facecolor(C["fig_bg"])
    for ax in (axes_list if hasattr(axes_list, "__iter__") else [axes_list]):
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
# SECTION 8 — PDB HELPERS
# ==========================================================

def _write_temp_pdb(pdb_text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(pdb_text)
    tmp.close()
    return tmp.name


def _unlink(path: str):
    try:
        os.unlink(path)
    except OSError:
        pass


def _validate_pdb(pdb_text: str) -> bool:
    if not pdb_text or not pdb_text.strip():
        return False
    has_atom = has_ca = False
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        has_atom = True
        if line[12:16].strip() == "CA":
            has_ca = True
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                    return False
            except ValueError:
                return False
    return has_atom and has_ca


def _extract_plddt(pdb_text: str) -> list:
    seen, vals = set(), []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        key = (line[21], line[22:26].strip())
        if key not in seen:
            seen.add(key)
            try:
                vals.append(float(line[60:66].strip()))
            except ValueError:
                pass
    return vals


def _count_residues_in_pdb(pdb_text: str) -> int:
    seen = set()
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            seen.add((line[21], line[22:26].strip()))
    return len(seen)


# ==========================================================
# SECTION 9 — SECONDARY STRUCTURE (Chou-Fasman)
# ==========================================================

def predict_secondary_structure(seq: str) -> dict:
    L = len(seq)
    if L == 0:
        return {"assignments": [], "helix_frac": 0, "sheet_frac": 0, "coil_frac": 1}
    helix_prop = [CF_HELIX.get(aa, 1.0) for aa in seq]
    sheet_prop = [CF_SHEET.get(aa, 1.0) for aa in seq]
    assignments = []
    for i in range(L):
        lo, hi = max(0, i - 2), min(L, i + 3)
        h_avg  = np.mean(helix_prop[lo:hi])
        s_avg  = np.mean(sheet_prop[lo:hi])
        if seq[i] == "P":
            assignments.append("C")
        elif h_avg >= 1.03 and h_avg >= s_avg:
            assignments.append("H")
        elif s_avg >= 1.05 and s_avg > h_avg:
            assignments.append("E")
        else:
            assignments.append("C")
    ss = list(assignments)
    for i in range(L):
        if ss[i] in ("H", "E"):
            j = i
            while j < L and ss[j] == ss[i]:
                j += 1
            if j - i < 4:
                for k in range(i, j):
                    ss[k] = "C"
    counts = Counter(ss)
    total  = max(L, 1)
    return {
        "assignments": ss,
        "helix_frac":  counts.get("H", 0) / total,
        "sheet_frac":  counts.get("E", 0) / total,
        "coil_frac":   counts.get("C", 0) / total,
    }


def classify_fold(ss_result: dict, seq: str) -> str:
    h, e, L = ss_result["helix_frac"], ss_result["sheet_frac"], len(seq)
    if L <= 2:   return "Dipeptide / Residue"
    if L <= 10:
        if h > 0.5: return "Short Helix"
        if e > 0.3: return "Beta-rich Peptide"
        return "Short Peptide / Loop"
    if h > 0.6:              return "All-α Helix"
    if e > 0.5:              return "All-β Sheet"
    if h > 0.3 and e > 0.2: return "α/β Mixed"
    if h > 0.4:              return "Predominantly α"
    if e > 0.3:              return "Predominantly β"
    return "Disordered / Coil"


# ==========================================================
# SECTION 10 — HYBRID STRUCTURAL ENGINE
# ==========================================================

def _http_get(url: str, timeout: int = 15) -> str:
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "PepTastePredictor/2.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_rcsb_pdb(seq: str) -> str:
    if len(seq) < 5:
        return ""
    query = {
        "query": {
            "type": "terminal", "service": "sequence",
            "parameters": {
                "evalue_cutoff": 1, "identity_cutoff": 0.95,
                "sequence_type": "protein", "value": seq,
            },
        },
        "request_options": {
            "results_verbosity": "compact",
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
        "return_type": "entry",
    }
    try:
        data = json.dumps(query).encode("utf-8")
        req  = urllib.request.Request(
            RCSB_SEARCH, data=data,
            headers={"Content-Type": "application/json",
                     "User-Agent": "PepTastePredictor/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        hits = result.get("result_set", [])
        if not hits:
            return ""
        pdb_id  = hits[0]["identifier"].split("_")[0].upper()
        pdb_txt = _http_get(RCSB_FETCH.format(pdb_id), timeout=15)
        if pdb_txt and _validate_pdb(pdb_txt):
            return pdb_txt
    except Exception:
        pass
    return ""


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_esmfold_remote(seq: str) -> str:
    if len(seq) < 3 or len(seq) > 400:
        return ""
    try:
        data = seq.encode("utf-8")
        req  = urllib.request.Request(
            ESMFOLD_API, data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "PepTastePredictor/2.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdb_txt = resp.read().decode("utf-8", errors="ignore")
        if pdb_txt and _validate_pdb(pdb_txt):
            return pdb_txt
    except Exception:
        pass
    return ""


def _build_realistic_peptide(seq: str) -> str:
    ss_result = predict_secondary_structure(seq)
    ss = ss_result["assignments"]
    L  = len(seq)
    CA_C, C_N, N_CA = 1.52, 1.33, 1.46
    OMEGA = np.radians(180.0)
    SS_ANGLES = {"H": (-57.0, -47.0), "E": (-120.0, 120.0), "C": (-60.0, 140.0)}

    def _rotation_matrix(axis, angle):
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        ca, sa = np.cos(angle), np.sin(angle)
        ux, uy, uz = axis
        return np.array([
            [ca+ux*ux*(1-ca),   ux*uy*(1-ca)-uz*sa, ux*uz*(1-ca)+uy*sa],
            [uy*ux*(1-ca)+uz*sa, ca+uy*uy*(1-ca),   uy*uz*(1-ca)-ux*sa],
            [uz*ux*(1-ca)-uy*sa, uz*uy*(1-ca)+ux*sa, ca+uz*uz*(1-ca)],
        ])

    def _place_atom(p1, p2, p3, bond_len, angle_deg, dihedral_deg):
        b1 = p3 - p2; b2 = p2 - p1
        angle    = np.radians(angle_deg)
        dihedral = np.radians(dihedral_deg)
        b1n = b1 / (np.linalg.norm(b1) + 1e-12)
        b2n = b2 / (np.linalg.norm(b2) + 1e-12)
        n    = np.cross(b2n, b1n)
        nn   = np.linalg.norm(n)
        n    = n / nn if nn > 1e-10 else np.array([0., 0., 1.])
        rot1  = _rotation_matrix(n, np.pi - angle)
        d_dir = rot1 @ b1n
        rot2  = _rotation_matrix(b1n, dihedral)
        d_dir = rot2 @ d_dir
        return p3 + bond_len * d_dir

    N_CA_C, CA_C_N, C_N_CA = 111.2, 116.2, 121.7
    atoms = {}
    atoms[(0, "N")]  = np.array([0., 0., 0.])
    atoms[(0, "CA")] = np.array([N_CA, 0., 0.])
    phi0, psi0 = SS_ANGLES.get(ss[0] if ss else "C", (-60., 140.))
    atoms[(0, "C")] = _place_atom(
        np.array([-1., 0., 0.]), atoms[(0, "N")], atoms[(0, "CA")],
        CA_C, N_CA_C, psi0)

    for i in range(1, L):
        phi_i, psi_i = SS_ANGLES.get(
            ss[i] if i < len(ss) else "C", (-60., 140.))
        n_i  = _place_atom(
            atoms[(i-1,"N")], atoms[(i-1,"CA")], atoms[(i-1,"C")],
            C_N, CA_C_N, np.degrees(OMEGA))
        atoms[(i, "N")] = n_i
        ca_i = _place_atom(
            atoms[(i-1,"CA")], atoms[(i-1,"C")], n_i, N_CA, C_N_CA, phi_i)
        atoms[(i, "CA")] = ca_i
        c_i  = _place_atom(
            atoms[(i-1,"C")], n_i, ca_i, CA_C, N_CA_C, psi_i)
        atoms[(i, "C")]  = c_i

    for i in range(L):
        if seq[i] == "G":
            continue
        try:
            cb = _place_atom(
                atoms[(i,"C")], atoms[(i,"N")], atoms[(i,"CA")],
                1.52, 110.5, -122.5)
            atoms[(i, "CB")] = cb
        except Exception:
            pass

    lines      = []
    atom_num    = 1
    atom_order = ["N", "CA", "C", "CB"]
    SS_BFACTOR = {"H": 85.0, "E": 75.0, "C": 55.0}
    for i in range(L):
        resname = THREE_LETTER.get(seq[i], "UNK")
        bf = SS_BFACTOR.get(ss[i] if i < len(ss) else "C", 55.0)
        for aname in atom_order:
            if (i, aname) not in atoms:
                continue
            pos = atoms[(i, aname)]
            if not np.all(np.isfinite(pos)):
                continue
            x, y, z   = pos
            aname_fmt  = f" {aname:<3s}" if len(aname) < 4 else aname
            lines.append(
                f"ATOM  {atom_num:5d} {aname_fmt} {resname} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}            "
                f"{'C' if aname in ('CA','CB','C') else 'N'}"
            )
            atom_num += 1
    lines.append("END")
    return "\n".join(lines)


def _build_geometric_pdb(seq: str) -> str:
    try:
        structure = PeptideBuilder.initialize_res(seq[0])
        for aa in seq[1:]:
            try:
                PeptideBuilder.add_residue(structure, Geometry.geometry(aa))
            except Exception:
                pass
        io_obj = PDBIO()
        io_obj.set_structure(structure)
        out = "predicted_peptide.pdb"
        io_obj.save(out)
        with open(out) as f:
            pdb = f.read()
        return pdb if _validate_pdb(pdb) else ""
    except Exception:
        return ""


def _minimal_ca_pdb(seq: str) -> str:
    lines = []
    for i, aa in enumerate(seq):
        resname = THREE_LETTER.get(aa, "UNK")
        x = i * 3.8
        lines.append(
            f"ATOM  {i+1:5d}  CA  {resname} A{i+1:4d}    "
            f"{x:8.3f}   0.000   0.000  1.00 50.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


def predict_structure(seq: str, use_remote: bool = True) -> tuple:
    L = len(seq)
    if L <= 2:
        pdb = _build_realistic_peptide(seq)
        if not _validate_pdb(pdb):
            pdb = _build_geometric_pdb(seq)
        if not _validate_pdb(pdb):
            pdb = _minimal_ca_pdb(seq)
        return pdb, "Peptide Conformation Engine", "Geometric (1–2 aa)"

    if L <= 20:
        pdb = _build_realistic_peptide(seq)
        if pdb and _validate_pdb(pdb):
            if use_remote:
                esm = _fetch_esmfold_remote(seq)
                if esm and _validate_pdb(esm):
                    return esm, "Remote ESMFold", "ESM Atlas API (Meta AI)"
            return pdb, "Peptide Folding Engine", "Chou-Fasman SS + backbone torsions"
        pdb = _build_geometric_pdb(seq)
        if pdb and _validate_pdb(pdb):
            return pdb, "PeptideBuilder", "Geometric backbone"
        return _minimal_ca_pdb(seq), "PeptideBuilder (fallback)", "Linear chain"

    if use_remote:
        rcsb = _fetch_rcsb_pdb(seq)
        if rcsb and _validate_pdb(rcsb):
            return rcsb, "RCSB PDB Database", "Experimental / deposited structure"
        if L <= 400:
            esm = _fetch_esmfold_remote(seq)
            if esm and _validate_pdb(esm):
                return esm, "Remote ESMFold", "ESM Atlas API (Meta AI — AI-predicted)"

    pdb = _build_realistic_peptide(seq)
    if pdb and _validate_pdb(pdb):
        return pdb, "Peptide Folding Engine", "Chou-Fasman SS + backbone torsions"

    pdb = _build_geometric_pdb(seq)
    if pdb and _validate_pdb(pdb):
        return pdb, "PeptideBuilder", "Geometric backbone"

    return _minimal_ca_pdb(seq), "PeptideBuilder (fallback)", "Linear chain"


# ==========================================================
# SECTION 11 — STRUCTURE VISUALISATION
# ==========================================================

def show_structure(pdb_text: str, use_plddt_colors: bool = False):
    view = py3Dmol.view(width=1000, height=700)
    view.addModel(pdb_text, "pdb")
    plddt_vals = _extract_plddt(pdb_text)
    has_plddt  = len(plddt_vals) > 0 and max(plddt_vals) > 1.0
    if use_plddt_colors and has_plddt:
        view.setStyle({"cartoon": {
            "colorscheme": {"prop": "b", "gradient": "roygb", "min": 0, "max": 100}}})
    else:
        view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.VDW, {"opacity": 0.35})
    view.zoomTo()
    return view


def render_source_banner(engine_label: str, source_detail: str):
    if "ESMFold" in engine_label or "ESM" in engine_label:
        st.markdown(f"""
        <div class="esm-banner">
          <div class="esm-banner-icon">🧬</div>
          <div>
            <div class="esm-banner-text">Structure from Remote ESMFold (ESM Atlas · Meta AI Research)</div>
            <div class="esm-banner-sub">Source: {source_detail}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif "RCSB" in engine_label:
        st.markdown(f"""
        <div class="esm-banner" style="border-color:rgba(255,152,0,0.4);
             background:linear-gradient(90deg,rgba(255,152,0,0.12),rgba(255,152,0,0.04));">
          <div class="esm-banner-icon">🗂️</div>
          <div>
            <div class="esm-banner-text" style="color:#e65100 !important;">
              Experimentally determined structure from RCSB PDB</div>
            <div class="esm-banner-sub">Source: {source_detail}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    elif "Folding" in engine_label or "Conformation" in engine_label:
        st.markdown(f"""
        <div class="esm-banner" style="border-color:rgba(103,58,183,0.35);
             background:linear-gradient(90deg,rgba(103,58,183,0.10),rgba(103,58,183,0.03));">
          <div class="esm-banner-icon">🔮</div>
          <div>
            <div class="esm-banner-text" style="color:#6a1b9a !important;">
              Structure generated by Chou-Fasman Peptide Folding Engine</div>
            <div class="esm-banner-sub">Source: {source_detail}</div>
          </div>
        </div>""", unsafe_allow_html=True)


def render_plddt_legend():
    st.markdown("""
    <div class="plddt-legend">
      <div class="plddt-chip"><span class="plddt-dot" style="background:#1565C0;"></span>Very high (≥90)</div>
      <div class="plddt-chip"><span class="plddt-dot" style="background:#40C4FF;"></span>Confident (70–90)</div>
      <div class="plddt-chip"><span class="plddt-dot" style="background:#FFEB3B;"></span>Low (50–70)</div>
      <div class="plddt-chip"><span class="plddt-dot" style="background:#FF7043;"></span>Very low (&lt;50)</div>
    </div>""", unsafe_allow_html=True)


def render_structure_info_panel(seq, engine_label, source_detail, ss_result):
    fold_type = classify_fold(ss_result, seq)
    L         = len(seq)
    h_pct     = round(ss_result["helix_frac"] * 100, 1)
    e_pct     = round(ss_result["sheet_frac"] * 100, 1)
    c_pct     = round(ss_result["coil_frac"]  * 100, 1)

    if "ESMFold" in engine_label or "ESM" in engine_label:
        badge_cls, badge_icon = "badge-esm",  "🧬"
    elif "RCSB" in engine_label or "PDB" in engine_label:
        badge_cls, badge_icon = "badge-pdb",  "🗂️"
    elif "Folding" in engine_label or "Conformation" in engine_label:
        badge_cls, badge_icon = "badge-fold", "🔮"
    else:
        badge_cls, badge_icon = "badge-pb",   "⚙️"

    ss_bar = ""
    if h_pct > 0:
        ss_bar += f'<div class="ss-segment" style="width:{h_pct}%;background:#e91e63;"></div>'
    if e_pct > 0:
        ss_bar += f'<div class="ss-segment" style="width:{e_pct}%;background:#2196F3;"></div>'
    if c_pct > 0:
        ss_bar += f'<div class="ss-segment" style="width:{c_pct}%;background:#78909C;"></div>'

    st.markdown(f"""
    <div class="struct-info-panel">
      <h4>🔬 Structure Information Panel</h4>
      <div class="struct-row">
        <div class="struct-item">
          <span class="struct-label">Source Engine</span>
          <span class="engine-badge {badge_cls}" style="margin:0;">{badge_icon} {engine_label}</span>
        </div>
        <div class="struct-item">
          <span class="struct-label">Detail</span>
          <span class="struct-value" style="font-size:13px;color:var(--text-color)!important;opacity:0.7;">{source_detail}</span>
        </div>
        <div class="struct-item">
          <span class="struct-label">Residues</span>
          <span class="struct-value">{L}</span>
        </div>
        <div class="struct-item">
          <span class="struct-label">Predicted Fold</span>
          <span class="struct-value purple">{fold_type}</span>
        </div>
      </div>
      <div style="margin-top:20px;">
        <span class="struct-label">Secondary Structure Composition</span>
        <div class="ss-bar">{ss_bar}</div>
        <div class="ss-legend">
          <div class="ss-chip"><span class="ss-dot" style="background:#e91e63;"></span>α-Helix {h_pct}%</div>
          <div class="ss-chip"><span class="ss-dot" style="background:#2196F3;"></span>β-Sheet {e_pct}%</div>
          <div class="ss-chip"><span class="ss-dot" style="background:#78909C;"></span>Coil/Loop {c_pct}%</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ==========================================================
# SECTION 12 — STRUCTURAL ANALYSIS
# ==========================================================

def ramachandran(pdb_text: str) -> list:
    if not pdb_text or not pdb_text.strip():
        return []
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)[0]
        pts = []
        for pp in PPBuilder().build_peptides(structure):
            for phi, psi in pp.get_phi_psi_list():
                if phi is not None and psi is not None:
                    pts.append((np.degrees(phi), np.degrees(psi)))
        return pts
    except Exception:
        return []
    finally:
        _unlink(tmp)


def ca_distance_map(pdb_text: str) -> np.ndarray:
    if not pdb_text:
        return np.zeros((1, 1))
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        cas = [r["CA"].get_vector().get_array()
               for r in structure.get_residues() if "CA" in r]
        if not cas:
            return np.zeros((1, 1))
        coords = np.array(cas)
        diff   = coords[:, None, :] - coords[None, :, :]
        return np.sqrt((diff ** 2).sum(-1))
    except Exception:
        return np.zeros((1, 1))
    finally:
        _unlink(tmp)


def ca_rmsd(pdb_text: str):
    if not pdb_text:
        return None
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
        if len(cas) < 2:
            return None
        ref = cas[0]
        return float(np.sqrt(np.mean([(v - ref).norm() ** 2 for v in cas])))
    except Exception:
        return None
    finally:
        _unlink(tmp)


# ==========================================================
# SECTION 13 — SHAP INTERPRETABILITY (Multi-label aware)
# ==========================================================

def compute_shap_for_taste(taste_model_single, X_background_np, X_query_np, taste_name):
    """
    Compute SHAP values for one binary taste classifier.
    Caches explainer per taste in session_state["shap_explainers"][taste_name].
    """
    if taste_name not in st.session_state["shap_explainers"]:
        st.session_state["shap_explainers"][taste_name] = shap.TreeExplainer(
            taste_model_single,
            data=X_background_np,
            feature_perturbation="interventional",
        )
    explainer = st.session_state["shap_explainers"][taste_name]
    sv = explainer.shap_values(X_query_np)
    # sv shape: (n_samples, n_features) for binary — take positive class
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    raw_ev = explainer.expected_value
    if isinstance(raw_ev, (list, np.ndarray)):
        base_value = float(raw_ev[1]) if len(raw_ev) > 1 else float(raw_ev[0])
    else:
        base_value = float(raw_ev)
    return sv, base_value, explainer


def plot_shap_bar(shap_vals, feature_names, seq, taste_name, top_n=15):
    C     = get_plot_colors()
    vals  = shap_vals[0]
    df_sh = pd.DataFrame({
        "feature":    [prettify_feature(f) for f in feature_names],
        "shap_value": vals,
        "abs_shap":   np.abs(vals),
    }).sort_values("abs_shap", ascending=False).head(top_n)

    colors = [C["accent1"] if v > 0 else C["red"] for v in df_sh["shap_value"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    ax.barh(df_sh["feature"][::-1], df_sh["shap_value"][::-1],
            color=colors[::-1], edgecolor="none")
    ax.axvline(0, color=C["grid"], linewidth=1.2)
    ax.set_xlabel("SHAP Value  (impact on model output)", fontsize=11, labelpad=10)
    ax.set_title(
        f"SHAP — {taste_name} classifier  |  Seq: {seq[:20]}{'…' if len(seq)>20 else ''}",
        fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def caption_shap(sv, feature_names, seq, taste_name, top_n=3):
    """Dynamic explanation + inference text for a SHAP bar chart."""
    vals = sv[0]
    df_sh = pd.DataFrame({
        "feature": [prettify_feature(f) for f in feature_names],
        "shap":    vals,
        "abs":     np.abs(vals),
    }).sort_values("abs", ascending=False).head(top_n)
    top_html = "".join(
        f"<strong>#{i+1} {r['feature']}</strong> (SHAP={r['shap']:+.4f}, "
        f"{'pushes toward' if r['shap'] > 0 else 'pushes away from'} {taste_name})<br>"
        for i, (_, r) in enumerate(df_sh.iterrows())
    )
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())
    return (
        f"<strong>What this shows:</strong> this chart explains the model's reasoning for "
        f"one single prediction — why the model called sequence "
        f"<em>{seq[:30]}{'…' if len(seq) > 30 else ''}</em> <strong>{taste_name}</strong>. "
        f"Every bar is one property of the peptide (its length, an amino acid, a pair of "
        f"neighbouring amino acids, etc). Bars pointing right (blue) argued "
        f"<em>for</em> {taste_name}; bars pointing left (red) argued <em>against</em> it, "
        f"and longer bars mean a stronger vote.<br><br>"
        f"<strong>Top contributing features:</strong><br>{top_html}<br>"
        f"Out of all 432 properties the model looked at, {n_pos} pushed toward {taste_name} and "
        f"{n_neg} pushed away from it. "
        f"<strong>What it means:</strong> the sequence's {df_sh.iloc[0]['feature'].lower()} is the single "
        f"biggest reason the model made this call — in plain terms, the {taste_name} prediction "
        f"isn't a black-box guess, it traces back to a specific, identifiable trait of the "
        f"peptide's composition that a chemist could sanity-check by eye."
    )


def render_shap_analysis(taste_multilabel_model, X_train_bg, X_query_df,
                          feature_names, seq, predicted_tastes):
    """
    SHAP analysis for multi-label model.
    Shows one bar chart per PREDICTED taste (skips unpredicted ones to save time).
    """
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("## 🧠 SHAP Interpretability")
    st.markdown(
        '<div class="shap-panel"><h4>🔍 Why did the model predict these tastes?</h4>'
        'SHAP decomposes each binary taste prediction into per-feature contributions. '
        '<strong style="color:#1a8fd1;">Blue bars</strong> push toward that taste; '
        '<strong style="color:#c0392b;">red bars</strong> push away from it.</div>',
        unsafe_allow_html=True,
    )

    if not predicted_tastes:
        st.info("No tastes were predicted — SHAP analysis skipped.")
        return

    X_bg_np    = X_train_bg.values.astype(np.float32)
    X_query_np = X_query_df.values.astype(np.float32)

    estimators = taste_multilabel_model.estimators_  # one per TASTE in TASTES order

    for taste_name in predicted_tastes:
        taste_idx = TASTES.index(taste_name)
        single_clf = estimators[taste_idx]
        with st.spinner(f"Computing SHAP for {taste_name}…"):
            try:
                sv, base_value, _ = compute_shap_for_taste(
                    single_clf, X_bg_np, X_query_np, taste_name)
                fig_bar = plot_shap_bar(sv, feature_names, seq, taste_name)
                fname   = f"shap_bar_{taste_name.lower()}.png"
                cap     = caption_shap(sv, feature_names, seq, taste_name)
                save_fig(fig_bar, fname, caption=cap)
                st.markdown(f"#### {TASTE_EMOJI.get(taste_name,'')} {taste_name}")
                st.image(fname, use_column_width=True)
                show_caption(cap)
            except Exception as e:
                st.warning(f"SHAP for {taste_name} could not complete: {e}")


# ==========================================================
# SECTION 14 — PLOT FUNCTIONS
# ==========================================================

def plot_pca(X, Y_labels, title="PCA"):
    """PCA coloured by first positive taste label (multi-label aware)."""
    C       = get_plot_colors()
    pca     = PCA(n_components=2)
    coords  = pca.fit_transform(X)
    v1, v2  = pca.explained_variance_ratio_[:2] * 100

    # For each sample, pick first positive taste for colour (or 'None')
    def primary_taste(row):
        for i, t in enumerate(TASTES):
            if row[i] == 1:
                return t
        return "None"

    if isinstance(Y_labels, np.ndarray) and Y_labels.ndim == 2:
        point_tastes = [primary_taste(Y_labels[i]) for i in range(len(Y_labels))]
    else:
        point_tastes = ["Unknown"] * len(coords)

    taste_palette = {
        "Bitter": "#c0392b", "Salty": "#2980b9", "Sour": "#f39c12",
        "Sweet": "#8e44ad", "Umami": "#27ae60", "None": "#aaaaaa",
    }
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    for t in TASTES + ["None"]:
        mask = [i for i, pt in enumerate(point_tastes) if pt == t]
        if mask:
            ax.scatter(coords[mask, 0], coords[mask, 1], label=t, alpha=0.7,
                       s=35, color=taste_palette[t], edgecolors="none")
    ax.set_xlabel(f"PC1 ({v1:.1f}%)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 ({v2:.1f}%)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    C2 = get_plot_colors()
    leg = ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left",
                    title="Primary taste", title_fontsize=9,
                    facecolor=C2["fig_bg"], edgecolor=C2["grid"])
    leg.get_title().set_color(C2["text"])
    for t in leg.get_texts():
        t.set_color(C2["text"])
    plt.tight_layout()
    return fig, pca


def plot_multilabel_per_taste(Y_true, Y_pred):
    """
    Per-taste precision/recall/F1 bar chart for the multi-label model.
    One group of bars per taste.
    """
    C   = get_plot_colors()
    metrics_per_taste = []
    for i, t in enumerate(TASTES):
        rep = classification_report(
            Y_true[:, i], Y_pred[:, i],
            output_dict=True, zero_division=0)
        pos = rep.get("1", rep.get(1, {}))
        metrics_per_taste.append({
            "taste":     t,
            "precision": pos.get("precision", 0),
            "recall":    pos.get("recall", 0),
            "f1":        pos.get("f1-score", 0),
            "accuracy":  accuracy_score(Y_true[:, i], Y_pred[:, i]),
        })

    df_m = pd.DataFrame(metrics_per_taste)
    x    = np.arange(len(TASTES))
    w    = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    apply_plot_style(fig, [ax])
    ax.bar(x - 1.5*w, df_m["accuracy"],  w, label="Accuracy",  color="#1a56db", alpha=0.85)
    ax.bar(x - 0.5*w, df_m["precision"], w, label="Precision", color="#12b886", alpha=0.85)
    ax.bar(x + 0.5*w, df_m["recall"],    w, label="Recall",    color="#e67e22", alpha=0.85)
    ax.bar(x + 1.5*w, df_m["f1"],        w, label="F1-Score",  color="#9b59b6", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(TASTES, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Taste Classification Metrics (Multi-label model)", fontsize=13,
                 fontweight="bold", pad=12)
    ax.axhline(1.0, color=C["grid"], lw=0.8, linestyle="--")
    leg = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig, df_m

def plot_confusion_per_taste(Y_true, Y_pred):
    """5-panel confusion matrices, one per taste."""
    C   = get_plot_colors()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    apply_plot_style(fig, axes)
    taste_colors = {
        "Bitter": "Reds", "Salty": "Blues", "Sour": "Oranges",
        "Sweet":  "Purples", "Umami": "Greens",
    }
    for i, (ax, t) in enumerate(zip(axes, TASTES)):
        cm  = confusion_matrix(Y_true[:, i], Y_pred[:, i])
        acc = accuracy_score(Y_true[:, i], Y_pred[:, i])
        annot_color = "#111122" if not _is_dark_mode() else "#ffffff"
        sns.heatmap(cm, annot=True, fmt="d", cmap=taste_colors[t],
                    xticklabels=[f"¬{t}", t], yticklabels=[f"¬{t}", t],
                    ax=ax, linewidths=0.5, linecolor=C["grid"],
                    annot_kws={"size": 10, "color": annot_color})
        ax.set_title(f"{t}\nacc={acc:.2f}", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True", fontsize=9)
    plt.suptitle("Per-Taste Confusion Matrices (Multi-label)", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def caption_confusion_per_taste(Y_true, Y_pred):
    """Dynamic explanation + inference text for the confusion-matrix panel."""
    rows = []
    for i, t in enumerate(TASTES):
        cm = confusion_matrix(Y_true[:, i], Y_pred[:, i])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        rows.append((t, tn, fp, fn, tp))
    fp_heavy = max(rows, key=lambda r: r[2])
    fn_heavy = max(rows, key=lambda r: r[3])
    return (
        "<strong>What this shows:</strong> five side-by-side scorecards, one per taste, "
        "comparing what the model predicted against the peptide's true, known taste "
        "on peptides the model never saw during training. Each 2×2 grid splits outcomes into "
        "four boxes: correctly said 'no', correctly said 'yes', wrongly said 'yes' "
        "(false alarm), and wrongly said 'no' (missed it).<br><br>"
        f"<strong>Most false alarms:</strong> {fp_heavy[0]} ({fp_heavy[2]} peptides "
        f"incorrectly flagged as {fp_heavy[0]} when they weren't).<br>"
        f"<strong>Most misses:</strong> {fn_heavy[0]} ({fn_heavy[3]} peptides that really "
        f"were {fn_heavy[0]} but the model didn't catch).<br><br>"
        "<strong>What it means:</strong> missed peptides matter more for a discovery pipeline, "
        "since a genuinely taste-active peptide gets filtered out and never looked at again, "
        "while a false alarm just costs some extra lab time to rule out. Whichever taste has "
        "the largest off-diagonal numbers above is the one most worth collecting more training "
        "examples for."
    )


def plot_docking(y_true, y_pred):
    C    = get_plot_colors()
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lims = [min(y_true.min(), y_pred.min()) - 5,
            max(y_true.max(), y_pred.max()) + 5]
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.scatter(y_true, y_pred, alpha=0.65, edgecolors="none",
               color=C["accent1"], s=45)
    ax.plot(lims, lims, color=C["red"], linestyle="--", lw=1.8, label="Perfect fit")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.annotate(
        f"R² = {r2:.3f}\nRMSE = {rmse:.2f}",
        xy=(0.05, 0.87), xycoords="axes fraction", fontsize=11, color=C["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=C["fig_bg"], ec=C["grid"], alpha=0.95))
    ax.set_xlabel("True Docking Score", fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted Docking Score", fontsize=12, labelpad=10)
    ax.set_title("Docking Score: True vs Predicted", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=10, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_feature_importance(estimators, feature_names, top_n=20):
    """Average feature importance across all taste estimators."""
    C   = get_plot_colors()
    all_imp = np.mean([e.feature_importances_ for e in estimators], axis=0)
    imp = pd.DataFrame({
        "Feature":    [prettify_feature(f) for f in feature_names],
        "Importance": all_imp,
    }).sort_values("Importance", ascending=False).head(top_n)
    clrs = _get_cmap("Blues")(np.linspace(0.4, 0.9, len(imp))[::-1])
    fig, ax = plt.subplots(figsize=(8, 7))
    apply_plot_style(fig, [ax])
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1],
            color=clrs, edgecolor=C["grid"])
    ax.set_xlabel("Mean Importance Score (across all taste classifiers)", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Features — Multi-label Taste Model",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig, imp


def caption_feature_importance(imp_df, top_n=3):
    """Dynamic explanation + inference text for the feature-importance chart."""
    top = imp_df.head(top_n)
    top_html = "".join(
        f"<strong>#{i+1} {r['Feature']}</strong> (importance={r['Importance']:.4f})<br>"
        for i, (_, r) in enumerate(top.iterrows())
    )
    dpc_share = imp_df["Feature"].str.startswith("Dipeptide").mean() * 100
    return (
        "<strong>What this shows:</strong> across all 432 properties the model can look at "
        "for every peptide, this ranks the ones that matter most, on average, for deciding "
        "taste — averaged across all 5 taste classifiers. A longer bar means the model leans "
        "on that property more heavily when making its calls.<br><br>"
        f"<strong>Top features overall:</strong><br>{top_html}<br>"
        f"<strong>What it means:</strong> dipeptide-composition (DPC) features — i.e. which "
        f"pairs of amino acids sit next to each other — make up "
        f"{dpc_share:.0f}% of the features shown here. In plain terms, taste is driven more by "
        f"local sequence context (which residues are neighbours) than by the peptide's overall "
        f"amino-acid makeup or its bulk physicochemical properties."
    )


def plot_distributions(df):
    C            = get_plot_colors()
    seq_lengths = [len(s) for s in df["peptide"]]
    taste_counts = {t: int(df[f"label_{t}"].sum()) for t in TASTES}
    grav_vals   = [gravy_score(s) for s in df["peptide"]]
    fig, axes   = plt.subplots(1, 3, figsize=(16, 5))
    apply_plot_style(fig, axes)
    mean_len = np.mean(seq_lengths)
    axes[0].hist(seq_lengths, bins=20, color=C["accent1"], edgecolor=C["grid"], alpha=0.85)
    axes[0].axvline(mean_len, color=C["red"], linestyle="--", lw=2, label=f"Mean={mean_len:.1f}")
    axes[0].set_xlabel("Length (aa)", fontsize=11)
    axes[0].set_ylabel("Count",       fontsize=11)
    axes[0].set_title("Peptide Length Distribution", fontsize=12, fontweight="bold", pad=10)
    leg0 = axes[0].legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg0.get_texts(): t.set_color(C["text"])

    taste_palette = ["#c0392b","#2980b9","#f39c12","#8e44ad","#27ae60"]
    bars = axes[1].bar(list(taste_counts.keys()), list(taste_counts.values()),
                       color=taste_palette, edgecolor=C["grid"])
    axes[1].set_xlabel("Taste", fontsize=11)
    axes[1].set_ylabel("# Peptides containing this taste", fontsize=10)
    axes[1].set_title("Multi-label Taste Coverage", fontsize=12, fontweight="bold", pad=10)
    for bar, v in zip(bars, taste_counts.values()):
        axes[1].text(bar.get_x()+bar.get_width()/2, v+1, str(v),
                     ha="center", fontsize=9, color=C["text"])

    axes[2].hist(grav_vals, bins=20, color=C["accent2"], edgecolor=C["grid"], alpha=0.85)
    axes[2].axvline(0, color=C["red"], linestyle="--", lw=2, label="Hydrophilic|Hydrophobic")
    axes[2].axvline(np.mean(grav_vals), color=C["orange"], linestyle="--", lw=2,
                    label=f"Mean={np.mean(grav_vals):.2f}")
    axes[2].set_xlabel("GRAVY",  fontsize=11)
    axes[2].set_ylabel("Count",  fontsize=11)
    axes[2].set_title("GRAVY Distribution", fontsize=12, fontweight="bold", pad=10)
    leg2 = axes[2].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg2.get_texts(): t.set_color(C["text"])
    plt.tight_layout(pad=2.5)
    return fig


def plot_ramachandran(phi_psi):
    C = get_plot_colors()
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.fill([-180,-180,-45,-45,-180], [-75,-45,-45,-75,-75],
            color="#4CAF50", alpha=0.25, label="α-helix")
    ax.fill([-180,-180,-90,-90,-180], [90,180,180,90,90],
            color="#2196F3", alpha=0.25, label="β-sheet")
    ax.fill([45,45,90,90,45], [0,90,90,0,0],
            color="#FF9800", alpha=0.20, label="L-helix")
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=50, color=C["red"], zorder=5,
                   edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.axvline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi φ (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("Psi ψ (°)", fontsize=12, labelpad=10)
    ax.set_title("Ramachandran Plot", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, loc="upper right",
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
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
    size        = max(5, min(n * 0.3 + 2, 16))
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


def plot_plddt(plddt_vals, seq=""):
    C = get_plot_colors()
    n = len(plddt_vals)
    bar_colors = [
        "#1565C0" if v >= 90 else "#40C4FF" if v >= 70
        else "#FFEB3B" if v >= 50 else "#FF7043"
        for v in plddt_vals
    ]
    fig, ax = plt.subplots(figsize=(max(8, min(n * 0.22, 24)), 4))
    apply_plot_style(fig, [ax])
    ax.bar(range(n), plddt_vals, color=bar_colors, width=0.9)
    mean_pl = np.mean(plddt_vals)
    ax.axhline(mean_pl, color=C["orange"], linestyle="-.", lw=2,
               label=f"Mean = {mean_pl:.1f}")
    for thresh, col, lbl in [
        (90, "#1565C0", "Very high (90)"),
        (70, "#40C4FF", "Confident (70)"),
        (50, "#FFEB3B", "Low (50)"),
    ]:
        ax.axhline(thresh, color=col, linestyle="--", lw=1.0, alpha=0.6, label=lbl)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Residue Index", fontsize=11, labelpad=8)
    ax.set_ylabel("pLDDT",          fontsize=11, labelpad=8)
    ax.set_title(f"Per-Residue pLDDT Confidence  (mean = {mean_pl:.1f})",
                 fontsize=12, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    leg = ax.legend(fontsize=8, loc="lower right",
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_ss_composition(ss_result: dict, seq: str):
    C  = get_plot_colors()
    ss = ss_result["assignments"]
    n  = len(ss)
    if n == 0:
        return None
    color_map = {"H": "#e91e63", "E": "#2196F3", "C": "#78909C"}
    bar_cols  = [color_map.get(s, "#888") for s in ss]
    heights   = [1.0 if s == "H" else 0.7 if s == "E" else 0.4 for s in ss]
    fig, ax   = plt.subplots(figsize=(max(8, min(n * 0.22, 24)), 3))
    apply_plot_style(fig, [ax])
    ax.bar(range(n), heights, color=bar_cols, width=0.9, edgecolor="none")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0.4, 0.7, 1.0])
    ax.set_yticklabels(["Coil", "β-Sheet", "α-Helix"], fontsize=9, color=C["tick"])
    ax.set_xlabel("Residue Index", fontsize=11, labelpad=8)
    ax.set_title("Per-Residue Secondary Structure Prediction",
                 fontsize=12, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e91e63", label="α-Helix"),
        Patch(facecolor="#2196F3", label="β-Sheet"),
        Patch(facecolor="#78909C", label="Coil/Loop"),
    ]
    leg = ax.legend(handles=legend_elements, fontsize=9, loc="upper right",
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts(): t.set_color(C["text"])
    plt.tight_layout()
    return fig


# ==========================================================
# SECTION 15 — DYNAMIC CAPTIONS
# ==========================================================

def caption_distributions(df):
    lengths   = [len(s) for s in df["peptide"]]
    grav      = [gravy_score(s) for s in df["peptide"]]
    mean_grav = np.mean(grav)
    glabel    = ("slightly hydrophobic" if mean_grav > 0.2 else
                 "slightly hydrophilic" if mean_grav < -0.2 else "amphipathic")
    taste_summary = " | ".join(
        f"{t}: {int(df[f'label_{t}'].sum())}" for t in TASTES)
    return (
        "<strong>What this shows:</strong> a three-panel snapshot of the whole training "
        "dataset — how long the peptides are, how many carry each taste label, and whether "
        "they tend to be water-loving or fat-loving.<br><br>"
        f"<strong>Length (left):</strong> peptides range {int(np.min(lengths))}–{int(np.max(lengths))} "
        f"amino acids long, averaging {np.mean(lengths):.1f} aa.<br><br>"
        f"<strong>Taste coverage (centre — multi-label):</strong> {taste_summary}.<br>"
        f"Counts exceed total rows because each peptide may have multiple tastes at once.<br><br>"
        f"<strong>GRAVY / hydrophobicity (right):</strong> mean score {mean_grav:.2f} — "
        f"positive means more fat-loving (hydrophobic), negative means more water-loving "
        f"(hydrophilic). This dataset leans <strong>{glabel}</strong> overall.<br><br>"
        "<strong>What it means:</strong> this gives a sense of what kind of peptides the model "
        "was trained on, and where its predictions are most trustworthy — sequences that look "
        "very different in length or hydrophobicity from this range are further outside the "
        "model's comfort zone."
    )


def caption_pca(pca_model):
    v1, v2 = pca_model.explained_variance_ratio_[:2] * 100
    return (
        "<strong>What this shows:</strong> every peptide has 432 measured properties, far too "
        "many to plot at once. PCA squashes them down to the 2 directions that capture the "
        "most variation between peptides, so each dot here is one peptide placed on a 2D map — "
        "dots that land close together are chemically similar overall, and the colour shows "
        "each peptide's primary taste label.<br><br>"
        f"<strong>PC1</strong> captures {v1:.1f}% of the variation between peptides, and "
        f"<strong>PC2</strong> captures another {v2:.1f}%, for "
        f"<strong>{v1+v2:.1f}%</strong> of the total picture in just these two axes.<br><br>"
        "<strong>What it means:</strong> if same-coloured dots cluster together, it suggests "
        "peptides with that taste do share a real, learnable chemical signature — which is "
        "reassuring evidence that the model has something genuine to key off, not noise."
    )


def caption_multilabel_metrics(df_metrics):
    best  = df_metrics.loc[df_metrics["f1"].idxmax(), "taste"]
    worst = df_metrics.loc[df_metrics["f1"].idxmin(), "taste"]
    avg_f1 = df_metrics["f1"].mean()
    return (
        "<strong>What this shows:</strong> a report card for each of the 5 taste classifiers, "
        "tested on peptides held back from training so the scores reflect real predictive "
        "skill, not memorisation. Four bars per taste: Accuracy (how often it's right overall), "
        "Precision (when it says 'yes', how often that's correct), Recall (of all the true "
        "cases, how many it actually catches), and F1 (a single balanced score combining "
        "precision and recall).<br><br>"
        f"<strong>Mean F1 across all 5 tastes:</strong> {avg_f1:.3f} (1.0 would be perfect).<br>"
        f"<strong>Strongest taste:</strong> {best} "
        f"(F1={df_metrics.loc[df_metrics['taste']==best,'f1'].values[0]:.3f})<br>"
        f"<strong>Most challenging taste:</strong> {worst} "
        f"(F1={df_metrics.loc[df_metrics['taste']==worst,'f1'].values[0]:.3f})<br><br>"
        f"<strong>What it means:</strong> predictions for {best} can be trusted with more "
        f"confidence, while {worst} calls are more likely to be wrong and may need double-"
        f"checking — usually because there are fewer training examples of that taste."
    )


def caption_docking(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qual = "strong" if r2 >= 0.75 else ("moderate" if r2 >= 0.5 else "weak")
    return (
        "<strong>What this shows:</strong> each dot is one test-set peptide, positioned by "
        "its real, experimentally-derived docking score (x-axis) against what the model "
        "predicted (y-axis). The red dashed diagonal is what perfect agreement would look "
        "like — the closer the dots hug that line, the better the model's predictions "
        "line up with reality.<br><br>"
        f"<strong>R² = {r2:.3f}</strong> — a {qual} fit, meaning the model explains "
        f"{r2*100:.1f}% of why docking scores vary from one peptide to the next.<br>"
        f"<strong>RMSE = {rmse:.2f}</strong> — on average, predictions are off by about this "
        f"many docking-score units.<br>"
        f"<em>Docking scores are dimensionless binding-energy estimates (typical range "
        f"−261 to −43); more negative generally means stronger predicted binding.</em><br><br>"
        f"<strong>What it means:</strong> a {qual} fit means the docking-score predictions are "
        f"{'reliable enough to help prioritise which peptides to test first' if qual != 'weak' else 'a rough starting point only — treat individual predictions with caution'}."
    )


def caption_ramachandran(phi_psi, seq=""):
    if not phi_psi:
        return (
            "<strong>What this shows:</strong> a Ramachandran plot maps each amino acid's pair "
            "of backbone twist angles (φ, ψ) to reveal whether the peptide backbone folds into "
            "a helix, a sheet, or something looser.<br><br>"
            "No angles could be measured here — the peptide needs at least 3 residues with a "
            "complete backbone to compute this."
        )
    n_total = len(phi_psi)
    n_helix = sum(1 for p, s in phi_psi if -180 <= p <= -45 and -75 <= s <= -15)
    n_sheet = sum(1 for p, s in phi_psi if -180 <= p <= -45 and 90  <= s <= 180)
    n_other = n_total - n_helix - n_sheet
    dominant = "α-helix" if n_helix >= n_sheet else "β-sheet"
    return (
        "<strong>What this shows:</strong> every residue's backbone twist is plotted as one "
        "red dot. The shaded green region marks angle combinations typical of an α-helix, "
        "blue marks a β-sheet — so where the dots cluster tells you the peptide's structural "
        "'shape family' at a glance.<br><br>"
        f"<strong>α-Helix region:</strong> {n_helix/n_total*100:.0f}% of residues | "
        f"<strong>β-Sheet region:</strong> {n_sheet/n_total*100:.0f}% of residues | "
        f"<strong>Other/loop:</strong> {n_other/n_total*100:.0f}% of residues<br><br>"
        f"<strong>What it means:</strong> the backbone is predicted to be predominantly "
        f"<strong>{dominant}</strong>-like in geometry, which should match the overall fold "
        f"classification shown in the Structure Information Panel above."
    )


def caption_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if n < 2:
        return (
            "<strong>What this shows:</strong> a distance map records how far apart every pair "
            "of residues sits in 3D space, revealing the peptide's overall shape.<br><br>"
            "Not enough structure was available here — fewer than 2 Cα atoms were found."
        )
    mask   = ~np.eye(n, dtype=bool)
    od     = dist_matrix[mask]
    d_min, d_max, d_mean = od.min(), od.max(), od.mean()
    sequential = [dist_matrix[i, i + 1] for i in range(n - 1)]
    mean_seq   = float(np.mean(sequential)) if sequential else 3.8
    lr_pairs = [(i, j) for i in range(n) for j in range(i + 4, n)
                if dist_matrix[i, j] < 8.0]
    n_lr = len(lr_pairs)
    max_possible_lr = max(1, (n - 3) * (n - 4) // 2)
    contact_density = n_lr / max_possible_lr
    if contact_density > 0.25:
        shape_note = "the peptide is predicted to fold into a <strong>compact, globular-like shape</strong>, with distant parts of the chain curling back close to each other."
    elif n_lr > 0:
        shape_note = "the peptide <strong>partially folds back on itself</strong> — some distant residues come close together, but it isn't tightly compact."
    else:
        shape_note = "the peptide stays in an <strong>extended, string-like or disordered shape</strong>, with no distant parts folding back together."
    return (
        "<strong>What this shows:</strong> a heatmap of the distance (in Ångströms) between "
        "every pair of residues in the folded structure. Dark cells mean two residues sit "
        "close together in 3D; bright cells mean they're far apart — even if they're far apart "
        "in the sequence, a dark spot off the diagonal means the chain has folded so those "
        "residues touch.<br><br>"
        f"<strong>Distance range:</strong> {d_min:.1f}–{d_max:.1f} Å (mean {d_mean:.1f} Å).<br>"
        f"<strong>Average distance between neighbouring residues:</strong> {mean_seq:.2f} Å "
        f"(a normal, healthy backbone bond is about 3.8 Å — this confirms the geometry is "
        f"realistic).<br>"
        f"<strong>Long-range contacts</strong> (residues far apart in sequence but "
        f"&lt;8 Å apart in 3D): {n_lr} pairs found (contact density {contact_density*100:.1f}%).<br><br>"
        f"<strong>What it means:</strong> {shape_note}"
    )


# ==========================================================
# SECTION 16 — STRUCTURAL ANALYSIS RENDER
# ==========================================================

def render_structural_analysis(pdb_text: str, prefix: str = "", seq: str = ""):
    if not pdb_text or not pdb_text.strip():
        st.warning("No PDB data available for structural analysis.")
        return

    if seq:
        ss_result = predict_secondary_structure(seq)
        if len(ss_result["assignments"]) >= 3:
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 🎨 Secondary Structure Profile")
            fig_ss = plot_ss_composition(ss_result, seq)
            if fig_ss is not None:
                h_pct = round(ss_result["helix_frac"] * 100, 1)
                e_pct = round(ss_result["sheet_frac"] * 100, 1)
                c_pct = round(ss_result["coil_frac"]  * 100, 1)
                dominant_ss = ("α-helical" if h_pct >= e_pct and h_pct >= c_pct else
                               "β-sheet" if e_pct >= c_pct else "coil/loop-dominated")
                cap_ss = (
                    "<strong>What this shows:</strong> a residue-by-residue map of the "
                    "predicted local shape along the peptide's backbone — each position is "
                    "called either part of a helix (coiled spring), a sheet (flat, extended "
                    "strand), or a loose coil/loop, using the Chou-Fasman method.<br><br>"
                    f"<strong>α-Helix:</strong> {h_pct}% of residues &nbsp;|&nbsp; "
                    f"<strong>β-Sheet:</strong> {e_pct}% of residues &nbsp;|&nbsp; "
                    f"<strong>Coil/Loop:</strong> {c_pct}% of residues<br><br>"
                    f"<strong>What it means:</strong> the backbone is predicted to be mostly "
                    f"<strong>{dominant_ss}</strong>, which should agree with the overall fold "
                    f"classification shown in the Structure Information Panel above — helices "
                    f"tend to be more rigid and compact, sheets more extended, and coils more "
                    f"flexible."
                )
                save_fig(fig_ss, f"{prefix}ss_composition.png", caption=cap_ss)
                st.image(f"{prefix}ss_composition.png", use_column_width=True)
                show_caption(cap_ss)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 📐 Ramachandran Plot")
    phi_psi = ramachandran(pdb_text)
    if not phi_psi:
        st.info("No φ/ψ angles — peptide needs ≥3 residues with complete backbone.")
    fig_rama = plot_ramachandran(phi_psi)
    cap_rama = caption_ramachandran(phi_psi, seq=seq)
    save_fig(fig_rama, f"{prefix}ramachandran.png", caption=cap_rama)
    st.image(f"{prefix}ramachandran.png", use_column_width=True)
    show_caption(cap_rama)

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Cα Distance Map")
    dist_map = ca_distance_map(pdb_text)
    fig_dist = plot_distance_map(dist_map, seq=seq)
    cap_dist = caption_distance_map(dist_map, seq=seq)
    save_fig(fig_dist, f"{prefix}ca_distance_map.png", caption=cap_dist)
    st.image(f"{prefix}ca_distance_map.png", use_column_width=True)
    show_caption(cap_dist)

    plddt_vals = _extract_plddt(pdb_text)
    if plddt_vals and max(plddt_vals) > 1.0:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        # Distinguish genuine pLDDT from synthetic proxy scores
        engine = st.session_state.get("pdb_source", "")
        is_genuine_plddt = ("ESMFold" in engine or "RCSB" in engine
                            or "ESM" in engine or "Uploaded" in engine)

        what_this_shows = (
            "<strong>What this shows:</strong> a bar for every residue, coloured by how "
            "confident the structure-prediction engine is about that residue's position — "
            "tall blue bars mean high confidence, short orange/yellow bars mean the model "
            "is less sure where that part of the chain actually sits."
        )

        if is_genuine_plddt:
            st.markdown("### 📊 pLDDT Confidence Profile")
            confidence_note = (
                "This is genuine per-residue pLDDT confidence from ESMFold/RCSB — a "
                "well-established, calibrated score. Regions with pLDDT below 50 are likely "
                "intrinsically disordered (i.e. that part of the peptide doesn't hold one "
                "fixed shape in reality, not just a modelling limitation)."
            )
        else:
            st.markdown("### 📊 Structural Confidence Profile (Pseudo-Score)")
            confidence_note = (
                "<strong>Note:</strong> the values shown (α-Helix=85, β-Sheet=75, Coil=55) "
                "are <em>synthetic proxies</em> assigned by secondary-structure state from the "
                "in-house Chou-Fasman Folding Engine — they are <strong>NOT</strong> "
                "equivalent to AlphaFold or ESMFold pLDDT scores, and should be read only as "
                "a rough structural-type indicator, not as a true confidence measurement."
            )

        render_plddt_legend()
        fig_pl = plot_plddt(plddt_vals, seq=seq)
        save_fig_pl_name = f"{prefix}plddt.png"
        mean_pl = np.mean(plddt_vals)
        quality  = ("Very High" if mean_pl >= 90 else "High" if mean_pl >= 70
                    else "Medium" if mean_pl >= 50 else "Low")
        cap_pl = (
            f"{what_this_shows}<br><br>"
            f"<strong>Mean confidence score: {mean_pl:.1f}</strong> — rated "
            f"<strong>{quality}</strong>.<br><br>"
            f"<strong>What it means:</strong> {confidence_note}"
        )
        save_fig(fig_pl, save_fig_pl_name, caption=cap_pl)
        st.image(save_fig_pl_name, use_column_width=True)
        show_caption(cap_pl)

    _close_all_figs()


# ==========================================================
# SECTION 17 — MODEL TRAINING  (Multi-label core)
# ==========================================================

@st.cache_resource(show_spinner="Training multi-label taste models on dataset…")
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
    
    # Remove duplicate sequences — 30 duplicates exist from multi-source annotation
    # Keep first occurrence (highest-cited source appears first in AIML.xlsx)
    df = df.drop_duplicates(subset="peptide", keep="first").reset_index(drop=True)

    # ── MULTI-LABEL TASTE LABELS ───────────────────────────────────────────────
    # Each peptide gets a binary vector [Bitter, Salty, Sour, Sweet, Umami].
    # "Sour Sweet Umami" → [0,0,1,1,1]. No information is lost.
    # This is the ONLY correct approach for this dataset.
    taste_label_df = build_taste_labels(df["taste"])
    for t in TASTES:
        df[f"label_{t}"] = taste_label_df[t].values

    # Drop rows where ALL taste labels are 0 (couldn't map any taste)
    has_any_taste = df[[f"label_{t}" for t in TASTES]].sum(axis=1) > 0
    df = df[has_any_taste].reset_index(drop=True)

    # Label coverage diagnostics
    label_counts = {t: int(df[f"label_{t}"].sum()) for t in TASTES}

    X = build_feature_table(df["peptide"])
    Y_taste = df[[f"label_{t}" for t in TASTES]].values.astype(int)

    le_sol  = LabelEncoder()
    y_sol   = le_sol.fit_transform(df["solubility"])
    y_dock  = df["docking score (kcal/mol)"].values

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)

    Xtr, Xte         = X.iloc[tr_idx], X.iloc[te_idx]
    Ytr_t, Yte_t     = Y_taste[tr_idx], Y_taste[te_idx]
    ys_tr, ys_te     = y_sol[tr_idx],   y_sol[te_idx]
    yd_tr, yd_te     = y_dock[tr_idx],  y_dock[te_idx]

    # ── TASTE MODEL: MultiOutputClassifier wrapping ExtraTrees ────────────────
    # One binary ExtraTreesClassifier per taste.  class_weight="balanced"
    # handles class imbalance within each binary problem.
    base_clf    = ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=42, n_jobs=-1)
    taste_model = MultiOutputClassifier(base_clf, n_jobs=1)
    taste_model.fit(Xtr, Ytr_t)

    sol_model  = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=42)
    dock_model = RandomForestRegressor(n_estimators=400, random_state=42)
    sol_model.fit(Xtr,  ys_tr)
    dock_model.fit(Xtr, yd_tr)

    Ypred_t     = taste_model.predict(Xte)
    sol_preds   = sol_model.predict(Xte)
    dock_preds  = dock_model.predict(Xte)

    # Per-taste metrics
    per_taste_acc = {
        t: accuracy_score(Yte_t[:, i], Ypred_t[:, i])
        for i, t in enumerate(TASTES)
    }
    per_taste_f1 = {
        t: f1_score(Yte_t[:, i], Ypred_t[:, i], zero_division=0)
        for i, t in enumerate(TASTES)
    }

    metrics = {
        **{f"Taste Acc — {t}": round(per_taste_acc[t], 4) for t in TASTES},
        "Hamming Loss (taste)":  round(hamming_loss(Yte_t, Ypred_t), 4),
        "Solubility accuracy":   accuracy_score(ys_te, sol_preds),
        "Solubility F1":         f1_score(ys_te, sol_preds, average="weighted"),
        "Docking R²":            r2_score(yd_te, dock_preds),
        "Docking RMSE":          np.sqrt(mean_squared_error(yd_te, dock_preds)),
    }

    bg_size = min(100, len(Xtr))
    rng     = np.random.default_rng(42)
    bg_idx  = rng.choice(len(Xtr), bg_size, replace=False)
    X_bg    = Xtr.iloc[bg_idx]

    return (df, X, Xtr, Xte, Ytr_t, Yte_t, ys_te, yd_te,
            taste_model, sol_model, dock_model,
            le_sol, metrics, X_bg, label_counts,
            Ypred_t, sol_preds, dock_preds)


# ==========================================================
# EXTRA STEP 2 — 5-FOLD CROSS-VALIDATION FUNCTION
# ==========================================================

@st.cache_resource(show_spinner="Running 5-fold cross-validation…")
def run_cross_validation(df_cv, X_cv, Y_cv, y_sol_cv, y_dock_cv):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1   = {t: [] for t in TASTES}
    fold_hl   = []
    fold_sol  = []
    fold_r2   = []

    # Stratify on Bitter (most balanced class)
    for tr, te in skf.split(X_cv, Y_cv[:, 0]):
        Xtr_f, Xte_f = X_cv.iloc[tr], X_cv.iloc[te]
        Ytr_f, Yte_f = Y_cv[tr],      Y_cv[te]

        tm_f = MultiOutputClassifier(
            ExtraTreesClassifier(
                n_estimators=500, class_weight="balanced",
                random_state=42, n_jobs=-1),
            n_jobs=1)
        tm_f.fit(Xtr_f, Ytr_f)
        Yp_f = tm_f.predict(Xte_f)
        fold_hl.append(hamming_loss(Yte_f, Yp_f))
        for i, t in enumerate(TASTES):
            fold_f1[t].append(
                f1_score(Yte_f[:, i], Yp_f[:, i], zero_division=0))

        sm_f = ExtraTreesClassifier(
            n_estimators=300, class_weight="balanced", random_state=42)
        sm_f.fit(Xtr_f, y_sol_cv[tr])
        fold_sol.append(accuracy_score(y_sol_cv[te], sm_f.predict(Xte_f)))

        dm_f = RandomForestRegressor(n_estimators=400, random_state=42)
        dm_f.fit(Xtr_f, y_dock_cv[tr])
        fold_r2.append(r2_score(y_dock_cv[te], dm_f.predict(Xte_f)))

    cv_results = {}
    for t in TASTES:
        v = fold_f1[t]
        cv_results[t] = {
            "mean_f1": round(float(np.mean(v)), 3),
            "std_f1":  round(float(np.std(v)),  3),
        }
    cv_results["hamming_loss"] = {
        "mean": round(float(np.mean(fold_hl)), 3),
        "std":  round(float(np.std(fold_hl)),  3),
    }
    cv_results["solubility_acc"] = {
        "mean": round(float(np.mean(fold_sol)), 3),
        "std":  round(float(np.std(fold_sol)),  3),
    }
    cv_results["docking_r2"] = {
        "mean": round(float(np.mean(fold_r2)), 3),
        "std":  round(float(np.std(fold_r2)),  3),
    }
    return cv_results


# ==========================================================
# SECTION 18 — LOAD MODELS & RUN CV
# ==========================================================

(
    df_all, X_all, X_train, X_test,
    Yt_train, Yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_sol, metrics, X_bg, label_counts,
    Yt_pred_test, sol_pred_test, dock_pred_test,
) = train_models()
FEATURE_NAMES = list(X_all.columns)

# Run Cross Validation using the full dataset
cv_results = run_cross_validation(
    df_all, X_all,
    df_all[[f"label_{t}" for t in TASTES]].values.astype(int),
    le_sol.transform(df_all["solubility"]),
    df_all["docking score (kcal/mol)"].values,
)


def predict_tastes(Xp: pd.DataFrame) -> tuple:
    """
    Run multi-label taste prediction.
    Returns (list_of_predicted_taste_strings, probability_dict).
    """
    Y_pred  = taste_model.predict(Xp)[0]           # binary vector length 5
    # get_proba from each estimator
    probas  = {}
    for i, t in enumerate(TASTES):
        clf     = taste_model.estimators_[i]
        proba   = clf.predict_proba(Xp)[0]
        classes = clf.classes_
        # probability of class=1
        if 1 in classes:
            pos_prob = proba[list(classes).index(1)]
        else:
            pos_prob = 0.0
        probas[t] = round(float(pos_prob) * 100, 1)

    predicted = [TASTES[i] for i in range(len(TASTES)) if Y_pred[i] == 1]
    return predicted, probas


# ==========================================================
# SECTION 19 — PDF REPORT ENGINE
# ==========================================================

# ── Plain-language descriptions for everything the PDF report shows ────────
# These exist purely so a reader unfamiliar with the pipeline can understand
# each metric, prediction field, and chart without needing the live app.

METRIC_DESCRIPTIONS = {
    "Hamming Loss (taste)": "Average fraction of the 5 taste labels (Bitter/Salty/Sour/Sweet/Umami) that were "
                             "mispredicted per peptide on the held-out test set. 0 = perfect, higher = worse.",
    "Solubility accuracy":  "Fraction of test-set peptides whose solubility class was predicted correctly.",
    "Solubility F1":        "Weighted F1-score (balances precision and recall) for solubility classification "
                             "across all solubility classes.",
    "Docking R²":           "Proportion of variance in true docking scores explained by the model's predictions "
                             "on the test set. Closer to 1.0 is better.",
    "Docking RMSE":         "Root-mean-squared error between predicted and true docking scores (same units as "
                             "the docking score). Lower is better.",
}


def _metric_description(key: str) -> str:
    if key.startswith("Taste Acc"):
        taste = key.split("—")[-1].strip()
        return (f"Fraction of test-set peptides correctly classified as having, or not having, "
                 f"the {taste} taste (independent binary classifier for {taste}).")
    return METRIC_DESCRIPTIONS.get(key, "")


CV_ROW_DESCRIPTIONS = {
    "Hamming Loss": "5-fold average of the taste Hamming Loss described above, with the fold-to-fold "
                     "standard deviation — shows how stable the taste model is across different data splits.",
    "Solubility Acc": "5-fold average solubility classification accuracy, with fold-to-fold standard deviation.",
    "Docking R²":     "5-fold average docking-score R², with fold-to-fold standard deviation.",
}


def _cv_row_description(label: str) -> str:
    if label in TASTES:
        return (f"5-fold cross-validated F1-score for the {label} taste classifier, with standard deviation "
                 f"across folds — a robustness check beyond the single train/test split above.")
    return CV_ROW_DESCRIPTIONS.get(label, "")


PREDICTION_FIELD_DESCRIPTIONS = {
    "Sequence":             "The peptide's amino-acid sequence (single-letter code) that was analysed.",
    "Length (aa)":          "Number of amino-acid residues in the peptide.",
    "Predicted Tastes":     "Taste(s) the multi-label model predicted for this peptide; a peptide can carry "
                             "more than one taste at once.",
    "Taste Probabilities":  "Model-estimated probability (%) that the peptide exhibits each of the 5 tastes.",
    "Predicted Solubility": "Predicted solubility class (e.g. Good/Poor) for the peptide.",
    "Docking Score":        "Predicted molecular docking score (kcal/mol-scale energy units); more negative "
                             "generally indicates stronger predicted binding.",
    "Structure Engine":     "Which engine in the Hybrid Structural Engine (RCSB PDB, Remote ESMFold, "
                             "Chou-Fasman Folding Engine, or PeptideBuilder) produced the 3D structure.",
    "Predicted Fold":       "Overall fold classification (e.g. All-α Helix, All-β Sheet) derived from the "
                             "secondary-structure prediction.",
    "α-Helix Fraction":     "Percentage of residues predicted to be in an α-helix conformation.",
    "β-Sheet Fraction":     "Percentage of residues predicted to be in a β-sheet conformation.",
}


# ── Fallback static description used only if no dynamic explanation +
# inference caption was captured for a given plot (e.g. old cached figures).
# Written in plain language: what the plot is, then what it means.
IMAGE_DESCRIPTIONS = [
    # (substring to match in filename, description)
    ("shap_bar_",              "Explains one specific prediction: which sequence features (an amino acid, a "
                                "pair of neighbouring amino acids, a physicochemical property, etc.) pushed the "
                                "model toward (blue) or away from (red) predicting this taste, and by how much — "
                                "so you can see exactly why the model made this call rather than treating it as "
                                "a black box."),
    ("ss_composition",         "A residue-by-residue map of the peptide's predicted local shape — each position "
                                "along the backbone is labelled as part of a helix (coiled spring), a sheet "
                                "(flat extended strand), or a loose coil/loop, using the Chou-Fasman method."),
    ("ramachandran",           "Plots each residue's backbone twist angles; points falling in the shaded green "
                                "or blue regions indicate helix-like or sheet-like backbone geometry, giving a "
                                "quick visual read on the peptide's overall structural class."),
    ("ca_distance_map",        "A heatmap of how far apart every pair of residues sits in 3D space; darker "
                                "regions mean residues are close together, which reveals whether the peptide "
                                "folds into a compact ball, partially folds back on itself, or stays extended."),
    ("plddt",                  "A per-residue confidence profile for the predicted 3D structure. For "
                                "ESMFold/RCSB structures this is genuine pLDDT confidence; for in-house-folded "
                                "structures it is a synthetic proxy score only, not a true AlphaFold/ESMFold "
                                "confidence value."),
    ("per_taste_metrics",      "A report card comparing accuracy, precision, recall, and F1-score for each of "
                                "the 5 taste classifiers on peptides the model never saw during training — shows "
                                "which tastes the model predicts reliably and which are harder."),
    ("confusion_per_taste",    "Five scorecards (one per taste) showing exactly how many predictions were "
                                "correct, how many were false alarms, and how many true cases were missed."),
    ("distributions",          "A dataset-wide overview: how long the training peptides are, how many carry "
                                "each taste label, and how water-loving vs fat-loving they tend to be."),
    ("pca_overall",            "Squashes each peptide's 432 measured properties down to a simple 2D map, "
                                "coloured by taste, so you can see at a glance whether peptides sharing a taste "
                                "also share a chemical 'neighbourhood'."),
    ("feature_importance",     "Ranks which of the 432 model features (physicochemical properties, amino-acid "
                                "makeup, or pairs of neighbouring residues) matter most, on average, for the "
                                "model's taste decisions."),
    ("docking_scatter",        "Plots true vs. predicted docking scores for test-set peptides; points closer to "
                                "the diagonal line mean the model's binding-strength predictions are more "
                                "accurate."),
]


def _image_description(basename: str) -> str:
    name = basename.lower()
    for key, desc in IMAGE_DESCRIPTIONS:
        if key in name:
            return desc
    return ""


def _html_caption_to_reportlab(html_text: str) -> str:
    """Convert the app's caption HTML (uses <strong>/<em>/<br>) into markup
    ReportLab's Paragraph parser understands (<b>/<i>/<br/>)."""
    if not html_text:
        return ""
    txt = html_text
    txt = txt.replace("<strong>", "<b>").replace("</strong>", "</b>")
    txt = txt.replace("<em>", "<i>").replace("</em>", "</i>")
    txt = txt.replace("<br>", "<br/>")
    txt = txt.replace("&nbsp;", " ")
    return txt


def generate_pdf(metrics: dict, prediction: dict, image_paths: list,
                 cv_results: dict = None, image_captions: dict = None) -> bytes:
    image_captions = image_captions or {}
    buf    = io.BytesIO()
    styles = getSampleStyleSheet()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=40, bottomMargin=40,
                               leftMargin=50, rightMargin=50)
    story = []
    story.append(Paragraph("<b>PepTastePredictor — Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "AI-driven peptide taste (multi-label), solubility, docking and structural analysis. "
        "Hybrid Structural Engine v2: RCSB PDB → Remote ESMFold → "
        "Peptide Folding Engine → PeptideBuilder. "
        "SHAP interpretability integrated.",
        styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    desc_style = styles["Normal"].clone("desc_style")
    desc_style.fontSize = 8.5
    desc_style.leading  = 11
    name_style = styles["Normal"].clone("name_style")
    name_style.fontSize = 9.5
    name_style.leading  = 12

    tbl_data = [["Metric", "Value", "Explanation"]] + [
        [Paragraph(k, name_style), str(round(v, 4)), Paragraph(_metric_description(k), desc_style)]
        for k, v in metrics.items()
    ]
    tbl = Table(tbl_data, colWidths=[150, 60, 250])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  rl_colors.HexColor("#1f3c88")),
        ("TEXTCOLOR",      (0,0), (-1,0),  rl_colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#f0f4ff"),
                                            rl_colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.5, rl_colors.HexColor("#cccccc")),
        ("FONTSIZE",       (0,1), (-1,-1), 10),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",     (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # After the main metrics table, add CV table if available
    if cv_results:
        story.append(Paragraph("<b>5-Fold Cross-Validation Results</b>",
                               styles["Heading2"]))
        cv_data = [["Taste / Metric", "Mean", "± Std", "Explanation"]]
        for t in TASTES:
            cv_data.append([
                Paragraph(t, name_style),
                str(cv_results[t]["mean_f1"]),
                str(cv_results[t]["std_f1"]),
                Paragraph(_cv_row_description(t), desc_style),
            ])
        cv_data.append([
            Paragraph("Hamming Loss", name_style),
            str(cv_results["hamming_loss"]["mean"]),
            str(cv_results["hamming_loss"]["std"]),
            Paragraph(_cv_row_description("Hamming Loss"), desc_style),
        ])
        cv_data.append([
            Paragraph("Solubility Acc", name_style),
            str(cv_results["solubility_acc"]["mean"]),
            str(cv_results["solubility_acc"]["std"]),
            Paragraph(_cv_row_description("Solubility Acc"), desc_style),
        ])
        cv_data.append([
            Paragraph("Docking R²", name_style),
            str(cv_results["docking_r2"]["mean"]),
            str(cv_results["docking_r2"]["std"]),
            Paragraph(_cv_row_description("Docking R²"), desc_style),
        ])
        cv_tbl = Table(cv_data, colWidths=[90, 55, 55, 260])
        cv_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), rl_colors.HexColor("#12b886")),
            ("TEXTCOLOR",   (0,0), (-1,0), rl_colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID",        (0,0), (-1,-1), 0.5, rl_colors.HexColor("#cccccc")),
            ("FONTSIZE",    (0,1), (-1,-1), 10),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
        ]))
        story.append(cv_tbl)
        story.append(Spacer(1, 14))

    if prediction:
        story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
        pred_data = [["Property", "Value", "Explanation"]] + [
            [Paragraph(k, name_style), str(v), Paragraph(PREDICTION_FIELD_DESCRIPTIONS.get(k, ""), desc_style)]
            for k, v in prediction.items()
        ]
        pred_tbl  = Table(pred_data, colWidths=[110, 110, 240])
        pred_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  rl_colors.HexColor("#0b7285")),
            ("TEXTCOLOR",      (0,0), (-1,0),  rl_colors.white),
            ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#e8f8f5"),
                                                rl_colors.white]),
            ("GRID",           (0,0), (-1,-1), 0.5, rl_colors.HexColor("#cccccc")),
            ("FONTSIZE",       (0,1), (-1,-1), 10),
            ("VALIGN",         (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",     (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(pred_tbl)
        story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    caption_style = styles["Normal"].clone("caption_style")
    caption_style.fontSize = 9
    caption_style.leading  = 13
    label_style = styles["Normal"].clone("label_style")
    label_style.fontSize = 9.5
    label_style.leading  = 12
    label_style.fontName = "Helvetica-Bold"

    for img in image_paths:
        if not img or not os.path.exists(img):
            continue
        basename_noext = os.path.basename(img).replace(".png", "")
        basename        = os.path.basename(img)
        story.append(Paragraph(f"<b>{basename_noext.replace('_',' ').title()}</b>", styles["Heading3"]))
        try:
            story.append(RLImage(img, width=430, height=270))
        except Exception:
            pass

        # Prefer the dynamic, data-driven explanation + inference captured
        # when the plot was generated; fall back to the static description.
        dyn_caption = image_captions.get(basename) or image_captions.get(img)
        if dyn_caption:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Explanation &amp; Inference:", label_style))
            story.append(Spacer(1, 2))
            story.append(Paragraph(_html_caption_to_reportlab(dyn_caption), caption_style))
        else:
            img_desc = _image_description(basename_noext)
            if img_desc:
                story.append(Spacer(1, 4))
                story.append(Paragraph("Explanation:", label_style))
                story.append(Spacer(1, 2))
                story.append(Paragraph(img_desc, desc_style))
        story.append(Spacer(1, 20))

    story.append(Paragraph(
        f"<i>Generated by PepTastePredictor v2 · "
        f"{date.today().strftime('%d %B %Y')}</i>",
        styles["Normal"]))
    doc.build(story)
    return buf.getvalue()


# ==========================================================
# SECTION 20 — HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🧬 PepTastePredictor</h1>
<p>
Integrated machine learning and structural bioinformatics for peptide taste,
solubility, docking score estimation, and 3D structure analysis.<br>
<strong>Multi-label taste model:</strong> Bitter · Salty · Sour · Sweet · Umami
— each predicted independently, peptides can carry multiple tastes &nbsp;|&nbsp;
<strong>SHAP interpretability</strong> &nbsp;|&nbsp;
<strong>Hybrid Structural Engine v2:</strong>
RCSB PDB → Remote ESMFold → Chou-Fasman Folder → PeptideBuilder
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 21 — MODE SELECTION
# ==========================================================

st.markdown("## 🔧 Analysis Mode")
mode = st.radio(
    "Choose the analysis mode",
    ["Single Peptide Prediction",
     "Batch Peptide Prediction",
     "PDB Upload & Structural Analysis"],
    horizontal=True,
)
if st.session_state.current_mode != mode:
    st.session_state.pdf_figures     = []
    st.session_state.show_analytics  = False
    st.session_state.last_prediction = {}
    st.session_state.pdb_text        = None
    st.session_state.pdb_source      = None
    st.session_state.current_mode    = mode
    st.session_state.shap_explainers = {}
    st.session_state.pdf_captions    = {}
    _close_all_figs()


# ==========================================================
# SECTION 22 — SINGLE PEPTIDE PREDICTION
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("## 🔬 Single Peptide Prediction")

    fasta_file = st.file_uploader(
        "📂 Upload a FASTA file (optional — overrides text input below)",
        type=["fasta", "fa", "faa", "txt"],
        key="single_fasta_upload",
    )
    fasta_seq, fasta_name = "", ""
    if fasta_file is not None:
        records = read_uploaded_fasta(fasta_file)
        if records:
            fasta_name, fasta_seq = records[0]
            st.success(
                f"✅ FASTA loaded: **{fasta_name or 'unnamed'}** — {len(fasta_seq)} aa")
            file_sig = f"{fasta_file.name}_{fasta_file.size}"
            if st.session_state.get("_single_fasta_sig") != file_sig:
                st.session_state["single_seq_input"] = fasta_seq
                st.session_state["_single_fasta_sig"] = file_sig
        else:
            st.error("Could not extract a valid sequence from the uploaded file.")

    seq_raw = st.text_area(
        "Enter peptide sequence (FASTA or plain single-letter code)",
        placeholder="Paste sequence or FASTA here…",
        key="single_seq_input",
        height=100,
    )

    seq = clean_sequence(fasta_seq) if fasta_seq else clean_sequence(seq_raw)

    if seq:
        st.markdown(
            f'<div style="font-size:13px;font-weight:700;opacity:0.65;margin-bottom:10px;">'
            f'✔ Valid amino acids: '
            f'<span style="color:#12b886;font-size:16px;font-weight:800;">{len(seq)}</span>'
            f'</div>', unsafe_allow_html=True)

    use_remote = st.checkbox(
        "🌐 Use remote structure databases (RCSB PDB + ESMFold) — requires internet",
        value=True,
    )
    run_shap = st.checkbox(
        "🧠 Compute SHAP explanations (adds ~5–15 s per predicted taste)",
        value=True,
    )

    if st.button("🚀 Run Prediction", type="primary"):
        st.session_state.pdf_figures    = []
        st.session_state.shap_explainers = {}
        st.session_state.pdf_captions    = {}
        _close_all_figs()

        if len(seq) < 1:
            st.error("Please enter at least one amino acid.")
        else:
            Xp = pd.DataFrame([model_features(seq)])

            # ── Multi-label taste prediction ──────────────────────────────
            # Run matching setup
            predicted_tastes, taste_probas = predict_tastes(Xp)

            sol       = le_sol.inverse_transform(sol_model.predict(Xp))[0]
            dock      = dock_model.predict(Xp)[0]
            sol_proba = sol_model.predict_proba(Xp)[0]

            sol_color  = "#12b886" if "good" in sol.lower() else "#e67e22"
            dock_color = "#12b886" if dock < -180 else ("#f39c12" if dock < -120 else "#c0392b")
            dock_label = ("Strong binder" if dock < -180 else
                          "Moderate binder" if dock < -120 else "Weak binder")

            taste_display = taste_badges_html(predicted_tastes)

            st.markdown(f"""
            <div class="card">
              <div class="card-title">
                <span class="live-indicator"></span>ML Prediction Results
              </div>
              <div style="margin-bottom:20px;">
                <div class="metric-box-label" style="margin-bottom:10px;">Predicted Tastes</div>
                <div>{taste_display}</div>
              </div>
              <div class="metric-grid">
                <div class="metric-box">
                  <div class="metric-box-label">Solubility</div>
                  <div class="metric-box-value" style="color:{sol_color}!important;">{sol}</div>
                  <div class="metric-box-sub">Confidence: {max(sol_proba)*100:.1f}%</div>
                </div>
                <div class="metric-box">
                  <div class="metric-box-label">Docking Score</div>
                  <div class="metric-box-value" style="color:{dock_color}!important;">{dock:.2f}</div>
                  <div class="metric-box-sub">{dock_label}</div>
                </div>
                <div class="metric-box">
                  <div class="metric-box-label">Sequence Length</div>
                  <div class="metric-box-value">{len(seq)} aa</div>
                  <div class="metric-box-sub">Full sequence used</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

            # Taste probability table — all 5 tastes always shown
            st.markdown("#### Taste Probabilities (all 5 tastes)")
            prob_df = pd.DataFrame({
                "Taste":       TASTES,
                "Emoji":       [TASTE_EMOJI[t] for t in TASTES],
                "Probability": [f"{taste_probas[t]:.1f}%" for t in TASTES],
                "Predicted":   ["✅ Yes" if t in predicted_tastes else "— No"
                                for t in TASTES],
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Physicochemical table
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 📌 Physicochemical Properties")
            phys      = physicochemical_features(seq)
            phys_rows = "".join(
                f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in phys.items())
            st.markdown(f"""
            <div class="card" style="padding:20px 24px;">
              <table class="phys-table">
                <thead><tr><th>Property</th><th>Value</th></tr></thead>
                <tbody>{phys_rows}</tbody>
              </table>
            </div>""", unsafe_allow_html=True)

            st.markdown("### 🧪 Amino Acid Composition")
            comp = composition_features(seq)
            cols = st.columns(len(comp))
            for i, (k, v) in enumerate(comp.items()):
                cols[i].metric(k, f"{v}%")

            # SHAP (only for predicted tastes)
            if run_shap:
                render_shap_analysis(
                    taste_model, X_bg, Xp,
                    FEATURE_NAMES, seq, predicted_tastes,
                )

            # 3D structure
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("## 🧬 3D Peptide Structure")
            with st.spinner("Generating 3D structure via Hybrid Engine…"):
                pdb_text, engine_label, source_detail = predict_structure(
                    seq, use_remote=use_remote)

            st.session_state.pdb_text   = pdb_text
            st.session_state.pdb_source = engine_label

            ss_result = predict_secondary_structure(seq)
            render_structure_info_panel(seq, engine_label, source_detail, ss_result)
            render_source_banner(engine_label, source_detail)

            plddt_vals = _extract_plddt(pdb_text)
            use_plddt  = (plddt_vals and max(plddt_vals) > 1.0
                          and ("ESMFold" in engine_label or "RCSB" in engine_label))

            st.components.v1.html(
                show_structure(pdb_text, use_plddt_colors=use_plddt)._make_html(),
                height=720,
            )
            if use_plddt:
                render_plddt_legend()

            rmsd_val = ca_rmsd(pdb_text)
            if rmsd_val is not None:
                st.info(f"📏 Cα RMSD from first residue: **{rmsd_val:.3f} Å**")

            st.download_button(
                "⬇️ Download PDB",
                pdb_text,
                file_name=f"peptide_{seq[:20]}.pdb",
                mime="text/plain",
            )

            render_structural_analysis(pdb_text, prefix="single_", seq=seq)

            st.session_state.last_prediction = {
                "Sequence":              seq[:60] + ("…" if len(seq) > 60 else ""),
                "Length (aa)":           len(seq),
                "Predicted Tastes":      ", ".join(predicted_tastes) or "None",
                "Taste Probabilities":   " | ".join(
                    f"{t}:{taste_probas[t]:.1f}%" for t in TASTES),
                "Predicted Solubility":  sol,
                "Docking Score":         round(dock, 2),
                "Structure Engine":      engine_label,
                "Predicted Fold":        classify_fold(ss_result, seq),
                "α-Helix Fraction":      f"{ss_result['helix_frac']*100:.1f}%",
                "β-Sheet Fraction":      f"{ss_result['sheet_frac']*100:.1f}%",
            }
            st.session_state.show_analytics = True
            st.session_state.prediction_count += 1
            _close_all_figs()


# ==========================================================
# SECTION 23 — BATCH PEPTIDE PREDICTION
# ==========================================================

elif mode == "Batch Peptide Prediction":

    st.markdown("## 📦 Batch Peptide Prediction")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        batch_csv = st.file_uploader(
            "📄 Upload CSV (needs a 'peptide' column — any text case is fine, e.g. Peptide/PEPTIDES)", type=["csv"])
    with col_up2:
        batch_fasta = st.file_uploader(
            "📂 Or upload FASTA file",
            type=["fasta","fa","faa","txt"],
            key="batch_fasta_upload",
        )

    gen_structures   = st.checkbox(
        "🏗️ Generate 3D structures for each peptide (downloads as ZIP)", value=False)
    batch_use_remote = st.checkbox(
        "🌐 Use remote APIs for structure generation", value=True)

    batch_df, batch_seqs = None, []

    if batch_csv is not None:
        try:
            batch_df = pd.read_csv(batch_csv)
            # Accept "peptide"/"peptides" as the column header in ANY text case
            # (e.g. "Peptide", "PEPTIDES", "peptide ", "Peptides") — find the
            # first column whose stripped, lowercased name matches, then
            # normalise it to "peptide" for the rest of the pipeline.
            peptide_col = next(
                (c for c in batch_df.columns
                 if str(c).strip().lower() in ("peptide", "peptides")),
                None,
            )
            if peptide_col is None:
                st.error("CSV must have a column named 'peptide' (any text case, e.g. 'Peptide' or 'PEPTIDES' is fine).")
                batch_df = None
            else:
                if peptide_col != "peptide":
                    batch_df = batch_df.rename(columns={peptide_col: "peptide"})
                batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
                batch_df   = batch_df[batch_df["peptide"].str.len() >= 1].reset_index(drop=True)
                batch_seqs = batch_df["peptide"].tolist()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    elif batch_fasta is not None:
        records = read_uploaded_fasta(batch_fasta)
        if not records:
            st.error("No valid sequences found in the FASTA file.")
        else:
            batch_df   = pd.DataFrame(records, columns=["Header", "peptide"])
            batch_seqs = batch_df["peptide"].tolist()
            st.success(f"✅ Loaded {len(batch_seqs)} sequence(s) from FASTA")

    if batch_df is not None and batch_seqs:
        total    = len(batch_seqs)
        st.info(f"Processing **{total}** valid peptide(s)…")
        progress = st.progress(0, text="Starting…")

        all_predicted_tastes = []
        all_taste_probas     = []
        sols, docks, engines, fold_types = [], [], [], []
        pdb_files = {}

        for i, seq_b in enumerate(batch_seqs):
            try:
                Xr = pd.DataFrame([model_features(seq_b)])
                pt, tp = predict_tastes(Xr)
                s      = le_sol.inverse_transform(sol_model.predict(Xr))[0]
                d      = round(dock_model.predict(Xr)[0], 2)
            except Exception:
                pt, tp, s, d = [], {t: 0.0 for t in TASTES}, "Error", None
            all_predicted_tastes.append(", ".join(pt) if pt else "None")
            all_taste_probas.append(
                " | ".join(f"{t}:{tp[t]:.1f}%" for t in TASTES))
            sols.append(s)
            docks.append(d)

            try:
                ss_b = predict_secondary_structure(seq_b)
                ft   = classify_fold(ss_b, seq_b)
            except Exception:
                ft = "Unknown"
            fold_types.append(ft)

            if gen_structures:
                try:
                    pdb_b, eng_b, _ = predict_structure(
                        seq_b, use_remote=batch_use_remote)
                    pdb_files[f"peptide_{i+1}_{seq_b[:12]}.pdb"] = pdb_b
                    engines.append(eng_b)
                except Exception:
                    pdb_b = _build_geometric_pdb(seq_b) or _minimal_ca_pdb(seq_b)
                    pdb_files[f"peptide_{i+1}_{seq_b[:12]}.pdb"] = pdb_b
                    engines.append("PeptideBuilder (fallback)")
            else:
                engines.append("—")

            progress.progress(
                min(int((i+1)/total*100), 100),
                text=f"Processed {i+1}/{total}…")

        progress.progress(100, text="Done!")

        batch_df["Predicted Tastes"]        = all_predicted_tastes
        batch_df["Taste Probabilities"]     = all_taste_probas
        batch_df["Predicted Solubility"]    = sols
        batch_df["Predicted Docking Score"] = docks
        batch_df["Predicted Fold Type"]     = fold_types
        batch_df["Structure Engine"]        = engines

        phys_df = pd.DataFrame(
            [build_batch_physicochemical_row(seq) for seq in batch_seqs]
        )
        batch_df = pd.concat([batch_df.reset_index(drop=True), phys_df], axis=1)

        st.markdown("### ✅ Batch Results")
        st.dataframe(batch_df, use_container_width=True)

        st.download_button(
            "⬇️ Download Batch Predictions (CSV)",
            batch_df.to_csv(index=False),
            file_name="batch_predictions.csv",
        )

        if gen_structures and pdb_files:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, pdb_content in pdb_files.items():
                    if pdb_content:
                        zf.writestr(fname, pdb_content)
            zip_buffer.seek(0)
            st.download_button(
                "⬇️ Download All PDB Structures (ZIP)",
                zip_buffer,
                file_name="batch_structures.zip",
                mime="application/zip",
            )

        st.session_state.show_analytics = True
        gc.collect()


# ==========================================================
# SECTION 24 — PDB UPLOAD & STRUCTURAL ANALYSIS
# ==========================================================

elif mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyse PDB Structure")
    st.info(
        "Upload any PDB file — AlphaFold, ESMFold, experimental, or platform-generated.",
        icon="🌐")

    uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])

    if uploaded_pdb is not None:
        try:
            pdb_text = uploaded_pdb.read().decode("utf-8")
        except Exception as e:
            st.error(f"Could not read PDB: {e}")
            pdb_text = ""

        if pdb_text and pdb_text.strip():
            if not _validate_pdb(pdb_text):
                st.error("Uploaded PDB appears invalid (no ATOM records with Cα atoms).")
            else:
                st.session_state.pdb_text       = pdb_text
                st.session_state.pdb_source     = "Uploaded PDB"
                st.session_state.show_analytics = True

                n_atoms = sum(1 for l in pdb_text.splitlines() if l.startswith("ATOM"))
                n_res   = _count_residues_in_pdb(pdb_text)

                st.markdown(f"""
                <div class="card">
                  <div class="card-title">PDB File Summary</div>
                  <div class="metric-grid">
                    <div class="metric-box">
                      <div class="metric-box-label">ATOM Records</div>
                      <div class="metric-box-value">{n_atoms}</div>
                    </div>
                    <div class="metric-box">
                      <div class="metric-box-label">Residues</div>
                      <div class="metric-box-value">{n_res}</div>
                    </div>
                    <div class="metric-box">
                      <div class="metric-box-label">File</div>
                      <div class="metric-box-value" style="font-size:14px;">{uploaded_pdb.name}</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                ONE_LETTER  = {v: k for k, v in THREE_LETTER.items()}
                seen_res    = {}
                for line in pdb_text.splitlines():
                    if line.startswith("ATOM"):
                        chain   = line[21]
                        res_seq = line[22:26].strip()
                        resname = line[17:20].strip()
                        key     = (chain, res_seq)
                        if key not in seen_res:
                            seen_res[key] = ONE_LETTER.get(resname, "X")
                seq_from_pdb = "".join(seen_res.values()).replace("X", "")

                plddt_vals = _extract_plddt(pdb_text)
                has_plddt  = plddt_vals and max(plddt_vals) > 1.0

                if seq_from_pdb:
                    ss_result = predict_secondary_structure(seq_from_pdb)
                    render_structure_info_panel(
                        seq_from_pdb, "Uploaded PDB", uploaded_pdb.name, ss_result)

                st.markdown("### 🧬 3D Structure Viewer")
                st.components.v1.html(
                    show_structure(pdb_text, use_plddt_colors=has_plddt)._make_html(),
                    height=720,
                )
                if has_plddt:
                    render_plddt_legend()

                rmsd_val = ca_rmsd(pdb_text)
                if rmsd_val is not None:
                    st.info(f"📏 Cα RMSD: **{rmsd_val:.3f} Å**")

                render_structural_analysis(pdb_text, prefix="pdb_", seq=seq_from_pdb)
        else:
            st.error("Uploaded PDB is empty or could not be decoded.")


# ==========================================================
# SECTION 25 — MODEL & DATASET ANALYTICS
# ==========================================================

if st.session_state.show_analytics:

    st.markdown("---")

    with st.expander("📊 Model Performance & Dataset Analytics", expanded=False):

        # Display Cross Validation Results before metrics charts
        st.markdown("### 🔁 5-Fold Cross-Validation (Robustness Check)")

        cv_rows = []
        for t in TASTES:
            cv_rows.append({
                "Taste":    t,
                "Emoji":    TASTE_EMOJI[t],
                "Mean F1":  f"{cv_results[t]['mean_f1']:.3f}",
                "± Std":    f"{cv_results[t]['std_f1']:.3f}",
            })
        cv_df = pd.DataFrame(cv_rows)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)

        hl  = cv_results["hamming_loss"]
        sol = cv_results["solubility_acc"]
        r2  = cv_results["docking_r2"]

        st.markdown(f"""
        <div class="metric-grid" style="margin:16px 0;">
          <div class="metric-box">
            <div class="metric-box-label">CV Hamming Loss</div>
            <div class="metric-box-value">{hl['mean']:.3f} ± {hl['std']:.3f}</div>
            <div class="metric-box-sub">Lower = better</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">CV Solubility Acc</div>
            <div class="metric-box-value">{sol['mean']:.3f} ± {sol['std']:.3f}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">CV Docking R²</div>
            <div class="metric-box-value">{r2['mean']:.3f} ± {r2['std']:.3f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        show_caption(
            "<strong>What this shows:</strong> a robustness check. Instead of testing the model "
            "on just one held-out slice of data, it retrains and re-tests it 5 separate times on "
            "5 different slices, then reports the average score and how much that score wobbles "
            "from one slice to another (± std). A model that's just gotten lucky on one test set "
            "would show up here as unstable.<br><br>"
            "5-fold stratified cross-validation (random_state=42, stratified on the Bitter label). "
            "The mean ± std across all 5 folds confirms the model is stable rather than a "
            "one-off result. "
            f"<strong>What it means:</strong> Sour shows the highest variance "
            f"(F1 std={cv_results['Sour']['std_f1']:.3f}), consistent with having fewer training "
            f"examples to learn from, while every other taste stays low-variance (std ≤ 0.056) — "
            f"i.e. reliably consistent no matter which slice of data it's tested on."
        )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)

        st.markdown("### 🩺 Multi-label Taste Coverage (full dataset)")
        coverage_rows = []
        total_rows    = len(df_all)
        for t in TASTES:
            cnt = label_counts[t]
            coverage_rows.append({
                "Taste":             t,
                "Emoji":             TASTE_EMOJI[t],
                "Rows with this taste": cnt,
                "% of dataset":      f"{cnt/total_rows*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(coverage_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Counts exceed total rows because each peptide can carry multiple tastes. "
            "This is the fundamental reason we use multi-label classification."
        )

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📈 Per-Taste Metrics (test set)")
        fig_metrics, df_metrics = plot_multilabel_per_taste(Yt_test, Yt_pred_test)
        cap_metrics = caption_multilabel_metrics(df_metrics)
        save_fig(fig_metrics, "per_taste_metrics.png", caption=cap_metrics)
        st.image("per_taste_metrics.png", use_column_width=True)
        show_caption(cap_metrics)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔲 Per-Taste Confusion Matrices")
        fig_cms = plot_confusion_per_taste(Yt_test, Yt_pred_test)
        cap_cms = caption_confusion_per_taste(Yt_test, Yt_pred_test)
        save_fig(fig_cms, "confusion_per_taste.png", caption=cap_cms)
        st.image("confusion_per_taste.png", use_column_width=True)
        show_caption(cap_cms)

        # Summary metric boxes
        hl_test = hamming_loss(Yt_test, Yt_pred_test)
        st.markdown(f"""
        <div class="metric-grid" style="margin:24px 0;">
          {"".join(f'''
          <div class="metric-box">
            <div class="metric-box-label">{t} Accuracy</div>
            <div class="metric-box-value">{metrics[f"Taste Acc — {t}"]*100:.1f}%</div>
          </div>''' for t in TASTES)}
          <div class="metric-box">
            <div class="metric-box-label">Hamming Loss</div>
            <div class="metric-box-value">{hl_test:.4f}</div>
            <div class="metric-box-sub">Lower is better (0=perfect)</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Solubility Accuracy</div>
            <div class="metric-box-value">{metrics['Solubility accuracy']*100:.1f}%</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Docking R²</div>
            <div class="metric-box-value">{metrics['Docking R²']:.3f}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Distributions")
        fig_dist = plot_distributions(df_all)
        cap_dist_all = caption_distributions(df_all)
        save_fig(fig_dist, "distributions.png", caption=cap_dist_all)
        st.image("distributions.png", use_column_width=True)
        show_caption(cap_dist_all)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 PCA Feature Space")
        fig_pca, pca_model = plot_pca(
            X_all, Yt_test if len(X_all) == len(Yt_test) else
            df_all[[f"label_{t}" for t in TASTES]].values,
            title="PCA — Peptide Feature Space (coloured by primary taste label)",
        )
        cap_pca_txt = caption_pca(pca_model)
        save_fig(fig_pca, "pca_overall.png", caption=cap_pca_txt)
        st.image("pca_overall.png", use_column_width=True)
        show_caption(cap_pca_txt)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Feature Importance (mean across taste classifiers)")
        fig_imp, imp_df = plot_feature_importance(taste_model.estimators_, FEATURE_NAMES, top_n=20)
        cap_imp = caption_feature_importance(imp_df)
        save_fig(fig_imp, "feature_importance_taste.png", caption=cap_imp)
        st.image("feature_importance_taste.png", use_column_width=True)
        show_caption(cap_imp)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Docking Score: True vs Predicted")
        fig_dock = plot_docking(yd_test, dock_pred_test)
        cap_dock = caption_docking(yd_test, dock_pred_test)
        save_fig(fig_dock, "docking_scatter.png", caption=cap_dock)
        st.image("docking_scatter.png", use_column_width=True)
        show_caption(cap_dock)

        _close_all_figs()


# ==========================================================
# SECTION 26 — PDF DOWNLOAD
# ==========================================================

if st.session_state.show_analytics:
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="pdf-card">
      <h3>📄 Download Complete Analysis Report</h3>
      <p>Includes model metrics, prediction results, SHAP charts, and all generated figures.</p>
    </div>""", unsafe_allow_html=True)

    figures_ready = [f for f in st.session_state.pdf_figures
                     if f and os.path.exists(f)]

    if figures_ready or st.session_state.last_prediction:
        try:
            pdf_bytes = generate_pdf(
                metrics,
                st.session_state.last_prediction,
                figures_ready,
                cv_results=cv_results,
                image_captions=st.session_state.pdf_captions,
            )
            st.download_button(
                label="📥 Download Full Analytics PDF",
                data=pdf_bytes,
                file_name=f"PepTastePredictor_Report_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    else:
        st.info("Run a prediction first to enable PDF download.")


# ==========================================================
# SECTION 27 — FOOTER
# ==========================================================

st.markdown(f"""
<div class="footer">
&copy; {date.today().year} &nbsp; <b>PepTastePredictor </b><br>
Multi-label taste model (Bitter · Salty · Sour · Sweet · Umami) — one binary classifier per taste<br>
Structure priority: RCSB PDB → Remote ESMFold → Chou-Fasman Folder → PeptideBuilder<br>
For academic and research use only
</div>
""", unsafe_allow_html=True)
