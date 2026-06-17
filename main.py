# ==========================================================
# PepTastePredictor — app.py
# Hybrid Structural Engine v2 | Light + Dark Mode
# Real metrics: Taste 70.0% | Solubility 97.5% | Docking R²=0.760
# Taste classes: Bitter, Salty, Sour, Sweet, Umami, Neutral (6-class, primary taste)
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    r2_score, confusion_matrix, classification_report,
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

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

# Primary taste classes — 6-class model (fixed order, encoder pre-fitted to this)
TASTE_CLASSES = ["Bitter", "Neutral", "Salty", "Sour", "Sweet", "Umami"]

TASTE_EMOJI = {
    "Bitter":  "😖",
    "Sweet":   "😋",
    "Salty":   "🧂",
    "Sour":    "😮‍💨",
    "Umami":   "🍖",
    "Neutral": "😶",
}

# Chou-Fasman propensity tables
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
- 🎯 Taste prediction (6 classes)
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
    # FIX 3: cache SHAP explainer here instead of @st.cache_data
    "shap_explainer":   None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==========================================================
# SECTION 6 — UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename: str):
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    gc.collect()
    if filename not in st.session_state.pdf_figures:
        st.session_state.pdf_figures.append(filename)


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


# ==========================================================
# FIX 2 — CORRECTED simplify_taste
# Swapped loop order eliminates word-order bias in composite labels.
# Single-word labels are matched directly (no change to unambiguous rows).
# Composite labels pick the first recognised non-Neutral taste by the
# priority order defined in base_tastes, not by position in the raw string.
# ==========================================================

def simplify_taste(taste_series):
    """
    Map composite taste labels (e.g. 'Sour Sweet Umami') to a single
    primary taste class.

    Priority order is defined by base_tastes, not by word position in the
    raw label — this eliminates the original word-order bias where
    'Sour Sweet' would always map to 'Sour' simply because it appeared first.

    Single-word labels are matched directly and returned immediately,
    so they are completely unaffected by the loop-order change.

    Falls back to 'Neutral' if no recognised taste word is found.
    """
    base_tastes = ["Bitter", "Sweet", "Salty", "Sour", "Umami", "Neutral"]

    def _map(t):
        words = [w.strip().lower() for w in str(t).split()]

        # Fast path: single unambiguous word — no priority logic needed
        if len(words) == 1:
            for bt in base_tastes:
                if words[0] == bt.lower():
                    return bt
            return "Neutral"

        # Composite label: iterate base_tastes (priority order), not words
        # Skip "Neutral" in first pass so it only wins if nothing else matches
        for bt in base_tastes[:-1]:
            if bt.lower() in words:
                return bt

        # Explicit Neutral as last resort
        if "neutral" in words:
            return "Neutral"

        return "Neutral"

    return taste_series.apply(_map)


def model_features(seq: str) -> dict:
    """Extract 432-dimensional feature vector from a peptide sequence."""
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


def physicochemical_features(seq: str) -> dict:
    L = len(seq)
    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            h, t, s = ana.secondary_structure_fraction()
            return {
                "Length":                L,
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


def taste_emoji(taste: str) -> str:
    for k, v in TASTE_EMOJI.items():
        if k.lower() in taste.lower():
            return v
    return "🧬"


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
        n   = np.cross(b2n, b1n)
        nn  = np.linalg.norm(n)
        n   = n / nn if nn > 1e-10 else np.array([0., 0., 1.])
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
    atom_num   = 1
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
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}           "
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
# SECTION 13 — SHAP INTERPRETABILITY
# FIX 3: Removed @st.cache_data (wrong decorator for model objects).
# The TreeExplainer is now cached in st.session_state["shap_explainer"]
# so it is built once per session and reused, avoiding repeated
# re-instantiation and hash-mismatch errors.
# ==========================================================

def compute_shap_values(_model, _X_background, _X_query, feature_names):
    """
    Compute SHAP values using TreeExplainer with interventional perturbation.

    The explainer is cached in st.session_state to avoid rebuilding it on
    every prediction call (the original @st.cache_data caused silent cache
    misses because sklearn model objects are not safely hashable by value).

    Returns a list of arrays — one per class — each shaped (1, n_features).
    Handles both the list-of-arrays format (shap < 0.42) and the 3-D ndarray
    format introduced in newer shap releases transparently.
    """
    # Build the explainer once per session; invalidate if the model changes
    if st.session_state.get("shap_explainer") is None:
        st.session_state["shap_explainer"] = shap.TreeExplainer(
            _model,
            data=_X_background,
            feature_perturbation="interventional",
        )

    explainer = st.session_state["shap_explainer"]
    sv = explainer.shap_values(_X_query)

    # Normalise output to a list-of-arrays regardless of shap version
    if isinstance(sv, np.ndarray):
        if sv.ndim == 3:          # shape: (n_samples, n_features, n_classes)
            sv = [sv[:, :, i] for i in range(sv.shape[2])]
        elif sv.ndim == 2:        # binary edge case — mirror as two-class list
            sv = [-sv, sv]
    return sv


def plot_shap_bar(shap_vals_class, feature_names, seq, predicted_class, top_n=15):
    C     = get_plot_colors()
    vals  = shap_vals_class[0]
    df_sh = pd.DataFrame({
        "feature":    [prettify_feature(f) for f in feature_names],
        "shap_value": vals,
        "abs_shap":   np.abs(vals),
    }).sort_values("abs_shap", ascending=False).head(top_n)

    colors = [C["accent1"] if v > 0 else C["red"] for v in df_sh["shap_value"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    ax.barh(
        df_sh["feature"][::-1],
        df_sh["shap_value"][::-1],
        color=colors[::-1],
        edgecolor="none",
    )
    ax.axvline(0, color=C["grid"], linewidth=1.2)
    ax.set_xlabel("SHAP Value  (impact on model output)", fontsize=11, labelpad=10)
    ax.set_title(
        f"SHAP Feature Contributions — Predicted: {predicted_class}  |  Seq: {seq[:20]}{'…' if len(seq)>20 else ''}",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    return fig


def plot_shap_waterfall_simple(shap_vals_class, feature_names, base_value,
                                seq, predicted_class, top_n=10):
    C    = get_plot_colors()
    vals = shap_vals_class[0]
    idx  = np.argsort(np.abs(vals))[::-1][:top_n]
    idx  = idx[np.argsort(idx)]

    selected_vals  = vals[idx]
    selected_names = [prettify_feature(feature_names[i]) for i in idx]
    cumulative     = np.cumsum(selected_vals)
    starts         = np.concatenate([[base_value], base_value + cumulative[:-1]])

    fig, ax = plt.subplots(figsize=(10, 5))
    apply_plot_style(fig, [ax])
    for i, (s, v, n) in enumerate(zip(starts, selected_vals, selected_names)):
        color = C["accent1"] if v >= 0 else C["red"]
        ax.barh(i, v, left=s, height=0.6, color=color, edgecolor="none")
        ax.text(s + v + (0.002 if v >= 0 else -0.002), i,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right",
                fontsize=8, color=C["text"])

    ax.axvline(base_value, color=C["orange"], linestyle="--", lw=1.5,
               label=f"Base value = {base_value:.3f}")
    ax.set_yticks(range(len(selected_names)))
    ax.set_yticklabels(selected_names, fontsize=9)
    ax.set_xlabel("Model Output (probability)", fontsize=11, labelpad=8)
    ax.set_title(
        f"SHAP Waterfall — {predicted_class} prediction  |  Seq: {seq[:20]}{'…' if len(seq)>20 else ''}",
        fontsize=12, fontweight="bold", pad=12,
    )
    leg = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def render_shap_analysis(taste_model, X_train_bg, X_query_df,
                          feature_names, seq, predicted_class, predicted_class_idx):
    """Full SHAP analysis block: bar chart + waterfall + caption."""
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("## 🧠 SHAP Interpretability")
    st.markdown(
        '<div class="shap-panel"><h4>🔍 Why did the model predict this taste?</h4>'
        'SHAP (SHapley Additive exPlanations) decomposes each prediction into '
        'per-feature contributions. <strong style="color:#1a8fd1;">Blue bars</strong> '
        'push toward the predicted class; '
        '<strong style="color:#c0392b;">red bars</strong> push away from it.</div>',
        unsafe_allow_html=True,
    )

    X_bg_np    = X_train_bg.values.astype(np.float32)
    X_query_np = X_query_df.values.astype(np.float32)

    with st.spinner("Computing SHAP values…"):
        try:
            shap_vals = compute_shap_values(
                taste_model, X_bg_np, X_query_np, tuple(feature_names))

            n_shap_classes = len(shap_vals)
            safe_idx       = min(predicted_class_idx, n_shap_classes - 1)
            sv_class       = shap_vals[safe_idx]

            # Re-use cached explainer to get expected_value — no second build
            explainer = st.session_state["shap_explainer"]
            raw_ev = explainer.expected_value
            if isinstance(raw_ev, (list, np.ndarray)):
                base_value = float(raw_ev[min(safe_idx, len(raw_ev) - 1)])
            else:
                base_value = float(raw_ev)

            fig_bar = plot_shap_bar(sv_class, feature_names, seq, predicted_class)
            save_fig(fig_bar, "shap_bar.png")
            st.image("shap_bar.png", use_column_width=True)

            top_features = pd.DataFrame({
                "feature": [prettify_feature(f) for f in feature_names],
                "shap":    sv_class[0],
                "abs":     np.abs(sv_class[0]),
            }).sort_values("abs", ascending=False).head(3)
            top3_html = "".join(
                f"<strong>#{i+1} {r['feature']}</strong> (SHAP={r['shap']:+.4f})<br>"
                for i, (_, r) in enumerate(top_features.iterrows())
            ) if not top_features.empty else ""

            show_caption(
                f"SHAP bar chart for prediction <strong>{predicted_class}</strong> "
                f"on sequence <em>{seq[:30]}{'…' if len(seq)>30 else ''}</em>.<br><br>"
                f"Top contributing features:<br>{top3_html}"
                f"Features with <strong>positive SHAP</strong> increase the probability of "
                f"{predicted_class}; negative SHAP values reduce it."
            )

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 📉 SHAP Waterfall Plot")
            fig_wf = plot_shap_waterfall_simple(
                sv_class, feature_names, base_value, seq, predicted_class)
            save_fig(fig_wf, "shap_waterfall.png")
            st.image("shap_waterfall.png", use_column_width=True)
            show_caption(
                f"Waterfall plot showing cumulative SHAP contributions for "
                f"<strong>{predicted_class}</strong> prediction. "
                f"Starting from the base value ({base_value:.3f}), each bar shifts "
                f"the output probability up or down until reaching the final prediction."
            )

        except Exception as e:
            st.warning(f"SHAP computation could not complete: {e}")


# ==========================================================
# SECTION 14 — PLOT FUNCTIONS
# ==========================================================

def plot_pca(X, y_labels, class_names, title="PCA"):
    C       = get_plot_colors()
    pca     = PCA(n_components=2)
    coords  = pca.fit_transform(X)
    v1, v2  = pca.explained_variance_ratio_[:2] * 100
    palette = _get_cmap("tab10", len(class_names))
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    for i, cls in enumerate(class_names):
        mask = y_labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1], label=cls, alpha=0.75,
                   s=35, color=palette(i / max(len(class_names)-1, 1)),
                   edgecolors="none")
    ax.set_xlabel(f"PC1 ({v1:.1f}%)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 ({v2:.1f}%)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, bbox_to_anchor=(1.02,1), loc="upper left",
                    title="Taste class", title_fontsize=9,
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    leg.get_title().set_color(C["text"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig, pca


def plot_confusion(y_true, y_pred, class_names, title, cmap):
    C   = get_plot_colors()
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), max(5, n * 0.75)))
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


def plot_feature_importance(model, feature_names, top_n=20):
    C   = get_plot_colors()
    imp = pd.DataFrame({
        "Feature":    [prettify_feature(f) for f in feature_names],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    clrs = _get_cmap("Blues")(np.linspace(0.4, 0.9, len(imp))[::-1])
    fig, ax = plt.subplots(figsize=(8, 7))
    apply_plot_style(fig, [ax])
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1],
            color=clrs, edgecolor=C["grid"])
    ax.set_xlabel("Importance Score", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Features — Taste Model",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_distributions(df):
    C           = get_plot_colors()
    seq_lengths = [len(s) for s in df["peptide"]]
    tc          = df["taste"].value_counts()
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
    for t in leg0.get_texts():
        t.set_color(C["text"])
    n_cls      = len(tc)
    cmap_tab10 = _get_cmap("tab10")
    bar_colors = [cmap_tab10(i / max(n_cls-1, 1)) for i in range(n_cls)]
    axes[1].barh(tc.index, tc.values, color=bar_colors, edgecolor=C["grid"])
    axes[1].set_xlabel("Count", fontsize=11)
    axes[1].set_title("Taste Class Distribution", fontsize=12, fontweight="bold", pad=10)
    for i, v in enumerate(tc.values):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=9, color=C["text"])
    axes[2].hist(grav_vals, bins=20, color=C["accent2"], edgecolor=C["grid"], alpha=0.85)
    axes[2].axvline(0, color=C["red"], linestyle="--", lw=2, label="Hydrophilic|Hydrophobic")
    axes[2].axvline(np.mean(grav_vals), color=C["orange"], linestyle="--", lw=2,
                    label=f"Mean={np.mean(grav_vals):.2f}")
    axes[2].set_xlabel("GRAVY",  fontsize=11)
    axes[2].set_ylabel("Count",  fontsize=11)
    axes[2].set_title("GRAVY Distribution", fontsize=12, fontweight="bold", pad=10)
    leg2 = axes[2].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg2.get_texts():
        t.set_color(C["text"])
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
    for t in leg.get_texts():
        t.set_color(C["text"])
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
    ax.set_ylabel("pLDDT",         fontsize=11, labelpad=8)
    ax.set_title(f"Per-Residue pLDDT Confidence  (mean = {mean_pl:.1f})",
                 fontsize=12, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    leg = ax.legend(fontsize=8, loc="lower right",
                    facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
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
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


# ==========================================================
# SECTION 15 — DYNAMIC CAPTIONS
# ==========================================================

def caption_distributions(df):
    lengths   = [len(s) for s in df["peptide"]]
    grav      = [gravy_score(s) for s in df["peptide"]]
    dom_taste = df["taste"].value_counts().idxmax()
    dom_count = df["taste"].value_counts().max()
    n_classes = df["taste"].nunique()
    mean_grav = np.mean(grav)
    glabel    = ("slightly hydrophobic" if mean_grav > 0.2 else
                 "slightly hydrophilic" if mean_grav < -0.2 else "amphipathic")
    return (
        f"<strong>Length (left):</strong> {int(np.min(lengths))}–{int(np.max(lengths))} aa, "
        f"mean {np.mean(lengths):.1f} aa.<br><br>"
        f"<strong>Taste classes (centre):</strong> &ldquo;{dom_taste}&rdquo; is the most common "
        f"({dom_count} of {len(df)} across {n_classes} classes).<br><br>"
        f"<strong>GRAVY (right):</strong> Mean {mean_grav:.2f} — dataset is <strong>{glabel}</strong>."
    )


def caption_pca(pca_model, class_names):
    v1, v2 = pca_model.explained_variance_ratio_[:2] * 100
    return (
        f"Each dot is one peptide projected into 2 principal components.<br><br>"
        f"<strong>PC1</strong> = {v1:.1f}% variance &nbsp;|&nbsp; "
        f"<strong>PC2</strong> = {v2:.1f}% variance &nbsp;|&nbsp; "
        f"<strong>Total</strong> = {v1+v2:.1f}%.<br><br>"
        f"Tight, well-separated clusters indicate reliable class discrimination."
    )


def caption_confusion_taste(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred) * 100
    cm  = confusion_matrix(y_true, y_pred)
    cp  = cm.astype(float); np.fill_diagonal(cp, 0)
    idx = np.unravel_index(np.argmax(cp), cp.shape)
    pca = cm.diagonal() / cm.sum(axis=1)
    return (
        f"Taste model: <strong>{acc:.1f}% overall accuracy</strong> on held-out test set "
        f"(stratified 80/20 split).<br><br>"
        f"Most common confusion: &ldquo;{class_names[idx[0]]}&rdquo; → "
        f"&ldquo;{class_names[idx[1]]}&rdquo; ({int(cp[idx])} instances).<br>"
        f"Best class: &ldquo;{class_names[np.argmax(pca)]}&rdquo; | "
        f"Hardest: &ldquo;{class_names[np.argmin(pca)]}&rdquo;."
    )


def caption_confusion_sol(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred) * 100
    cm  = confusion_matrix(y_true, y_pred)
    cp  = cm.astype(float); np.fill_diagonal(cp, 0)
    idx = np.unravel_index(np.argmax(cp), cp.shape)
    return (
        f"Solubility model: <strong>{acc:.1f}% accuracy</strong> on held-out test set.<br><br>"
        f"Most common error: &ldquo;{class_names[idx[0]]}&rdquo; → "
        f"&ldquo;{class_names[idx[1]]}&rdquo; ({int(cp[idx])} instances)."
    )


def caption_feature_importance(model, feature_names, top_n=20):
    imp  = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    imp  = imp.sort_values("Importance", ascending=False).head(top_n)
    top3 = [(prettify_feature(r["Feature"]), r["Importance"])
            for _, r in imp.head(3).iterrows()]
    n_d  = sum(1 for f in imp["Feature"] if f.startswith("DPC_"))
    n_a  = sum(1 for f in imp["Feature"] if f.startswith("AA_"))
    note = "Dipeptide context dominates." if n_d > n_a else "Single AA composition dominates."
    return (
        f"Top {top_n} features driving taste predictions (Gini impurity decrease).<br><br>"
        + "".join(f"<strong>#{i+1} {n}</strong> (score: {s:.4f})<br>"
                  for i, (n, s) in enumerate(top3))
        + f"<br>{n_d} DPC and {n_a} AA features in top {top_n}. {note}"
    )


def caption_docking(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qual = "strong" if r2 >= 0.75 else ("moderate" if r2 >= 0.5 else "weak")
    return (
        f"Test-set docking score predictions. Red dashed = perfect prediction.<br><br>"
        f"<strong>R² = {r2:.3f}</strong> ({qual} fit) — explains {r2*100:.1f}% of score variance.<br>"
        f"<strong>RMSE = {rmse:.2f}</strong> (docking score units).<br>"
        f"<em>Note: Scores in this dataset are dimensionless energy-function outputs "
        f"(range: −261 to −43). A proxy regression model is used to enable rapid "
        f"large-scale screening without running full docking simulations.</em>"
    )


def caption_ramachandran(phi_psi, seq=""):
    if not phi_psi:
        return ("No φ/ψ angles available — peptide requires ≥3 residues "
                "with complete backbone atoms.")
    n_total = len(phi_psi)
    n_helix = sum(1 for p, s in phi_psi if -180 <= p <= -45 and -75 <= s <= -15)
    n_sheet = sum(1 for p, s in phi_psi if -180 <= p <= -45 and 90  <= s <= 180)
    n_other = n_total - n_helix - n_sheet
    dominant = "α-helix" if n_helix >= n_sheet else "β-sheet"
    return (
        f"Backbone torsion angles{f' for <strong>{seq[:20]}</strong>' if seq else ''}.<br><br>"
        f"<strong>α-Helix region:</strong> {n_helix/n_total*100:.0f}% | "
        f"<strong>β-Sheet region:</strong> {n_sheet/n_total*100:.0f}% | "
        f"<strong>Other:</strong> {n_other/n_total*100:.0f}%<br><br>"
        f"Dominant backbone character: <strong>{dominant}</strong>."
    )


def caption_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if n < 2:
        return "Distance map unavailable — structure contains fewer than 2 Cα atoms."

    mask   = ~np.eye(n, dtype=bool)
    od     = dist_matrix[mask]
    d_min  = od.min()
    d_max  = od.max()
    d_mean = od.mean()

    sequential = [dist_matrix[i, i + 1] for i in range(n - 1)]
    mean_seq   = float(np.mean(sequential)) if sequential else 3.8

    lr_pairs = [(i, j)
                for i in range(n)
                for j in range(i + 4, n)
                if dist_matrix[i, j] < 8.0]
    n_lr = len(lr_pairs)

    n_sr = sum(1 for i in range(n) for j in range(n)
               if 1 < abs(i - j) <= 3 and dist_matrix[i, j] < 7.0)

    max_possible_lr = max(1, (n - 3) * (n - 4) // 2)
    contact_density = n_lr / max_possible_lr

    if contact_density > 0.25:
        shape_note = ("The high density of long-range contacts indicates a "
                      "<strong>compact, globular-like fold</strong> with substantial "
                      "tertiary packing between distant residues.")
    elif n_lr > 0:
        shape_note = ("The moderate number of long-range contacts suggests the peptide "
                      "<strong>partially folds back on itself</strong>, forming a loose "
                      "tertiary arrangement.")
    else:
        shape_note = ("The absence of long-range contacts is consistent with an "
                      "<strong>extended or fully disordered conformation</strong>, "
                      "typical of short linear peptides in solution.")

    seq_note = (f"Sequence <strong>{seq[:20]}{'…' if len(seq) > 20 else ''}</strong> "
                f"({n} residues). ") if seq else ""

    return (
        f"{seq_note}"
        f"The heatmap displays pairwise Cα–Cα Euclidean distances (Å) between every "
        f"residue pair in the peptide backbone. "
        f"<strong>Dark cells indicate close spatial proximity; bright cells indicate "
        f"greater separation.</strong> "
        f"The main diagonal (distance = 0 Å, each residue to itself) always appears "
        f"as the darkest stripe.<br><br>"
        f"<strong>Distance range:</strong> {d_min:.1f}–{d_max:.1f} Å "
        f"(mean {d_mean:.1f} Å across all pairs).<br>"
        f"<strong>Mean sequential Cα–Cα distance</strong> (adjacent residues): "
        f"{mean_seq:.2f} Å — the expected value for ideal backbone geometry is ~3.8 Å.<br>"
        f"<strong>Short-range contacts</strong> (|i−j| = 2–3, d &lt; 7 Å): {n_sr} "
        f"— characteristic of turns and loop regions in the backbone.<br>"
        f"<strong>Long-range contacts</strong> (|i−j| &gt; 3, d &lt; 8 Å): {n_lr} "
        f"(contact density: {contact_density * 100:.1f}% of all possible distant pairs).<br><br>"
        f"{shape_note}"
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
                save_fig(fig_ss, f"{prefix}ss_composition.png")
                st.image(f"{prefix}ss_composition.png", use_column_width=True)
                h_pct = round(ss_result["helix_frac"] * 100, 1)
                e_pct = round(ss_result["sheet_frac"] * 100, 1)
                c_pct = round(ss_result["coil_frac"]  * 100, 1)
                show_caption(
                    f"Chou-Fasman secondary-structure prediction for "
                    f"<strong>{seq[:30]}{'…' if len(seq)>30 else ''}</strong>.<br><br>"
                    f"<strong>α-Helix:</strong> {h_pct}% &nbsp;|&nbsp; "
                    f"<strong>β-Sheet:</strong> {e_pct}% &nbsp;|&nbsp; "
                    f"<strong>Coil/Loop:</strong> {c_pct}%"
                )

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 📐 Ramachandran Plot")
    phi_psi = ramachandran(pdb_text)
    if not phi_psi:
        st.info("No φ/ψ angles — peptide needs ≥3 residues with complete backbone.")
    fig_rama = plot_ramachandran(phi_psi)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.image(f"{prefix}ramachandran.png", use_column_width=True)
    show_caption(caption_ramachandran(phi_psi, seq=seq))

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Cα Distance Map")
    dist_map = ca_distance_map(pdb_text)
    fig_dist = plot_distance_map(dist_map, seq=seq)
    save_fig(fig_dist, f"{prefix}ca_distance_map.png")
    st.image(f"{prefix}ca_distance_map.png", use_column_width=True)
    show_caption(caption_distance_map(dist_map, seq=seq))

    plddt_vals = _extract_plddt(pdb_text)
    if plddt_vals and max(plddt_vals) > 1.0:
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 pLDDT Confidence Profile")
        render_plddt_legend()
        fig_pl = plot_plddt(plddt_vals, seq=seq)
        save_fig(fig_pl, f"{prefix}plddt.png")
        st.image(f"{prefix}plddt.png", use_column_width=True)
        mean_pl = np.mean(plddt_vals)
        quality = ("Very High" if mean_pl >= 90 else "High" if mean_pl >= 70
                   else "Medium" if mean_pl >= 50 else "Low")
        show_caption(
            f"Per-residue pLDDT confidence (ESMFold / RCSB source).<br><br>"
            f"<strong>Mean pLDDT:</strong> {mean_pl:.1f} — <strong>{quality} confidence</strong>. "
            f"Regions with pLDDT &lt; 50 are likely intrinsically disordered."
        )

    _close_all_figs()


# ==========================================================
# SECTION 17 — MODEL TRAINING
# FIX 1: le_taste is pre-fitted on TASTE_CLASSES before transform.
# This guarantees all 6 classes are always in the encoder regardless
# of which classes appear (or are rare) in the training data.
# The classifier itself is unaffected — it sees the same integer labels.
# ==========================================================

@st.cache_resource(show_spinner="Training models on dataset…")
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

    # ── FIX 2: corrected simplify_taste applied here ──────────────────────
    df["taste"] = simplify_taste(df["taste"])

    # Remove any rows whose taste did not map to a valid class
    df = df[df["taste"].isin(TASTE_CLASSES)].reset_index(drop=True)

    X = build_feature_table(df["peptide"])

    # ── FIX 1: pre-fit LabelEncoder on all 6 known classes ────────────────
    # le_taste.fit(TASTE_CLASSES) locks in a fixed, deterministic mapping:
    #   Bitter→0, Neutral→1, Salty→2, Sour→3, Sweet→4, Umami→5
    # This is independent of which classes happen to appear in the data,
    # so rare or zero-count classes are never silently dropped.
    le_taste = LabelEncoder()
    le_taste.fit(TASTE_CLASSES)          # <-- FIX 1
    y_taste  = le_taste.transform(df["taste"])

    le_sol   = LabelEncoder()
    y_sol    = le_sol.fit_transform(df["solubility"])
    y_dock   = df["docking score (kcal/mol)"].values

    idx            = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_taste)
    Xtr, Xte       = X.iloc[tr_idx], X.iloc[te_idx]
    yt_tr, yt_te   = y_taste[tr_idx], y_taste[te_idx]
    ys_tr, ys_te   = y_sol[tr_idx],   y_sol[te_idx]
    yd_tr, yd_te   = y_dock[tr_idx],  y_dock[te_idx]

    taste_model = ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=42)
    sol_model   = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=42)
    dock_model  = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr,   ys_tr)
    dock_model.fit(Xtr,  yd_tr)

    taste_preds = taste_model.predict(Xte)
    sol_preds   = sol_model.predict(Xte)
    dock_preds  = dock_model.predict(Xte)

    metrics = {
        "Taste accuracy":      accuracy_score(yt_te, taste_preds),
        "Taste F1 (weighted)": f1_score(yt_te, taste_preds, average="weighted"),
        "Solubility accuracy": accuracy_score(ys_te, sol_preds),
        "Solubility F1":       f1_score(ys_te, sol_preds, average="weighted"),
        "Docking R²":          r2_score(yd_te, dock_preds),
        "Docking RMSE":        np.sqrt(mean_squared_error(yd_te, dock_preds)),
    }

    bg_size = min(100, len(Xtr))
    rng     = np.random.default_rng(42)
    bg_idx  = rng.choice(len(Xtr), bg_size, replace=False)
    X_bg    = Xtr.iloc[bg_idx]

    return (df, X, Xtr, Xte, yt_te, ys_te, yd_te,
            taste_model, sol_model, dock_model,
            le_taste, le_sol, metrics, X_bg)


# ==========================================================
# SECTION 18 — LOAD MODELS
# ==========================================================

(
    df_all, X_all, X_train, X_test,
    yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics, X_bg,
) = train_models()
FEATURE_NAMES = list(X_all.columns)


# ==========================================================
# SECTION 19 — PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics: dict, prediction: dict, image_paths: list) -> bytes:
    buf    = io.BytesIO()
    styles = getSampleStyleSheet()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               topMargin=40, bottomMargin=40,
                               leftMargin=50, rightMargin=50)
    story = []
    story.append(Paragraph("<b>PepTastePredictor — Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking and structural analysis. "
        "Hybrid Structural Engine v2: RCSB PDB → Remote ESMFold → "
        "Peptide Folding Engine → PeptideBuilder. "
        "SHAP interpretability integrated for mechanistic insight.",
        styles["Normal"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    tbl_data = [["Metric", "Value"]] + [[k, str(round(v, 4))] for k, v in metrics.items()]
    tbl = Table(tbl_data, colWidths=[280, 150])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  rl_colors.HexColor("#1f3c88")),
        ("TEXTCOLOR",      (0,0), (-1,0),  rl_colors.white),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#f0f4ff"),
                                            rl_colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.5, rl_colors.HexColor("#cccccc")),
        ("FONTSIZE",       (0,1), (-1,-1), 10),
        ("TOPPADDING",     (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    if prediction:
        story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
        pred_data = [["Property", "Value"]] + [[k, str(v)] for k, v in prediction.items()]
        pred_tbl  = Table(pred_data, colWidths=[220, 210])
        pred_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0), (-1,0),  rl_colors.HexColor("#0b7285")),
            ("TEXTCOLOR",      (0,0), (-1,0),  rl_colors.white),
            ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.HexColor("#e8f8f5"),
                                                rl_colors.white]),
            ("GRID",           (0,0), (-1,-1), 0.5, rl_colors.HexColor("#cccccc")),
            ("FONTSIZE",       (0,1), (-1,-1), 10),
            ("TOPPADDING",     (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
        ]))
        story.append(pred_tbl)
        story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    figure_titles = {
        "distributions":            "Dataset Distributions",
        "pca_overall":              "PCA Feature Space",
        "confusion_taste":          "Taste Confusion Matrix",
        "confusion_solubility":     "Solubility Confusion Matrix",
        "feature_importance_taste": "Feature Importance (Gini)",
        "docking_scatter":          "Docking Score: True vs Predicted",
        "shap_bar":                 "SHAP Feature Contributions",
        "shap_waterfall":           "SHAP Waterfall Plot",
        "ss_composition":           "Secondary Structure Profile",
        "ramachandran":             "Ramachandran Plot",
        "ca_distance_map":          "Cα Distance Map",
        "plddt":                    "pLDDT Confidence Profile",
    }
    for img in image_paths:
        if not img or not os.path.exists(img):
            continue
        basename = os.path.basename(img).replace(".png", "")
        title    = next((v for k, v in figure_titles.items() if k in basename), basename)
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        try:
            story.append(RLImage(img, width=430, height=270))
        except Exception:
            pass
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
<strong>Taste classes:</strong> Bitter · Salty · Sour · Sweet · Umami · Neutral (6-class primary-taste model) &nbsp;|&nbsp;
<strong>SHAP interpretability</strong> for mechanistic insight &nbsp;|&nbsp;
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
    # FIX 3: also clear cached SHAP explainer on mode switch so it
    # is rebuilt fresh if the user returns to Single Peptide mode
    st.session_state.shap_explainer  = None
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
    fasta_seq = ""
    if fasta_file is not None:
        records = read_uploaded_fasta(fasta_file)
        if records:
            fasta_seq = records[0][1]
            st.success(
                f"✅ FASTA loaded: **{records[0][0] or 'unnamed'}** — {len(fasta_seq)} aa")
        else:
            st.error("Could not extract a valid sequence from the uploaded file.")

    seq_raw = st.text_area(
        "Enter peptide sequence (FASTA or plain single-letter code)",
        value=fasta_seq,
        placeholder="Paste sequence or FASTA here…",
        key="single_seq_input",
        height=100,
    )
    seq = clean_sequence(seq_raw)

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
        "🧠 Compute SHAP explanations (adds ~5–15 s for first run)",
        value=True,
    )

    if st.button("🚀 Run Prediction", type="primary"):
        st.session_state.pdf_figures = []
        # FIX 3: reset explainer on each new prediction run so a fresh
        # background dataset is used if X_bg has changed
        st.session_state.shap_explainer = None
        _close_all_figs()

        if len(seq) < 1:
            st.error("Please enter at least one amino acid.")
        else:
            Xp = pd.DataFrame([model_features(seq)])

            taste_raw   = taste_model.predict(Xp)[0]
            taste       = le_taste.inverse_transform([taste_raw])[0]
            taste_idx   = int(taste_raw)
            sol         = le_sol.inverse_transform(sol_model.predict(Xp))[0]
            dock        = dock_model.predict(Xp)[0]
            emoji       = taste_emoji(taste)
            taste_proba = taste_model.predict_proba(Xp)[0]
            sol_proba   = sol_model.predict_proba(Xp)[0]

            sol_color  = "#12b886" if "good" in sol.lower() else "#e67e22"
            dock_color = "#12b886" if dock < -180 else ("#f39c12" if dock < -120 else "#c0392b")
            dock_label = ("Strong binder" if dock < -180 else
                          "Moderate binder" if dock < -120 else "Weak binder")

            st.markdown(f"""
            <div class="card">
              <div class="card-title">
                <span class="live-indicator"></span>ML Prediction Results
              </div>
              <div class="metric-grid">
                <div class="metric-box">
                  <div class="metric-box-label">Primary Taste</div>
                  <div class="metric-box-value">{emoji} {taste}</div>
                  <div class="metric-box-sub">Confidence: {max(taste_proba)*100:.1f}%</div>
                </div>
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

            # Taste probability breakdown — always show all known taste classes.
            st.markdown("#### Taste Class Probabilities")
            full_taste_proba = np.zeros(len(le_taste.classes_), dtype=float)
            full_taste_proba[taste_model.classes_.astype(int)] = taste_proba
            prob_df = pd.DataFrame({
                "Taste Class": le_taste.classes_,
                "Probability": [f"{p*100:.1f}%" for p in full_taste_proba],
            })
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            # Physicochemical table
            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 📌 Physicochemical Properties")
            phys     = physicochemical_features(seq)
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

            # SHAP
            if run_shap:
                render_shap_analysis(
                    taste_model, X_bg, Xp,
                    FEATURE_NAMES, seq, taste, taste_idx,
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
                "Sequence":                 seq[:60] + ("…" if len(seq) > 60 else ""),
                "Length (aa)":              len(seq),
                "Predicted primary taste":  taste,
                "Taste confidence":         f"{max(taste_proba)*100:.1f}%",
                "Predicted solubility":     sol,
                "Docking score":            round(dock, 2),
                "Structure engine":         engine_label,
                "Predicted fold":           classify_fold(ss_result, seq),
                "α-Helix fraction":         f"{ss_result['helix_frac']*100:.1f}%",
                "β-Sheet fraction":         f"{ss_result['sheet_frac']*100:.1f}%",
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
            "📄 Upload CSV (must have a 'peptide' column)", type=["csv"])
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
            if "peptide" not in batch_df.columns:
                st.error("CSV must have a column named 'peptide'.")
                batch_df = None
            else:
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

        tastes, sols, docks, taste_confs, engines, fold_types = [], [], [], [], [], []
        pdb_files = {}

        for i, seq_b in enumerate(batch_seqs):
            try:
                Xr = pd.DataFrame([model_features(seq_b)])
                t  = le_taste.inverse_transform(taste_model.predict(Xr))[0]
                s  = le_sol.inverse_transform(sol_model.predict(Xr))[0]
                d  = round(dock_model.predict(Xr)[0], 2)
                tc = round(max(taste_model.predict_proba(Xr)[0]) * 100, 1)
            except Exception:
                t, s, d, tc = "Error", "Error", None, None
            tastes.append(t); sols.append(s)
            docks.append(d);  taste_confs.append(tc)

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

        batch_df["Predicted Taste"]         = tastes
        batch_df["Predicted Solubility"]    = sols
        batch_df["Predicted Docking Score"] = docks
        batch_df["Taste Confidence (%)"]    = taste_confs
        batch_df["Predicted Fold Type"]     = fold_types
        batch_df["Structure Engine"]        = engines

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

        st.markdown("### 📈 Model Performance Metrics")
        st.markdown(f"""
        <div class="metric-grid" style="margin-bottom:28px;">
          <div class="metric-box">
            <div class="metric-box-label">Taste Accuracy</div>
            <div class="metric-box-value">{metrics['Taste accuracy']*100:.1f}%</div>
            <div class="metric-box-sub">6-class primary taste</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Taste F1</div>
            <div class="metric-box-value">{metrics['Taste F1 (weighted)']:.3f}</div>
            <div class="metric-box-sub">Weighted average</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Solubility Accuracy</div>
            <div class="metric-box-value">{metrics['Solubility accuracy']*100:.1f}%</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Solubility F1</div>
            <div class="metric-box-value">{metrics['Solubility F1']:.3f}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Docking R²</div>
            <div class="metric-box-value">{metrics['Docking R²']:.3f}</div>
          </div>
          <div class="metric-box">
            <div class="metric-box-label">Docking RMSE</div>
            <div class="metric-box-value">{metrics['Docking RMSE']:.2f}</div>
            <div class="metric-box-sub">score units</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Distributions")
        fig_dist = plot_distributions(df_all)
        save_fig(fig_dist, "distributions.png")
        st.image("distributions.png", use_column_width=True)
        show_caption(caption_distributions(df_all))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 PCA Feature Space")
        fig_pca, pca_model = plot_pca(
            X_all, le_taste.transform(df_all["taste"]), le_taste.classes_,
            title="PCA — Peptide Feature Space (by primary taste class)",
        )
        save_fig(fig_pca, "pca_overall.png")
        st.image("pca_overall.png", use_column_width=True)
        show_caption(caption_pca(pca_model, le_taste.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Taste Confusion Matrix")
        taste_preds  = taste_model.predict(X_test)
        fig_cm_taste = plot_confusion(
            yt_test, taste_preds, le_taste.classes_,
            "Taste Confusion Matrix", "Blues")
        save_fig(fig_cm_taste, "confusion_taste.png")
        st.image("confusion_taste.png", use_column_width=True)
        show_caption(caption_confusion_taste(yt_test, taste_preds, le_taste.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Solubility Confusion Matrix")
        sol_preds  = sol_model.predict(X_test)
        fig_cm_sol = plot_confusion(
            ys_test, sol_preds, le_sol.classes_,
            "Solubility Confusion Matrix", "Greens")
        save_fig(fig_cm_sol, "confusion_solubility.png")
        st.image("confusion_solubility.png", use_column_width=True)
        show_caption(caption_confusion_sol(ys_test, sol_preds, le_sol.classes_))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Feature Importance (Gini)")
        fig_imp = plot_feature_importance(taste_model, FEATURE_NAMES, top_n=20)
        save_fig(fig_imp, "feature_importance_taste.png")
        st.image("feature_importance_taste.png", use_column_width=True)
        show_caption(caption_feature_importance(taste_model, FEATURE_NAMES, top_n=20))

        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Docking Score: True vs Predicted")
        dock_preds = dock_model.predict(X_test)
        fig_dock   = plot_docking(yd_test, dock_preds)
        save_fig(fig_dock, "docking_scatter.png")
        st.image("docking_scatter.png", use_column_width=True)
        show_caption(caption_docking(yd_test, dock_preds))

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
&copy; {date.today().year} &nbsp; <b>PepTastePredictor v2</b><br>
6-class primary-taste model (Bitter · Salty · Sour · Sweet · Umami · Neutral) · SHAP interpretability · Hybrid Structural Engine<br>
Structure priority: RCSB PDB → Remote ESMFold → Chou-Fasman Folder → PeptideBuilder<br>
For academic and research use only
</div>
""", unsafe_allow_html=True)






