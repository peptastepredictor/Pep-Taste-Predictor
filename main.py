# ==========================================================
# PepTastePredictor — app.py  (Code 3)
# Complete integration of Code 1 + Code 2
# Cloud-compatible | No ColabFold / ESMFold / JAX / GPU
# ==========================================================

# ==========================================================
# SECTION 1 - IMPORTS
# ==========================================================

import os
import re
import io
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
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import py3Dmol

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder
from Bio.PDB.SASA import ShrakeRupley

import PeptideBuilder
from PeptideBuilder import Geometry

from sklearn.ensemble import ExtraTreesClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_squared_error, r2_score,
    confusion_matrix,
)
from sklearn.decomposition import PCA

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph,
    Image as RLImage, Spacer, Table, TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors


# ==========================================================
# SECTION 2 - GLOBAL CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PATH    = "AIML.xlsx"
PREDICTIONS_DIR = Path("predictions")
PREDICTIONS_DIR.mkdir(exist_ok=True)

AA = "ACDEFGHIKLMNPQRSTVWY"
ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

TASTE_EMOJI = {
    "Bitter": "😖", "Sweet": "😋", "Salty": "🧂",
    "Sour": "😮‍💨", "Umami": "🤤",
}

AA_COLORS = {
    "A": "#80b1d3", "C": "#fdb462", "D": "#fb8072", "E": "#fb8072",
    "F": "#b3de69", "G": "#d9d9d9", "H": "#bebada", "I": "#80b1d3",
    "K": "#8dd3c7", "L": "#80b1d3", "M": "#fdb462", "N": "#fccde5",
    "P": "#ffffb3", "Q": "#fccde5", "R": "#8dd3c7", "S": "#fccde5",
    "T": "#fccde5", "V": "#80b1d3", "W": "#b3de69", "Y": "#b3de69",
}


# ==========================================================
# SECTION 3 - FRONTEND STYLING
# ==========================================================

st.markdown("""
<style>
/* Universal theme-aware text */
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
div[data-testid="stTextArea"] label,
div[data-testid="stTextArea"] label p { color: var(--text-color) !important; }
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
div[data-testid="stTabs"] button p { color: var(--text-color) !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #1f3c88 0%, #0b7285 60%, #12b886 100%);
    padding: 40px 44px; border-radius: 20px; margin-bottom: 36px;
    box-shadow: 0 8px 32px rgba(31,60,136,0.18);
}
.hero h1 { font-size: 2.4rem !important; font-weight: 800 !important;
    margin-bottom: 10px; color: #ffffff !important; letter-spacing: -0.5px; }
.hero p { font-size: 1.08rem !important; line-height: 1.8; color: #dce8ff !important; margin: 0; }

/* Cards */
.card { border: 1px solid rgba(128,128,180,0.3); padding: 28px 32px;
    border-radius: 16px; margin-bottom: 28px; background: rgba(128,128,180,0.05);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.live-card { border: 2px solid rgba(26,143,209,0.35); padding: 22px 28px;
    border-radius: 14px; margin-bottom: 18px; background: rgba(26,143,209,0.05); }
.ext-card { border: 2px solid rgba(18,184,134,0.4); padding: 26px 32px;
    border-radius: 16px; margin-bottom: 24px; background: rgba(18,184,134,0.05); }
.quality-card { border: 2px solid rgba(74,111,165,0.4); padding: 24px 28px;
    border-radius: 16px; margin-bottom: 24px; background: rgba(74,111,165,0.06); }

/* Steps */
.step-row { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 14px; }
.step-num { min-width: 32px; height: 32px; border-radius: 50%; background: #1a8fd1;
    color: #fff !important; font-weight: 800; font-size: 14px;
    display: flex; align-items: center; justify-content: center; }
.step-text { font-size: 15px; line-height: 1.6; color: var(--text-color) !important; padding-top: 4px; }

/* Metrics */
.metric-label { font-size: 12px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.6;
    margin-bottom: 4px; margin-top: 18px; color: var(--text-color) !important; }
.metric-label:first-child { margin-top: 0; }
.metric-value { font-size: 24px !important; font-weight: 800 !important;
    color: #1a8fd1 !important; margin-bottom: 2px; }

/* Progress bars */
.progress-bar-wrap { background: rgba(128,128,180,0.15); border-radius: 8px;
    height: 10px; margin: 6px 0 16px 0; overflow: hidden; }
.progress-bar-fill { height: 10px; border-radius: 8px; }

/* Badges */
.aa-badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 13px; font-weight: 700; margin: 3px 2px;
    background: rgba(26,143,209,0.15); color: #1a8fd1 !important;
    border: 1px solid rgba(26,143,209,0.3); }
.seq-counter { font-size: 13px; font-weight: 600; opacity: 0.65;
    margin-bottom: 6px; color: var(--text-color) !important; }

/* Captions */
.graph-caption { border-left: 5px solid #4a6fa5; border-radius: 0 10px 10px 0;
    padding: 18px 24px; margin-top: 12px; margin-bottom: 40px;
    font-size: 15px !important; line-height: 1.9; background: rgba(74,111,165,0.08);
    color: var(--text-color) !important; }
.graph-caption strong { font-weight: 700; color: var(--text-color) !important; }
.graph-caption em { font-style: italic; color: var(--text-color) !important; opacity: 0.85; }

/* Layout helpers */
.section-gap { margin-top: 40px; margin-bottom: 6px; }
.structure-badge { display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-size: 13px; font-weight: 700; margin-bottom: 12px; }

/* Footer */
.footer { text-align: center; font-size: 14px !important; padding: 44px 20px 20px;
    margin-top: 60px; line-height: 2.2;
    border-top: 1px solid rgba(128,128,180,0.25);
    color: var(--text-color) !important; opacity: 0.7; }

/* Live indicator */
@keyframes pulse { 0%{opacity:1;} 50%{opacity:0.5;} 100%{opacity:1;} }
.live-indicator { display: inline-block; width: 8px; height: 8px;
    background: #12b886; border-radius: 50%; animation: pulse 1.5s infinite; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 4 - SIDEBAR
# ==========================================================

if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=120)

st.sidebar.markdown("### 🧬 PepTastePredictor v3")
st.sidebar.write("AI-driven peptide analysis platform")
st.sidebar.write("• Taste prediction")
st.sidebar.write("• Solubility prediction")
st.sidebar.write("• Docking estimation")
st.sidebar.write("• Structural bioinformatics")
st.sidebar.write("• Batch screening")
st.sidebar.write("• Advanced structural analysis")

st.sidebar.markdown("---")
st.sidebar.markdown("**🌐 External Structure Servers**")
st.sidebar.markdown(
    '<div style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;'
    'border-radius:20px;font-size:13px;font-weight:600;background:rgba(18,184,134,0.12);'
    'border:1px solid rgba(18,184,134,0.35);color:#12b886;margin-bottom:6px;">'
    '🌿 ESM Atlas</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;'
    'border-radius:20px;font-size:13px;font-weight:600;background:rgba(26,143,209,0.12);'
    'border:1px solid rgba(26,143,209,0.35);color:#1a8fd1;">'
    '🔬 AlphaFold Server</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("**🏗️ Local Builder**")
st.sidebar.markdown(
    '<div style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;'
    'border-radius:20px;font-size:13px;font-weight:600;background:rgba(255,165,0,0.12);'
    'border:1px solid rgba(255,165,0,0.35);color:#e67e22;">'
    '⚙️ PeptideBuilder</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Generate structures externally or locally, then run full analysis.")
st.sidebar.info("For academic & educational use only")


# ==========================================================
# SECTION 5 - SESSION STATE
# ==========================================================

defaults = {
    "initialized":     True,
    "pdb_text":        None,
    "pdb_source":      None,   # "uploaded" | "peptidebuilder"
    "last_prediction": {},
    "show_analytics":  False,
    "pdf_figures":     [],
    "live_seq":        "",
    "current_mode":    None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==========================================================
# SECTION 6 - UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename: str):
    fig.savefig(filename, dpi=180, bbox_inches="tight")
    if filename not in st.session_state.pdf_figures:
        st.session_state.pdf_figures.append(filename)


def clean_sequence(seq) -> str:
    if not isinstance(seq, str):
        return ""
    lines = seq.splitlines()
    lines = [l for l in lines if not l.strip().startswith(">")]
    seq = "".join(lines)
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


def model_features(seq: str) -> dict:
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
            "mw": 111.0, "pI": 7.0,
            "aromaticity": 1.0 if seq in "FWY" else 0.0,
            "instability": 0.0,
            "gravy": KD_SCALE.get(seq, 0.0),
            "charge": 1.0 if seq in "KRH" else (-1.0 if seq in "DE" else 0.0),
        })
    for aa in AA:
        features[f"AA_{aa}"] = seq.count(aa) / L
    denom = max(L - 1, 1)
    for dp in ALL_DIPEPTIDES:
        features[f"DPC_{dp}"] = seq.count(dp) / denom
    groups = {
        "hydrophobic": "AILMFWV", "polar": "STNQ",
        "charged": "DEKRH", "aromatic": "FWY", "tiny": "AGSC",
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
        "Note":        "Extended analysis requires ≥2 residues",
    }


def composition_features(seq: str) -> dict:
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
    rare   = set(counts[counts < 5].index)
    def _map(t):
        if t in rare:
            for base in ["Bitter", "Sweet", "Salty", "Sour", "Umami"]:
                if base.lower() in t.lower():
                    return base
            return "Bitter"
        return t
    return taste_series.apply(_map)


def prettify_feature(name: str) -> str:
    if name.startswith("DPC_"):
        return f"Dipeptide {name[4:]}"
    if name.startswith("AA_"):
        return f"Amino acid: {name[3:]}"
    return name.replace("_", " ").title()


def gravy_score(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(KD_SCALE.get(a, 0) for a in seq) / len(seq)


def taste_emoji(taste: str) -> str:
    for k, v in TASTE_EMOJI.items():
        if k.lower() in taste.lower():
            return v
    return "🧬"


def show_caption(html_text: str):
    st.markdown(f'<div class="graph-caption">{html_text}</div>', unsafe_allow_html=True)


# ==========================================================
# SECTION 7 - MATPLOTLIB THEME
# ==========================================================

def _is_dark_mode() -> bool:
    try:
        return st.get_option("theme.base") == "dark"
    except Exception:
        return False


def get_plot_colors() -> dict:
    if _is_dark_mode():
        return {
            "fig_bg": "#1a1d2e", "ax_bg": "#1e2140", "text": "#e8edf8",
            "grid": "#2e3560", "accent1": "#5c7cfa", "accent2": "#748ffc",
            "accent3": "#4dd0e1", "red": "#ff6b6b", "orange": "#ffa94d",
            "tick": "#c5cff0", "green": "#51cf66",
        }
    return {
        "fig_bg": "#f8f9fc", "ax_bg": "#ffffff", "text": "#1a1d2e",
        "grid": "#d0d5e8", "accent1": "#1a56db", "accent2": "#4361ee",
        "accent3": "#0b7285", "red": "#c0392b", "orange": "#e67e22",
        "tick": "#4a5170", "green": "#12b886",
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
# SECTION 8 - LIVE SEQUENCE PREVIEW
# ==========================================================

def render_live_preview(seq: str):
    if not seq:
        return
    L  = len(seq)
    gv = gravy_score(seq)
    c  = Counter(seq)
    hydro_pct  = 100 * sum(c[a] for a in "AILMFWV") / L
    charge_pct = 100 * sum(c[a] for a in "DEKRH") / L
    arom_pct   = 100 * sum(c[a] for a in "FWY") / L
    hydro_color  = "#1a8fd1" if gv > 0 else "#e67e22"
    charge_label = "Positive" if sum(c[a] for a in "KRH") > sum(c[a] for a in "DE") else "Negative"
    mw_est = L * 110.0
    instab = "Stable" if L >= 2 and ProteinAnalysis(seq).instability_index() < 40 else "Unstable"
    try:
        pi_val = f"{ProteinAnalysis(seq).isoelectric_point():.1f}" if L >= 2 else "N/A"
    except Exception:
        pi_val = "N/A"

    st.markdown(f"""
    <div class="live-card">
      <span style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;letter-spacing:0.08em;">
        <span class="live-indicator"></span>Live Sequence Analysis
      </span>
      <div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:14px;">
        <div><div class="metric-label" style="margin-top:0;">Length</div>
             <div class="metric-value">{L} aa</div></div>
        <div><div class="metric-label" style="margin-top:0;">GRAVY</div>
             <div class="metric-value" style="color:{hydro_color} !important;">{gv:+.2f}</div>
             <div style="font-size:11px;opacity:0.6;">{'Hydrophobic' if gv>0 else 'Hydrophilic'}</div></div>
        <div><div class="metric-label" style="margin-top:0;">Charge</div>
             <div class="metric-value">{charge_label}</div></div>
        <div><div class="metric-label" style="margin-top:0;">Aromatic</div>
             <div class="metric-value">{arom_pct:.0f}%</div></div>
        <div><div class="metric-label" style="margin-top:0;">Est. MW</div>
             <div class="metric-value">{mw_est/1000:.1f} kDa</div></div>
        <div><div class="metric-label" style="margin-top:0;">pI</div>
             <div class="metric-value">{pi_val}</div></div>
        <div><div class="metric-label" style="margin-top:0;">Stability</div>
             <div class="metric-value" style="color:{'#12b886' if instab=='Stable' else '#e67e22'} !important;">{instab}</div></div>
      </div>
      <div style="margin-top:14px;">
        <div class="metric-label" style="margin-top:0;">Hydrophobic residues ({hydro_pct:.0f}%)</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar-fill" style="width:{min(hydro_pct,100):.0f}%;background:#1a8fd1;"></div></div>
        <div class="metric-label">Charged residues ({charge_pct:.0f}%)</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar-fill" style="width:{min(charge_pct,100):.0f}%;background:#e67e22;"></div></div>
        <div class="metric-label">Aromatic residues ({arom_pct:.0f}%)</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar-fill" style="width:{min(arom_pct,100):.0f}%;background:#b3de69;"></div></div>
      </div>
      <div style="margin-top:10px;">
        {''.join(f'<span class="aa-badge">{aa}</span>' for aa in seq[:40])}
        {'<span style="opacity:0.5;font-size:12px;"> +more</span>' if L > 40 else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# SECTION 9 - PDB HELPER UTILITIES
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


def _extract_plddt_from_pdb(pdb_text: str) -> list:
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
    if score >= 90:
        return "Very High", "#12b886"
    if score >= 70:
        return "High", "#1a8fd1"
    if score >= 50:
        return "Medium", "#f39c12"
    return "Low", "#c0392b"


# ==========================================================
# SECTION 10 - STRUCTURE GENERATION
# ==========================================================

def build_peptide_pdb(seq: str) -> str:
    """Generate a backbone PDB from sequence using PeptideBuilder."""
    try:
        structure = PeptideBuilder.initialize_res(seq[0])
        for aa in seq[1:]:
            try:
                PeptideBuilder.add_residue(structure, Geometry.geometry(aa))
            except Exception:
                pass
        io = PDBIO()
        io.set_structure(structure)
        out_path = "predicted_peptide.pdb"
        io.save(out_path)
        with open(out_path) as f:
            return f.read()
    except Exception as e:
        return ""


# ==========================================================
# SECTION 11 - STRUCTURAL ANALYSIS FUNCTIONS
# ==========================================================

def show_structure(pdb_text: str):
    view = py3Dmol.view(width=800, height=480)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addSurface(py3Dmol.SAS, {"opacity": 0.07})
    view.zoomTo()
    view.spin(False)
    return view


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
    if not pdb_text or not pdb_text.strip():
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
    if not pdb_text or not pdb_text.strip():
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


def run_dssp_fallback(pdb_text: str) -> dict:
    """
    Lightweight DSSP-like secondary structure assignment
    based on backbone geometry (no DSSP binary required).
    Uses phi/psi angles from PPBuilder.
    """
    fallback = {"helix": 0.0, "sheet": 0.0, "coil": 100.0, "raw": []}
    phi_psi  = ramachandran(pdb_text)
    if not phi_psi:
        return fallback
    raw   = []
    total = len(phi_psi)
    for phi, psi in phi_psi:
        if -180 <= phi <= -45 and -75 <= psi <= -15:
            raw.append("H")
        elif -180 <= phi <= -45 and (90 <= psi <= 180 or -180 <= psi <= -150):
            raw.append("E")
        else:
            raw.append("C")
    n_h = raw.count("H")
    n_e = raw.count("E")
    n_c = raw.count("C")
    return {
        "helix": round(n_h / total * 100, 1),
        "sheet": round(n_e / total * 100, 1),
        "coil":  round(n_c / total * 100, 1),
        "raw":   raw,
    }


def radius_of_gyration(pdb_text: str):
    if not pdb_text:
        return None
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        coords    = np.array([a.get_vector().get_array() for a in structure.get_atoms()])
        if len(coords) == 0:
            return None
        centroid = coords.mean(axis=0)
        return float(np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()))
    except Exception:
        return None
    finally:
        _unlink(tmp)


def compute_sasa(pdb_text: str):
    if not pdb_text:
        return None
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        sr        = ShrakeRupley()
        sr.compute(structure, level="S")
        return round(structure.sasa, 2)
    except Exception:
        return None
    finally:
        _unlink(tmp)


def per_residue_sasa(pdb_text: str) -> dict:
    if not pdb_text:
        return {}
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        sr        = ShrakeRupley()
        sr.compute(structure, level="R")
        result = {}
        for res in structure.get_residues():
            key = f"{res.get_resname()}{res.get_id()[1]}"
            try:
                result[key] = round(res.sasa, 2)
            except Exception:
                pass
        return result
    except Exception:
        return {}
    finally:
        _unlink(tmp)


def count_hbonds(pdb_text: str) -> int:
    if not pdb_text:
        return 0
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
        residues  = list(structure.get_residues())
        donors, acceptors = [], []
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
        _unlink(tmp)


def count_disulfide_bonds(pdb_text: str) -> int:
    if not pdb_text:
        return 0
    tmp = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp)
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
        _unlink(tmp)


def contact_map(pdb_text: str, threshold: float = 8.0) -> np.ndarray:
    dm = ca_distance_map(pdb_text)
    return (dm < threshold).astype(int)


def residue_exposure(pdb_text: str) -> dict:
    """Classify each residue as buried/exposed based on per-residue SASA."""
    sasa_data = per_residue_sasa(pdb_text)
    if not sasa_data:
        return {}
    vals = list(sasa_data.values())
    threshold = np.median(vals) if vals else 30.0
    return {k: ("Exposed" if v >= threshold else "Buried") for k, v in sasa_data.items()}


def structural_quality_score(
    phi_psi: list,
    dssp: dict,
    sasa_total,
    plddt_vals: list = None,
    n_residues: int = 1,
) -> tuple:
    """
    Compute an overall structural quality score (0–100) and label.
    Returns (score, label, breakdown_dict).
    """
    score = 0
    breakdown = {}

    # Ramachandran quality (max 30 pts)
    if phi_psi:
        total = len(phi_psi)
        allowed = sum(
            1 for p, s in phi_psi
            if (-180 <= p <= -45 and -75 <= s <= -15)
            or (-180 <= p <= -45 and 90 <= s <= 180)
            or (45 <= p <= 90 and 0 <= s <= 90)
        )
        rama_pct = allowed / total * 100
        rama_pts = rama_pct * 0.30
    else:
        rama_pct = 0.0
        rama_pts = 15  # neutral if can't compute
    score += rama_pts
    breakdown["Ramachandran"] = round(rama_pts, 1)

    # DSSP quality (max 30 pts) — reward structured content
    struct_pct = dssp.get("helix", 0) + dssp.get("sheet", 0)
    dssp_pts   = min(struct_pct * 0.30, 30)
    score     += dssp_pts
    breakdown["Secondary Structure"] = round(dssp_pts, 1)

    # SASA quality (max 20 pts)
    if sasa_total is not None and n_residues > 0:
        avg_sasa = sasa_total / n_residues
        sasa_pts = max(0, 20 - abs(avg_sasa - 80) * 0.15)
    else:
        sasa_pts = 10  # neutral
    score += sasa_pts
    breakdown["SASA Distribution"] = round(sasa_pts, 1)

    # pLDDT quality (max 20 pts)
    if plddt_vals and len(plddt_vals) > 0:
        mean_pl  = np.mean(plddt_vals)
        plddt_pts = mean_pl * 0.20
    else:
        plddt_pts = 10  # neutral
    score += plddt_pts
    breakdown["pLDDT Confidence"] = round(plddt_pts, 1)

    score = min(100, score)
    if score >= 80:
        label = "Excellent"
    elif score >= 60:
        label = "Good"
    elif score >= 40:
        label = "Moderate"
    else:
        label = "Poor"

    return round(score, 1), label, breakdown


# ==========================================================
# SECTION 12 - DYNAMIC CAPTIONS
# ==========================================================

def caption_distributions(df):
    lengths    = [len(s) for s in df["peptide"]]
    grav       = [gravy_score(s) for s in df["peptide"]]
    dom_taste  = df["taste"].value_counts().idxmax()
    dom_count  = df["taste"].value_counts().max()
    n_classes  = df["taste"].nunique()
    mean_grav  = np.mean(grav)
    glabel     = ("slightly hydrophobic" if mean_grav > 0.2 else
                  "slightly hydrophilic" if mean_grav < -0.2 else "amphipathic")
    return (
        f"<strong>Length (left):</strong> {int(np.min(lengths))}–{int(np.max(lengths))} aa, "
        f"mean {np.mean(lengths):.1f} aa. Short peptides dominate.<br><br>"
        f"<strong>Taste classes (centre):</strong> &ldquo;{dom_taste}&rdquo; is the most common "
        f"({dom_count} of {len(df)} across {n_classes} classes). Balanced class weighting is applied.<br><br>"
        f"<strong>GRAVY (right):</strong> Mean {mean_grav:.2f} — dataset is <strong>{glabel}</strong>."
    )


def caption_pca(pca_model, class_names):
    v1, v2 = pca_model.explained_variance_ratio_[:2] * 100
    return (
        f"Each dot is one peptide compressed from hundreds of features to 2 dimensions.<br><br>"
        f"<strong>PC1</strong> = {v1:.1f}% variance &nbsp; | &nbsp; "
        f"<strong>PC2</strong> = {v2:.1f}% variance &nbsp; | &nbsp; "
        f"<strong>Total</strong> = {v1+v2:.1f}%.<br><br>"
        f"Tight, separated clusters → reliable class distinction. Overlapping → expect confusion."
    )


def caption_confusion_taste(y_true, y_pred, class_names):
    acc     = accuracy_score(y_true, y_pred) * 100
    cm      = confusion_matrix(y_true, y_pred)
    cp      = cm.astype(float)
    np.fill_diagonal(cp, 0)
    idx     = np.unravel_index(np.argmax(cp), cp.shape)
    pca     = cm.diagonal() / cm.sum(axis=1)
    return (
        f"Taste model: <strong>{acc:.1f}% overall accuracy</strong>.<br><br>"
        f"Worst confusion: &ldquo;{class_names[idx[0]]}&rdquo; → &ldquo;{class_names[idx[1]]}&rdquo; "
        f"({int(cp[idx])} times).<br>"
        f"Best class: &ldquo;{class_names[np.argmax(pca)]}&rdquo; | "
        f"Hardest: &ldquo;{class_names[np.argmin(pca)]}&rdquo;."
    )


def caption_confusion_sol(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred) * 100
    cm  = confusion_matrix(y_true, y_pred)
    cp  = cm.astype(float)
    np.fill_diagonal(cp, 0)
    idx = np.unravel_index(np.argmax(cp), cp.shape)
    return (
        f"Solubility model: <strong>{acc:.1f}% accuracy</strong>.<br><br>"
        f"Most common error: &ldquo;{class_names[idx[0]]}&rdquo; → "
        f"&ldquo;{class_names[idx[1]]}&rdquo; ({int(cp[idx])} times). "
        f"Borderline hydrophobicity drives most errors."
    )


def caption_feature_importance(model, feature_names, top_n=20):
    imp  = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
    imp  = imp.sort_values("Importance", ascending=False).head(top_n)
    top3 = [(prettify_feature(r["Feature"]), r["Importance"]) for _, r in imp.head(3).iterrows()]
    n_d  = sum(1 for f in imp["Feature"] if f.startswith("DPC_"))
    n_a  = sum(1 for f in imp["Feature"] if f.startswith("AA_"))
    note = ("Dipeptide context dominates — sequential patterns matter." if n_d > n_a
            else "Single amino acid composition is the stronger predictor.")
    return (
        f"Top {top_n} features driving taste predictions.<br><br>"
        + "".join(f"<strong>#{i+1} — {n}</strong> (score: {s:.4f})<br>" for i, (n,s) in enumerate(top3))
        + f"<br>{n_d} DPC and {n_a} AA features in top {top_n}. {note}"
    )


def caption_docking(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qual = "strong" if r2 >= 0.75 else ("moderate" if r2 >= 0.5 else "weak")
    return (
        f"Test-set docking predictions. Red dashed = perfect prediction line.<br><br>"
        f"<strong>R² = {r2:.3f}</strong> ({qual} fit) — explains {r2*100:.1f}% of variance.<br>"
        f"<strong>RMSE = {rmse:.2f} kcal/mol</strong> — typical per-peptide error.<br>"
        f"True scores: {y_true.min():.2f} to {y_true.max():.2f} kcal/mol."
    )


def caption_ramachandran(phi_psi, seq=""):
    if not phi_psi:
        return "No φ/ψ angles extracted — peptide needs ≥3 residues."
    n      = len(phi_psi)
    n_h    = sum(1 for p, s in phi_psi if -180<=p<=-45 and -75<=s<=-15)
    n_e    = sum(1 for p, s in phi_psi if -180<=p<=-45 and 90<=s<=180)
    n_dis  = n - n_h - n_e
    dom    = "α-helix" if n_h >= n_e else "β-sheet"
    return (
        f"Backbone torsion angles{f' for <strong>{len(seq)}-residue peptide</strong>' if seq else ''}.<br><br>"
        f"<strong>α-helix region:</strong> {n_h/n*100:.0f}% | "
        f"<strong>β-sheet region:</strong> {n_e/n*100:.0f}% | "
        f"<strong>Outside allowed:</strong> {n_dis/n*100:.0f}%<br><br>"
        f"Dominant backbone character: <strong>{dom}</strong>."
    )


def caption_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if n < 2:
        return "Distance map unavailable — fewer than 2 Cα atoms."
    mask   = ~np.eye(n, dtype=bool)
    od     = dist_matrix[mask]
    lr     = sum(1 for i in range(n) for j in range(n) if abs(i-j)>3 and dist_matrix[i,j]<8.0)
    fold   = "folds back on itself" if lr > 0 else "extended / linear conformation"
    return (
        f"Cα–Cα distances — darker = spatially closer.<br><br>"
        f"Range: {od.min():.1f}–{od.max():.1f} Å | "
        f"Long-range contacts: <strong>{lr}</strong> — suggests <strong>{fold}</strong>."
    )


def caption_contact_map(cm_matrix, threshold=8.0):
    n        = cm_matrix.shape[0]
    mask     = ~np.eye(n, dtype=bool)
    contacts = cm_matrix[mask].sum() // 2
    density  = contacts / max((n * (n - 1)) // 2, 1) * 100
    lr = sum(1 for i in range(n) for j in range(n) if abs(i-j)>3 and cm_matrix[i,j]==1) // 2
    label = "compact / well-folded" if density > 30 else "extended / loosely packed"
    return (
        f"Binary contact map (threshold = {threshold} Å).<br><br>"
        f"Total contacts: <strong>{contacts}</strong> | "
        f"Contact density: <strong>{density:.1f}%</strong> → <strong>{label}</strong>.<br>"
        f"Long-range contacts (|i−j|>3): <strong>{lr}</strong>."
    )


def caption_sasa(sasa_total, pr_sasa: dict, n_residues: int):
    if sasa_total is None:
        return "SASA could not be computed for this structure."
    avg  = sasa_total / max(n_residues, 1)
    median_sasa = np.median(list(pr_sasa.values())) if pr_sasa else 0
    exp = sum(1 for v in pr_sasa.values() if v >= median_sasa)
    bur = len(pr_sasa) - exp
    return (
        f"Solvent-accessible surface area analysis.<br><br>"
        f"<strong>Total SASA:</strong> {sasa_total:.1f} Å² | "
        f"<strong>Average per residue:</strong> {avg:.1f} Å²<br>"
        f"<strong>Exposed residues:</strong> {exp} | <strong>Buried:</strong> {bur}<br><br>"
        f"Higher SASA → more solvent-exposed structure → typically higher solubility."
    )


def caption_hydrophobicity(seq: str):
    gv = gravy_score(seq)
    hyd_aa = [(i+1, a, KD_SCALE.get(a,0)) for i, a in enumerate(seq)]
    max_hyd = max(hyd_aa, key=lambda x: x[2])
    min_hyd = min(hyd_aa, key=lambda x: x[2])
    return (
        f"Per-residue hydrophobicity (Kyte–Doolittle scale).<br><br>"
        f"<strong>Overall GRAVY:</strong> {gv:+.2f} "
        f"({'hydrophobic' if gv>0 else 'hydrophilic'} character)<br>"
        f"<strong>Most hydrophobic:</strong> {max_hyd[1]}{max_hyd[0]} ({max_hyd[2]:+.1f})<br>"
        f"<strong>Most hydrophilic:</strong> {min_hyd[1]}{min_hyd[0]} ({min_hyd[2]:+.1f})"
    )


def caption_charge(seq: str):
    pos = sum(1 for a in seq if a in "KRH")
    neg = sum(1 for a in seq if a in "DE")
    net = pos - neg
    return (
        f"Per-residue charge distribution at pH 7.<br><br>"
        f"<strong>Positive residues (K/R/H):</strong> {pos} | "
        f"<strong>Negative (D/E):</strong> {neg} | "
        f"<strong>Net charge:</strong> {net:+d}<br><br>"
        f"{'Cationic — may interact with negatively charged membranes.' if net>0 else 'Anionic — repelled by negatively charged surfaces.' if net<0 else 'Near-neutral charge balance.'}"
    )


# ==========================================================
# SECTION 13 - PLOT FUNCTIONS
# ==========================================================

def plot_plddt(plddt_vals: list, seq: str = ""):
    C   = get_plot_colors()
    n   = len(plddt_vals)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.2), 4))
    apply_plot_style(fig, [ax])
    bar_colors = ["#12b886" if v>=90 else "#1a8fd1" if v>=70 else
                  "#f39c12" if v>=50 else "#c0392b" for v in plddt_vals]
    ax.bar(range(n), plddt_vals, color=bar_colors, width=0.85)
    for thresh, col, lbl in [(90,"#12b886","Very High"), (70,"#1a8fd1","High"), (50,"#f39c12","Medium")]:
        ax.axhline(thresh, color=col, linestyle="--", lw=1.2, alpha=0.7, label=f"{lbl} (≥{thresh})")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue Index", fontsize=11, labelpad=8)
    ax.set_ylabel("pLDDT", fontsize=11, labelpad=8)
    ax.set_title("Per-Residue pLDDT Confidence", fontsize=13, fontweight="bold", pad=12)
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    leg = ax.legend(fontsize=8, loc="lower right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_ramachandran(phi_psi: list):
    C   = get_plot_colors()
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.fill([-180,-180,-45,-45,-180], [-75,-45,-45,-75,-75], color="#4CAF50", alpha=0.25, label="α-helix")
    ax.fill([-180,-180,-90,-90,-180], [90,180,180,90,90],    color="#2196F3", alpha=0.25, label="β-sheet")
    ax.fill([45,45,90,90,45],         [0,90,90,0,0],         color="#FF9800", alpha=0.2,  label="L-helix")
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=55, color=C["red"], zorder=5, edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.axvline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi φ (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("Psi ψ (°)", fontsize=12, labelpad=10)
    ax.set_title("Ramachandran Plot", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, loc="upper right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    return fig


def plot_distance_map(dist_matrix: np.ndarray, seq: str = ""):
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


def plot_contact_map(cm_matrix: np.ndarray, seq: str = "", threshold: float = 8.0):
    C = get_plot_colors()
    n = cm_matrix.shape[0]
    if seq and len(seq) == n:
        labels = [f"{aa}{i+1}" for i, aa in enumerate(seq)]
    else:
        labels = [str(i+1) for i in range(n)]
    tick_step   = max(1, n // 15)
    show_labels = [labels[i] if i % tick_step == 0 else "" for i in range(n)]
    size        = max(5, n * 0.3 + 2)
    fig, ax     = plt.subplots(figsize=(size, size))
    apply_plot_style(fig, [ax])
    ax.imshow(cm_matrix, cmap="Blues", origin="upper", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels([show_labels[i] for i in range(0, n, tick_step)],
                       rotation=45, ha="right", fontsize=8, color=C["tick"])
    ax.set_yticks(range(0, n, tick_step))
    ax.set_yticklabels([show_labels[i] for i in range(0, n, tick_step)],
                       fontsize=8, color=C["tick"])
    ax.set_title(f"Contact Map (threshold {threshold} Å)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Residue", fontsize=12, labelpad=10)
    ax.set_ylabel("Residue", fontsize=12, labelpad=10)
    plt.tight_layout()
    return fig


def plot_dssp_pie(dssp: dict):
    C   = get_plot_colors()
    labels  = ["α-Helix", "β-Sheet", "Coil/Other"]
    sizes   = [dssp.get("helix", 0), dssp.get("sheet", 0), dssp.get("coil", 100)]
    colors  = ["#4CAF50", "#2196F3", "#9E9E9E"]
    fig, ax = plt.subplots(figsize=(5, 5))
    apply_plot_style(fig, [ax])
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"color": C["text"], "fontsize": 12},
    )
    for at in autotexts:
        at.set_color(C["text"])
        at.set_fontsize(11)
    ax.set_title("Secondary Structure Composition", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_sasa_per_residue(pr_sasa: dict, seq: str = ""):
    C = get_plot_colors()
    if not pr_sasa:
        return None
    keys = list(pr_sasa.keys())
    vals = list(pr_sasa.values())
    median_val = np.median(vals)
    bar_colors = ["#1a8fd1" if v >= median_val else "#e67e22" for v in vals]
    fig, ax    = plt.subplots(figsize=(max(8, len(keys) * 0.35), 4))
    apply_plot_style(fig, [ax])
    ax.bar(range(len(keys)), vals, color=bar_colors, width=0.85)
    ax.axhline(median_val, color=C["red"], linestyle="--", lw=1.5, label=f"Median = {median_val:.1f} Å²")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8, color=C["tick"])
    ax.set_xlabel("Residue", fontsize=11, labelpad=8)
    ax.set_ylabel("SASA (Å²)", fontsize=11, labelpad=8)
    ax.set_title("Per-Residue SASA", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_hydrophobicity(seq: str):
    C    = get_plot_colors()
    vals = [KD_SCALE.get(a, 0) for a in seq]
    win  = min(5, len(seq))
    if len(seq) >= win:
        smooth = [np.mean(vals[max(0, i - win//2):i + win//2 + 1]) for i in range(len(seq))]
    else:
        smooth = vals
    fig, ax = plt.subplots(figsize=(max(8, len(seq) * 0.25), 4))
    apply_plot_style(fig, [ax])
    bar_colors = [C["accent1"] if v > 0 else C["red"] for v in vals]
    ax.bar(range(len(seq)), vals, color=bar_colors, alpha=0.6, width=0.85)
    ax.plot(range(len(seq)), smooth, color=C["orange"], lw=2.5, label=f"Sliding avg (w={win})")
    ax.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels(list(seq), fontsize=9, color=C["tick"])
    ax.set_xlabel("Residue", fontsize=11, labelpad=8)
    ax.set_ylabel("Hydrophobicity (KD)", fontsize=11, labelpad=8)
    ax.set_title("Per-Residue Hydrophobicity", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_charge_distribution(seq: str):
    C    = get_plot_colors()
    charges = []
    for a in seq:
        if a in "KRH":
            charges.append(1)
        elif a in "DE":
            charges.append(-1)
        else:
            charges.append(0)
    cumulative = np.cumsum(charges)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(seq) * 0.25), 6), sharex=True)
    apply_plot_style(fig, [ax1, ax2])
    bar_colors = [C["green"] if c > 0 else C["red"] if c < 0 else C["grid"] for c in charges]
    ax1.bar(range(len(seq)), charges, color=bar_colors, width=0.85)
    ax1.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax1.set_ylabel("Charge", fontsize=11)
    ax1.set_title("Per-Residue Charge Distribution", fontsize=13, fontweight="bold", pad=12)
    ax2.plot(range(len(seq)), cumulative, color=C["accent1"], lw=2.5)
    ax2.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax2.set_ylabel("Cumulative Charge", fontsize=11)
    ax2.set_xlabel("Residue", fontsize=11)
    ax2.set_xticks(range(len(seq)))
    ax2.set_xticklabels(list(seq), fontsize=9, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_aa_composition(seq: str):
    C   = get_plot_colors()
    cnt = Counter(seq)
    L   = len(seq)
    sorted_aa  = sorted(AA)
    freqs      = [cnt.get(a, 0) / L * 100 for a in sorted_aa]
    bar_colors = [AA_COLORS.get(a, C["accent1"]) for a in sorted_aa]
    fig, ax    = plt.subplots(figsize=(12, 4))
    apply_plot_style(fig, [ax])
    ax.bar(sorted_aa, freqs, color=bar_colors, edgecolor=C["grid"], width=0.75)
    ax.set_xlabel("Amino Acid", fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency (%)", fontsize=11, labelpad=8)
    ax.set_title("Amino Acid Composition", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=10, colors=C["tick"])
    plt.tight_layout()
    return fig


def plot_sequence_logo_style(seq: str):
    """Bar chart styled like a sequence logo — height = KD hydrophobicity."""
    C   = get_plot_colors()
    fig, ax = plt.subplots(figsize=(max(8, len(seq) * 0.4), 3))
    apply_plot_style(fig, [ax])
    for i, aa in enumerate(seq):
        val   = KD_SCALE.get(aa, 0)
        color = AA_COLORS.get(aa, C["accent1"])
        ax.bar(i, abs(val), bottom=0 if val >= 0 else -abs(val),
               color=color, width=0.8, edgecolor=C["grid"], linewidth=0.4)
        ax.text(i, abs(val)/2 + (0 if val >= 0 else -abs(val)),
                aa, ha="center", va="center",
                fontsize=max(6, min(12, 160 // len(seq))),
                fontweight="bold", color=C["text"])
    ax.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax.set_xlim(-0.5, len(seq) - 0.5)
    ax.set_xlabel("Position", fontsize=11, labelpad=8)
    ax.set_ylabel("|Hydrophobicity|", fontsize=11, labelpad=8)
    ax.set_title("Sequence Logo-Style Hydrophobicity", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels([f"{a}{i+1}" for i, a in enumerate(seq)],
                       rotation=45, ha="right", fontsize=8, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_distributions(df):
    C           = get_plot_colors()
    seq_lengths = [len(s) for s in df["peptide"]]
    taste_counts = df["taste"].value_counts()
    grav_vals   = [gravy_score(s) for s in df["peptide"]]
    fig, axes   = plt.subplots(1, 3, figsize=(16, 5))
    apply_plot_style(fig, axes)
    mean_len = np.mean(seq_lengths)
    axes[0].hist(seq_lengths, bins=20, color=C["accent1"], edgecolor=C["grid"], alpha=0.85)
    axes[0].axvline(mean_len, color=C["red"], linestyle="--", lw=2, label=f"Mean={mean_len:.1f} aa")
    axes[0].set_xlabel("Length (aa)", fontsize=11); axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Peptide Length Distribution", fontsize=12, fontweight="bold", pad=10)
    leg0 = axes[0].legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg0.get_texts(): t.set_color(C["text"])
    n_cls      = len(taste_counts)
    bar_colors = plt.cm.get_cmap("tab20", n_cls)(np.linspace(0, 1, n_cls))
    axes[1].barh(taste_counts.index, taste_counts.values, color=bar_colors, edgecolor=C["grid"], alpha=0.9)
    axes[1].set_xlabel("Count", fontsize=11)
    axes[1].set_title("Taste Class Distribution", fontsize=12, fontweight="bold", pad=10)
    for i, v in enumerate(taste_counts.values):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=9, color=C["text"])
    axes[2].hist(grav_vals, bins=20, color=C["accent2"], edgecolor=C["grid"], alpha=0.85)
    axes[2].axvline(0, color=C["red"], linestyle="--", lw=2, label="Hydrophilic|Hydrophobic")
    axes[2].axvline(np.mean(grav_vals), color=C["orange"], linestyle="--", lw=2,
                    label=f"Mean={np.mean(grav_vals):.2f}")
    axes[2].set_xlabel("GRAVY", fontsize=11); axes[2].set_ylabel("Count", fontsize=11)
    axes[2].set_title("GRAVY Distribution", fontsize=12, fontweight="bold", pad=10)
    leg2 = axes[2].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg2.get_texts(): t.set_color(C["text"])
    plt.tight_layout(pad=2.5)
    return fig


def plot_pca(X, y_labels, class_names, title="PCA"):
    C       = get_plot_colors()
    pca     = PCA(n_components=2)
    coords  = pca.fit_transform(X)
    v1, v2  = pca.explained_variance_ratio_[:2] * 100
    palette = plt.cm.get_cmap("tab20", len(class_names))
    fig, ax = plt.subplots(figsize=(9, 6))
    apply_plot_style(fig, [ax])
    for i, cls in enumerate(class_names):
        mask = y_labels == i
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cls, alpha=0.75, s=35, color=palette(i), edgecolors="none")
    ax.set_xlabel(f"PC1 ({v1:.1f}%)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 ({v2:.1f}%)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left",
                       title="Taste class", title_fontsize=9,
                       facecolor=C["fig_bg"], edgecolor=C["grid"])
    legend.get_title().set_color(C["text"])
    for t in legend.get_texts(): t.set_color(C["text"])
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
    ax.set_title(f"{title} — Accuracy: {acc*100:.1f}%", fontsize=14, fontweight="bold", pad=14)
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
                xy=(0.05, 0.87), xycoords="axes fraction", fontsize=11, color=C["text"],
                bbox=dict(boxstyle="round,pad=0.5", fc=C["fig_bg"], ec=C["grid"], alpha=0.95))
    ax.set_xlabel("True Docking Score (kcal/mol)", fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted Docking Score (kcal/mol)", fontsize=12, labelpad=10)
    ax.set_title("Docking: True vs Predicted", fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(fontsize=10, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in legend.get_texts(): t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    C   = get_plot_colors()
    imp = pd.DataFrame({
        "Feature":    [prettify_feature(f) for f in feature_names],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    clrs = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp))[::-1])
    fig, ax = plt.subplots(figsize=(8, 7))
    apply_plot_style(fig, [ax])
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1], color=clrs, edgecolor=C["grid"])
    ax.set_xlabel("Importance Score", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Features — Taste Model", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_quality_dashboard(score: float, label: str, breakdown: dict):
    C   = get_plot_colors()
    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.6])
    apply_plot_style(fig, [])
    fig.patch.set_facecolor(C["fig_bg"])

    # Gauge-style donut
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C["ax_bg"])
    color_map = {"Excellent": "#12b886", "Good": "#1a8fd1", "Moderate": "#f39c12", "Poor": "#c0392b"}
    col       = color_map.get(label, "#9e9e9e")
    ax1.pie([score, 100 - score], colors=[col, C["grid"]],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.4))
    ax1.text(0, 0, f"{score:.0f}", ha="center", va="center",
             fontsize=32, fontweight="bold", color=col)
    ax1.set_title(f"Quality: {label}", fontsize=13, fontweight="bold",
                  color=C["text"], pad=12)

    # Breakdown bar chart
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C["ax_bg"])
    cats = list(breakdown.keys())
    vals = list(breakdown.values())
    max_per_cat = {"Ramachandran": 30, "Secondary Structure": 30,
                   "SASA Distribution": 20, "pLDDT Confidence": 20}
    max_vals    = [max_per_cat.get(c, 25) for c in cats]
    pcts        = [v / m * 100 for v, m in zip(vals, max_vals)]
    bar_colors  = [("#12b886" if p >= 80 else "#1a8fd1" if p >= 60 else
                    "#f39c12" if p >= 40 else "#c0392b") for p in pcts]
    ax2.barh(cats, vals, color=bar_colors, edgecolor=C["grid"], height=0.55)
    ax2.barh(cats, max_vals, color=C["grid"], height=0.55, alpha=0.25)
    for i, (v, mv) in enumerate(zip(vals, max_vals)):
        ax2.text(v + 0.3, i, f"{v:.1f}/{mv}", va="center", fontsize=9, color=C["text"])
    ax2.set_xlabel("Score", fontsize=11, color=C["text"])
    ax2.set_title("Score Breakdown", fontsize=12, fontweight="bold", color=C["text"], pad=10)
    ax2.tick_params(axis="y", labelsize=9, colors=C["tick"])
    ax2.tick_params(axis="x", labelsize=9, colors=C["tick"])
    for spine in ax2.spines.values(): spine.set_edgecolor(C["grid"])

    plt.tight_layout()
    return fig


# ==========================================================
# SECTION 14 - MODEL TRAINING
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

    return (df, X, Xte, yt_te, ys_te, yd_te,
            taste_model, sol_model, dock_model,
            le_taste, le_sol, metrics)


# ==========================================================
# SECTION 15 - LOAD MODELS
# ==========================================================

(
    df_all, X_all, X_test, yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics,
) = train_models()


# ==========================================================
# SECTION 16 - FULL STRUCTURAL ANALYSIS PANEL
# ==========================================================

def render_structural_analysis(
    pdb_text: str,
    prefix:   str   = "",
    seq:      str   = "",
    plddt_vals: list = None,
    source_label: str = "",
):
    if not pdb_text or not pdb_text.strip():
        st.warning("No PDB data available for structural analysis.")
        return

    if source_label:
        color = "#12b886" if "uploaded" in source_label.lower() else "#e67e22"
        st.markdown(
            f'<div class="structure-badge" style="background:rgba(0,0,0,0.07);'
            f'border:2px solid {color};color:{color};">{source_label}</div>',
            unsafe_allow_html=True,
        )

    # ── DSSP ──────────────────────────────────────────────
    st.markdown("### 🔩 Secondary Structure")
    dssp = run_dssp_fallback(pdb_text)
    c1, c2, c3 = st.columns(3)
    c1.metric("α-Helix",    f"{dssp['helix']}%")
    c2.metric("β-Sheet",    f"{dssp['sheet']}%")
    c3.metric("Coil/Other", f"{dssp['coil']}%")
    fig_pie = plot_dssp_pie(dssp)
    save_fig(fig_pie, f"{prefix}dssp_pie.png")
    st.pyplot(fig_pie)
    plt.close(fig_pie)
    dom = "α-helix" if dssp["helix"] >= dssp["sheet"] else "β-sheet"
    show_caption(
        f"Secondary structure estimated from backbone φ/ψ angles.<br>"
        f"Dominant structure: <strong>{dom}</strong>. "
        f"Helix: <strong>{dssp['helix']}%</strong> | "
        f"Sheet: <strong>{dssp['sheet']}%</strong> | "
        f"Coil: <strong>{dssp['coil']}%</strong>."
    )

    # ── Structural metrics ────────────────────────────────
    st.markdown("### 🔬 Structural Metrics")
    rg       = radius_of_gyration(pdb_text)
    sasa     = compute_sasa(pdb_text)
    hbonds   = count_hbonds(pdb_text)
    ss_bonds = count_disulfide_bonds(pdb_text)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Radius of Gyration", f"{rg:.2f} Å"    if rg   else "N/A")
    m2.metric("Total SASA",         f"{sasa:.1f} Å²"  if sasa else "N/A")
    m3.metric("H-Bonds (est.)",     str(hbonds))
    m4.metric("Disulfide Bonds",    str(ss_bonds))

    # ── Structural Quality Dashboard ──────────────────────
    st.markdown("### 🏅 Structural Quality Dashboard")
    phi_psi    = ramachandran(pdb_text)
    n_residues = len({l[22:26].strip() for l in pdb_text.splitlines() if l.startswith("ATOM")})
    q_score, q_label, q_breakdown = structural_quality_score(
        phi_psi, dssp, sasa, plddt_vals, n_residues,
    )
    fig_qual = plot_quality_dashboard(q_score, q_label, q_breakdown)
    save_fig(fig_qual, f"{prefix}quality_dashboard.png")
    st.pyplot(fig_qual)
    plt.close(fig_qual)
    qual_color = {"Excellent": "#12b886", "Good": "#1a8fd1",
                  "Moderate": "#f39c12", "Poor": "#c0392b"}.get(q_label, "#9e9e9e")
    show_caption(
        f"Overall structural quality score: "
        f"<strong style='color:{qual_color};'>{q_score}/100 — {q_label}</strong>.<br>"
        f"Ramachandran: {q_breakdown.get('Ramachandran', 0):.1f}/30 pts | "
        f"Secondary structure: {q_breakdown.get('Secondary Structure', 0):.1f}/30 pts | "
        f"SASA: {q_breakdown.get('SASA Distribution', 0):.1f}/20 pts | "
        f"pLDDT: {q_breakdown.get('pLDDT Confidence', 0):.1f}/20 pts."
    )

    # ── pLDDT ─────────────────────────────────────────────
    if plddt_vals and len(plddt_vals) > 0 and max(plddt_vals) > 1.0:
        st.markdown("### 📊 pLDDT Confidence Profile")
        mean_pl  = float(np.mean(plddt_vals))
        lbl, lcol = plddt_label(mean_pl)
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Mean pLDDT", f"{mean_pl:.1f}")
        pc2.metric("Min pLDDT",  f"{min(plddt_vals):.1f}")
        pc3.metric("Max pLDDT",  f"{max(plddt_vals):.1f}")
        fig_plddt = plot_plddt(plddt_vals, seq=seq)
        save_fig(fig_plddt, f"{prefix}plddt.png")
        st.pyplot(fig_plddt)
        plt.close(fig_plddt)
        vhigh = sum(1 for v in plddt_vals if v >= 90)
        high  = sum(1 for v in plddt_vals if 70 <= v < 90)
        med   = sum(1 for v in plddt_vals if 50 <= v < 70)
        low   = sum(1 for v in plddt_vals if v < 50)
        show_caption(
            f"Mean pLDDT: <strong style='color:{lcol}'>{mean_pl:.1f} ({lbl})</strong>.<br>"
            f"Very High (≥90): {vhigh} residues | High (70–90): {high} | "
            f"Medium (50–70): {med} | Low (&lt;50): {low}.<br>"
            f"Regions with pLDDT &lt; 50 are disordered or poorly predicted."
        )

    # ── Ramachandran ──────────────────────────────────────
    st.markdown("### 📐 Ramachandran Plot")
    if not phi_psi:
        st.info("No φ/ψ angles — peptide needs ≥3 residues.")
    fig_rama = plot_ramachandran(phi_psi)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.pyplot(fig_rama)
    plt.close(fig_rama)
    show_caption(caption_ramachandran(phi_psi, seq=seq))

    # ── Distance Map ──────────────────────────────────────
    st.markdown("### 🗺️ Cα Distance Map")
    try:
        dist_map = ca_distance_map(pdb_text)
        fig_dist = plot_distance_map(dist_map, seq=seq)
        save_fig(fig_dist, f"{prefix}ca_distance_map.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        show_caption(caption_distance_map(dist_map, seq=seq))
    except Exception as e:
        st.warning(f"Distance map error: {e}")

    # ── Contact Map ───────────────────────────────────────
    st.markdown("### 🔗 Contact Map (8 Å threshold)")
    try:
        cm_mat   = contact_map(pdb_text, threshold=8.0)
        fig_cm   = plot_contact_map(cm_mat, seq=seq, threshold=8.0)
        save_fig(fig_cm, f"{prefix}contact_map.png")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        show_caption(caption_contact_map(cm_mat, threshold=8.0))
    except Exception as e:
        st.warning(f"Contact map error: {e}")

    # ── Per-Residue SASA ──────────────────────────────────
    st.markdown("### 💧 Per-Residue SASA")
    pr_sasa = per_residue_sasa(pdb_text)
    if pr_sasa:
        fig_sasa = plot_sasa_per_residue(pr_sasa, seq=seq)
        if fig_sasa:
            save_fig(fig_sasa, f"{prefix}sasa_per_residue.png")
            st.pyplot(fig_sasa)
            plt.close(fig_sasa)
            show_caption(caption_sasa(sasa, pr_sasa, n_residues))
    else:
        st.info("Per-residue SASA could not be computed.")

    # ── Residue Exposure ──────────────────────────────────
    st.markdown("### 🔍 Residue Exposure Classification")
    exposure = residue_exposure(pdb_text)
    if exposure:
        exp_df = pd.DataFrame(
            [(k, v) for k, v in exposure.items()],
            columns=["Residue", "Exposure"],
        )
        n_exp = (exp_df["Exposure"] == "Exposed").sum()
        n_bur = (exp_df["Exposure"] == "Buried").sum()
        ec1, ec2 = st.columns(2)
        ec1.metric("Exposed Residues", n_exp)
        ec2.metric("Buried Residues",  n_bur)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("Residue exposure could not be computed.")


# ==========================================================
# SECTION 17 - SEQUENCE ANALYSIS PANEL
# ==========================================================

def render_sequence_analysis(seq: str, prefix: str = ""):
    if not seq:
        return

    st.markdown("### 🔠 Amino Acid Composition")
    fig_aa = plot_aa_composition(seq)
    save_fig(fig_aa, f"{prefix}aa_composition.png")
    st.pyplot(fig_aa)
    plt.close(fig_aa)
    cnt  = Counter(seq)
    dom  = max(cnt, key=cnt.get)
    show_caption(
        f"Amino acid frequency across all 20 standard residues.<br>"
        f"Dominant residue: <strong>{dom}</strong> ({cnt[dom]} times, {cnt[dom]/len(seq)*100:.1f}%)."
    )

    st.markdown("### 💧 Per-Residue Hydrophobicity")
    fig_hyd = plot_hydrophobicity(seq)
    save_fig(fig_hyd, f"{prefix}hydrophobicity.png")
    st.pyplot(fig_hyd)
    plt.close(fig_hyd)
    show_caption(caption_hydrophobicity(seq))

    st.markdown("### ⚡ Charge Distribution")
    fig_chg = plot_charge_distribution(seq)
    save_fig(fig_chg, f"{prefix}charge_distribution.png")
    st.pyplot(fig_chg)
    plt.close(fig_chg)
    show_caption(caption_charge(seq))

    st.markdown("### 🔤 Sequence Logo-Style Visualization")
    fig_logo = plot_sequence_logo_style(seq)
    save_fig(fig_logo, f"{prefix}sequence_logo.png")
    st.pyplot(fig_logo)
    plt.close(fig_logo)
    show_caption(
        f"Bar height represents hydrophobicity magnitude. "
        f"Each bar is labelled with the amino acid single-letter code and position."
    )


# ==========================================================
# SECTION 18 - PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics: dict, prediction: dict, image_paths: list) -> str:
    file_name = "PepTastePredictor_Report.pdf"
    styles    = getSampleStyleSheet()
    doc       = SimpleDocTemplate(file_name, pagesize=A4,
                                  topMargin=40, bottomMargin=40,
                                  leftMargin=50, rightMargin=50)
    story = []

    # Title
    story.append(Paragraph("<b>PepTastePredictor — Comprehensive Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking &amp; structural analysis. "
        "Structures generated locally (PeptideBuilder) or externally (ESM Atlas / AlphaFold Server).",
        styles["Normal"]))
    story.append(Spacer(1, 14))

    # Model performance table
    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    table_data = [["Metric", "Value"]] + [[k, str(round(v, 4))] for k, v in metrics.items()]
    tbl = Table(table_data, colWidths=[280, 150])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1f3c88")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 11),
        ("BACKGROUND", (0, 1), (-1, -1), rl_colors.HexColor("#f0f4ff")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#f0f4ff"), rl_colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#cccccc")),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 14))

    # Predictions
    if prediction:
        story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
        for k, v in prediction.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        story.append(Spacer(1, 14))

    # Figures
    story.append(Paragraph("<b>Visual Analytics</b>", styles["Heading2"]))
    story.append(Spacer(1, 8))
    figure_titles = {
        "distributions.png":       "Dataset Distributions",
        "pca_overall.png":         "PCA Feature Space",
        "confusion_taste.png":     "Taste Confusion Matrix",
        "confusion_solubility.png":"Solubility Confusion Matrix",
        "feature_importance_taste.png": "Feature Importance",
        "docking_scatter.png":     "Docking True vs Predicted",
        "dssp_pie.png":            "Secondary Structure (DSSP)",
        "quality_dashboard.png":   "Structural Quality Dashboard",
        "plddt.png":               "pLDDT Profile",
        "ramachandran.png":        "Ramachandran Plot",
        "ca_distance_map.png":     "Cα Distance Map",
        "contact_map.png":         "Contact Map",
        "sasa_per_residue.png":    "Per-Residue SASA",
        "aa_composition.png":      "Amino Acid Composition",
        "hydrophobicity.png":      "Hydrophobicity Profile",
        "charge_distribution.png": "Charge Distribution",
        "sequence_logo.png":       "Sequence Logo",
    }
    for img in image_paths:
        if not os.path.exists(img):
            continue
        basename = os.path.basename(img)
        title    = next((v for k, v in figure_titles.items() if k in basename), basename)
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        story.append(RLImage(img, width=430, height=270))
        story.append(Spacer(1, 20))

    story.append(Paragraph(
        f"<i>Generated by PepTastePredictor · {date.today().strftime('%d %B %Y')}</i>",
        styles["Normal"],
    ))
    doc.build(story)
    return file_name


# ==========================================================
# SECTION 19 - HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🧬 PepTastePredictor</h1>
<p>
An integrated machine learning &amp; structural bioinformatics platform for peptide
taste, solubility, docking, and 3D structural analysis.<br>
Supports <strong>local PeptideBuilder</strong> and external
<strong>ESM Atlas</strong> / <strong>AlphaFold Server</strong> workflows.
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 20 - MODE SELECTION
# ==========================================================

st.markdown("## 🔧 Analysis Mode")
mode = st.radio(
    "Choose the analysis mode",
    ["Single Peptide Prediction", "Batch Peptide Prediction", "PDB Upload & Structural Analysis"],
    horizontal=True,
)
if "current_mode" not in st.session_state or st.session_state.current_mode != mode:
    st.session_state.pdf_figures     = []
    st.session_state.show_analytics  = False
    st.session_state.last_prediction = {}
    st.session_state.pdb_text        = None
    st.session_state.pdb_source      = None
    st.session_state.current_mode    = mode


# ==========================================================
# SECTION 21 - SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    # ── Tabs ──────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📝 Sequence", "🤖 ML Predictions", "🧬 Structure Viewer",
        "📐 Structural Analysis", "🔬 Advanced Bioinformatics",
        "📊 Analytics Dashboard", "📄 Report",
    ])

    # ── TAB 1 : Sequence Input ────────────────────────────
    with tab1:
        st.markdown("## 🔬 Single Peptide Prediction")
        seq_raw = st.text_area(
            "Enter peptide sequence (FASTA or plain single-letter code)",
            help="Accepts 1–2500 amino acids. FASTA headers are stripped automatically.",
            placeholder="Paste sequence or FASTA here…",
            key="single_seq_input",
            height=120,
        )
        seq_clean = clean_sequence(seq_raw)

        if seq_raw:
            raw_stripped = re.sub(r">[^\n]*\n?", "", seq_raw)
            raw_stripped = raw_stripped.replace(" ", "").replace("\n", "").replace("\t", "").upper()
            invalid      = len([c for c in raw_stripped if c not in AA])
            badge_color  = "#12b886" if len(seq_clean) > 0 else "#c0392b"
            inv_note     = (
                f" &nbsp; <span style='color:#c0392b;'>({invalid} invalid character(s) removed)</span>"
                if invalid else ""
            )
            st.markdown(
                f'<div class="seq-counter">Valid amino acids: '
                f'<span style="color:{badge_color};font-weight:800;">{len(seq_clean)}</span>'
                f'{inv_note}</div>',
                unsafe_allow_html=True,
            )

        if seq_clean:
            render_live_preview(seq_clean)
            render_sequence_analysis(seq_clean, prefix="single_")

            # Download FASTA
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download as FASTA",
                               f">peptide\n{seq_clean}\n",
                               file_name="peptide.fasta", mime="text/plain")
            d2.download_button("⬇️ Download sequence (.txt)",
                               seq_clean, file_name="peptide_sequence.txt", mime="text/plain")

    # ── TAB 2 : ML Predictions ────────────────────────────
    with tab2:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not seq_clean:
            st.info("Enter a sequence in the Sequence tab first.")
        else:
            ml_seq = seq_clean[:100]
            Xp     = pd.DataFrame([model_features(ml_seq)])
            taste  = le_taste.inverse_transform(taste_model.predict(Xp))[0]
            sol    = le_sol.inverse_transform(sol_model.predict(Xp))[0]
            dock   = dock_model.predict(Xp)[0]
            emoji  = taste_emoji(taste)
            sol_color  = "#12b886" if "soluble" in sol.lower() else "#e67e22"
            dock_color = "#12b886" if dock < -6 else ("#f39c12" if dock < -4 else "#c0392b")

            st.markdown(f"""
            <div class="card">
              <div style="font-size:12px;font-weight:700;opacity:0.5;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:16px;">
                <span class="live-indicator"></span>ML Prediction Results
              </div>
              <div style="display:flex;gap:40px;flex-wrap:wrap;">
                <div><div class="metric-label" style="margin-top:0;">Taste</div>
                     <div class="metric-value">{emoji} {taste}</div></div>
                <div><div class="metric-label" style="margin-top:0;">Solubility</div>
                     <div class="metric-value" style="color:{sol_color} !important;">{sol}</div></div>
                <div><div class="metric-label" style="margin-top:0;">Docking Score</div>
                     <div class="metric-value" style="color:{dock_color} !important;">{dock:.3f} kcal/mol</div>
                     <div style="font-size:11px;opacity:0.6;">
                       {'Strong binder' if dock<-6 else 'Moderate' if dock<-4 else 'Weak binder'}
                     </div></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if len(seq_clean) > 100:
                st.info(f"ML predictions use first 100 aa of your {len(seq_clean)}-aa sequence.")

            st.markdown("### 📌 Physicochemical Properties")
            phys = physicochemical_features(ml_seq)
            cols = st.columns(min(len(phys), 4))
            for i, (k, v) in enumerate(phys.items()):
                cols[i % len(cols)].metric(k, v)

            st.markdown("### 🧪 Amino Acid Group Composition")
            comp      = composition_features(seq_clean)
            comp_cols = st.columns(len(comp))
            for i, (k, v) in enumerate(comp.items()):
                comp_cols[i].metric(k, f"{v}%")
                comp_cols[i].markdown(
                    f'<div class="progress-bar-wrap"><div class="progress-bar-fill" '
                    f'style="width:{v}%;background:#1a8fd1;"></div></div>',
                    unsafe_allow_html=True,
                )

            # Confidence scores
            st.markdown("### 🎯 Prediction Confidence")
            taste_proba  = taste_model.predict_proba(Xp)[0]
            taste_conf   = max(taste_proba) * 100
            sol_proba    = sol_model.predict_proba(Xp)[0]
            sol_conf     = max(sol_proba) * 100
            conf1, conf2 = st.columns(2)
            conf1.metric("Taste Confidence",      f"{taste_conf:.1f}%")
            conf2.metric("Solubility Confidence", f"{sol_conf:.1f}%")

            # Top 3 taste classes
            st.markdown("#### Top Taste Probabilities")
            prob_df = pd.DataFrame({
                "Taste Class": le_taste.classes_,
                "Probability": taste_proba,
            }).sort_values("Probability", ascending=False).head(5)
            prob_df["Probability"] = prob_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(prob_df.reset_index(drop=True), use_container_width=True)

            st.session_state.last_prediction = {
                "Sequence (first 60 aa)":   seq_clean[:60] + ("…" if len(seq_clean) > 60 else ""),
                "Full sequence length":     len(seq_clean),
                "Predicted taste":          taste,
                "Predicted solubility":     sol,
                "Docking score (kcal/mol)": round(dock, 3),
                "Taste confidence":         f"{taste_conf:.1f}%",
                "Solubility confidence":    f"{sol_conf:.1f}%",
            }
            st.session_state.show_analytics = True

    # ── TAB 3 : Structure Viewer ──────────────────────────
    with tab3:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not seq_clean:
            st.info("Enter a sequence in the Sequence tab first.")
        else:
            # External folding section
            st.markdown("## 🌐 External Structure Prediction")
            st.markdown("**Your sequence (copy this):**")
            st.code(seq_clean, language=None)
            b1, b2 = st.columns(2)
            b1.markdown(
                '<a href="https://esmatlas.com/resources?action=fold" target="_blank">'
                '<button style="width:100%;padding:14px 0;font-size:15px;font-weight:700;'
                'border-radius:10px;border:2px solid #12b886;background:rgba(18,184,134,0.12);'
                'color:#12b886;cursor:pointer;">🌿 Open ESM Atlas</button></a>',
                unsafe_allow_html=True,
            )
            b2.markdown(
                '<a href="https://alphafoldserver.com" target="_blank">'
                '<button style="width:100%;padding:14px 0;font-size:15px;font-weight:700;'
                'border-radius:10px;border:2px solid #1a8fd1;background:rgba(26,143,209,0.12);'
                'color:#1a8fd1;cursor:pointer;">🔬 Open AlphaFold Server</button></a>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="ext-card">
              <div style="font-size:13px;font-weight:700;opacity:0.55;text-transform:uppercase;
                          letter-spacing:0.08em;margin-bottom:18px;">How to get your PDB structure</div>
              <div class="step-row"><div class="step-num">1</div>
                <div class="step-text">Copy the sequence above.</div></div>
              <div class="step-row"><div class="step-num">2</div>
                <div class="step-text">Open ESM Atlas or AlphaFold Server.</div></div>
              <div class="step-row"><div class="step-num">3</div>
                <div class="step-text">Paste and run the prediction.</div></div>
              <div class="step-row"><div class="step-num">4</div>
                <div class="step-text">Download the resulting PDB file.</div></div>
              <div class="step-row"><div class="step-num">5</div>
                <div class="step-text">Upload the PDB below, or use the local builder.</div></div>
            </div>
            """, unsafe_allow_html=True)

            # ── PDB Upload ────────────────────────────────
            st.markdown("### 📂 Upload External PDB")
            uploaded_pdb = st.file_uploader("Upload PDB from ESM Atlas / AlphaFold Server",
                                            type=["pdb"], key="single_pdb_upload")
            if uploaded_pdb is not None:
                try:
                    pdb_text = uploaded_pdb.read().decode("utf-8")
                    if pdb_text.strip():
                        st.session_state.pdb_text   = pdb_text
                        st.session_state.pdb_source = "Uploaded Structure (ESM / AlphaFold)"
                        st.success("PDB uploaded successfully.")
                    else:
                        st.error("Uploaded PDB is empty.")
                except Exception as e:
                    st.error(f"Could not read PDB: {e}")

            # ── Local PeptideBuilder ───────────────────────
            st.markdown("### ⚙️ Or Generate Locally with PeptideBuilder")
            st.info(
                "PeptideBuilder creates a backbone-only model (no side-chain optimization). "
                "Ideal for quick structural estimates of short peptides (≤30 aa)."
            )
            if st.button("🏗️ Build Backbone Structure Locally"):
                with st.spinner("Building peptide backbone with PeptideBuilder…"):
                    pdb_built = build_peptide_pdb(seq_clean[:50])
                if pdb_built:
                    st.session_state.pdb_text   = pdb_built
                    st.session_state.pdb_source = "PeptideBuilder Backbone Model"
                    st.success(f"Backbone structure built for {min(len(seq_clean), 50)} residues.")
                else:
                    st.error("PeptideBuilder failed. Check that your sequence uses standard amino acids.")

            # ── 3D Viewer ─────────────────────────────────
            if st.session_state.pdb_text:
                pdb_text = st.session_state.pdb_text
                n_atoms  = sum(1 for l in pdb_text.splitlines() if l.startswith("ATOM"))
                n_res    = len({l[22:26].strip() for l in pdb_text.splitlines() if l.startswith("ATOM")})
                ic1, ic2, ic3 = st.columns(3)
                ic1.metric("ATOM records", n_atoms)
                ic2.metric("Residues",     n_res)
                ic3.metric("File size",    f"{max(1, len(pdb_text)//1024)} KB")

                st.markdown("### 🧬 3D Structure Viewer")
                try:
                    st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)
                except Exception as e:
                    st.warning(f"3D viewer error: {e}")

                rmsd_val = ca_rmsd(pdb_text)
                if rmsd_val is not None:
                    st.success(f"Cα RMSD from first residue: **{rmsd_val:.3f} Å**")

                st.download_button("⬇️ Download PDB", pdb_text,
                                   file_name="structure.pdb", mime="text/plain")

    # ── TAB 4 : Structural Analysis ───────────────────────
    with tab4:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not st.session_state.pdb_text:
            st.info("Generate or upload a structure in the **Structure Viewer** tab first.")
        else:
            pdb_text    = st.session_state.pdb_text
            plddt_vals  = _extract_plddt_from_pdb(pdb_text)
            has_plddt   = len(plddt_vals) > 0 and max(plddt_vals) > 1.0
            render_structural_analysis(
                pdb_text,
                prefix="single_",
                seq=seq_clean,
                plddt_vals=plddt_vals if has_plddt else None,
                source_label=st.session_state.pdb_source or "",
            )

    # ── TAB 5 : Advanced Bioinformatics ──────────────────
    with tab5:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not seq_clean:
            st.info("Enter a sequence in the Sequence tab first.")
        else:
            st.markdown("## 🔬 Advanced Bioinformatics")

            # Codon usage hint
            with st.expander("🧬 Sequence Properties Summary"):
                phys  = physicochemical_features(seq_clean[:100])
                for k, v in phys.items():
                    st.write(f"**{k}**: {v}")

            # Dipeptide composition heatmap
            st.markdown("### 🔢 Dipeptide Frequency Heatmap")
            seq_for_dp = seq_clean[:100]
            L          = len(seq_for_dp)
            denom      = max(L - 1, 1)
            dp_matrix  = np.zeros((20, 20))
            aa_list    = list(AA)
            for i, a1 in enumerate(aa_list):
                for j, a2 in enumerate(aa_list):
                    dp_matrix[i, j] = seq_for_dp.count(a1 + a2) / denom * 100
            C      = get_plot_colors()
            fig_dp, ax_dp = plt.subplots(figsize=(10, 8))
            apply_plot_style(fig_dp, [ax_dp])
            sns.heatmap(dp_matrix, xticklabels=aa_list, yticklabels=aa_list,
                        cmap="YlOrRd", ax=ax_dp, linewidths=0.2,
                        cbar_kws={"label": "Frequency (%)"})
            ax_dp.set_title("Dipeptide Composition Heatmap", fontsize=13, fontweight="bold", pad=12)
            ax_dp.set_xlabel("Second AA", fontsize=11)
            ax_dp.set_ylabel("First AA",  fontsize=11)
            cbar = ax_dp.collections[0].colorbar
            cbar.ax.yaxis.label.set_color(C["text"])
            cbar.ax.tick_params(colors=C["text"])
            plt.xticks(fontsize=9, color=C["tick"])
            plt.yticks(fontsize=9, color=C["tick"], rotation=0)
            plt.tight_layout()
            save_fig(fig_dp, "single_dipeptide_heatmap.png")
            st.pyplot(fig_dp)
            plt.close(fig_dp)
            top_dp = max(
                [(a1 + a2, seq_for_dp.count(a1 + a2)) for a1 in AA for a2 in AA],
                key=lambda x: x[1],
            )
            show_caption(
                f"Frequency of all dipeptide pairs in the first {L} residues.<br>"
                f"Most frequent dipeptide: <strong>{top_dp[0]}</strong> "
                f"({top_dp[1]} occurrence(s))."
            )

            # Window hydrophobicity
            st.markdown("### 🪟 Sliding-Window Hydrophobicity (w=7)")
            win_size = 7
            if len(seq_clean) >= win_size:
                win_vals = [
                    np.mean([KD_SCALE.get(seq_clean[k], 0) for k in range(i, i + win_size)])
                    for i in range(len(seq_clean) - win_size + 1)
                ]
                fig_win, ax_win = plt.subplots(figsize=(max(8, len(win_vals) * 0.25), 4))
                apply_plot_style(fig_win, [ax_win])
                ax_win.plot(range(len(win_vals)), win_vals, color=C["accent1"], lw=2.5)
                ax_win.fill_between(range(len(win_vals)), win_vals, 0,
                                    where=[v > 0 for v in win_vals],
                                    color=C["accent1"], alpha=0.25, label="Hydrophobic")
                ax_win.fill_between(range(len(win_vals)), win_vals, 0,
                                    where=[v <= 0 for v in win_vals],
                                    color=C["red"], alpha=0.25, label="Hydrophilic")
                ax_win.axhline(0, color=C["grid"], lw=1, linestyle="--")
                ax_win.set_xlabel("Window start position", fontsize=11)
                ax_win.set_ylabel("Avg hydrophobicity", fontsize=11)
                ax_win.set_title(f"Sliding-Window Hydrophobicity (w={win_size})", fontsize=13, fontweight="bold")
                leg_w = ax_win.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
                for t in leg_w.get_texts(): t.set_color(C["text"])
                plt.tight_layout()
                save_fig(fig_win, "single_window_hydrophobicity.png")
                st.pyplot(fig_win)
                plt.close(fig_win)
                show_caption(
                    f"Hydrophobicity averaged over a window of {win_size} residues. "
                    f"Regions above 0 are hydrophobic (potential membrane-insertion segments); "
                    f"below 0 are hydrophilic (potential solvent-exposed regions)."
                )
            else:
                st.info(f"Sequence must be ≥{win_size} residues for sliding-window analysis.")

    # ── TAB 6 : Analytics Dashboard ──────────────────────
    with tab6:
        if not st.session_state.show_analytics:
            st.info("Run a prediction (ML Predictions tab) to see analytics.")
        else:
            st.markdown("### 📈 Model Performance")
            mc = st.columns(3)
            mc[0].metric("Taste Accuracy",      f"{metrics['Taste accuracy']*100:.1f}%")
            mc[0].metric("Taste F1",            f"{metrics['Taste F1']:.3f}")
            mc[1].metric("Solubility Accuracy", f"{metrics['Solubility accuracy']*100:.1f}%")
            mc[1].metric("Solubility F1",       f"{metrics['Solubility F1']:.3f}")
            mc[2].metric("Docking R²",          f"{metrics['Docking R2']:.3f}")
            mc[2].metric("Docking RMSE",        f"{metrics['Docking RMSE']:.3f} kcal/mol")

            st.markdown("### 📊 Dataset Distributions")
            fig_dist = plot_distributions(df_all)
            save_fig(fig_dist, "distributions.png")
            st.pyplot(fig_dist)
            plt.close(fig_dist)
            show_caption(caption_distributions(df_all))

            st.markdown("### 🔹 PCA Feature Space")
            fig_pca, pca_model = plot_pca(
                X_all, le_taste.transform(df_all["taste"]), le_taste.classes_,
                title="PCA — Peptide Feature Space (by taste class)",
            )
            save_fig(fig_pca, "pca_overall.png")
            st.pyplot(fig_pca)
            plt.close(fig_pca)
            show_caption(caption_pca(pca_model, le_taste.classes_))

            st.markdown("### 🔹 Taste Confusion Matrix")
            taste_preds  = taste_model.predict(X_test)
            fig_cm_taste = plot_confusion(yt_test, taste_preds, le_taste.classes_,
                                          "Taste Confusion Matrix", "Blues")
            save_fig(fig_cm_taste, "confusion_taste.png")
            st.pyplot(fig_cm_taste)
            plt.close(fig_cm_taste)
            show_caption(caption_confusion_taste(yt_test, taste_preds, le_taste.classes_))

            st.markdown("### 🔹 Solubility Confusion Matrix")
            sol_preds  = sol_model.predict(X_test)
            fig_cm_sol = plot_confusion(ys_test, sol_preds, le_sol.classes_,
                                        "Solubility Confusion Matrix", "Greens")
            save_fig(fig_cm_sol, "confusion_solubility.png")
            st.pyplot(fig_cm_sol)
            plt.close(fig_cm_sol)
            show_caption(caption_confusion_sol(ys_test, sol_preds, le_sol.classes_))

            st.markdown("### 🔹 Feature Importance")
            fig_imp = plot_feature_importance(taste_model, X_all.columns, top_n=20)
            save_fig(fig_imp, "feature_importance_taste.png")
            st.pyplot(fig_imp)
            plt.close(fig_imp)
            show_caption(caption_feature_importance(taste_model, X_all.columns, top_n=20))

            st.markdown("### 🔹 Docking: True vs Predicted")
            dock_preds = dock_model.predict(X_test)
            fig_dock   = plot_docking(yd_test, dock_preds)
            save_fig(fig_dock, "docking_scatter.png")
            st.pyplot(fig_dock)
            plt.close(fig_dock)
            show_caption(caption_docking(yd_test, dock_preds))

    # ── TAB 7 : PDF Report ────────────────────────────────
    with tab7:
        if not st.session_state.show_analytics:
            st.info("Complete predictions and structural analysis first.")
        else:
            st.markdown("### 📄 Download Complete PDF Report")
            if len(st.session_state.pdf_figures) > 0:
                pdf_path = generate_pdf(
                    metrics,
                    st.session_state.last_prediction,
                    st.session_state.pdf_figures,
                )
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📥 Download Full Analytics PDF", f,
                            file_name="PepTastePredictor_Report.pdf",
                            mime="application/pdf",
                        )
            else:
                st.info("Generate plots across the tabs first — they will be collected here.")


# ==========================================================
# SECTION 22 - BATCH PEPTIDE PREDICTION MODE
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
                st.error("No valid peptide sequences found.")
            else:
                total    = len(batch_df)
                progress = st.progress(0, text="Processing peptides…")
                tastes, sols, docks, taste_confs = [], [], [], []

                for i, row in batch_df.iterrows():
                    try:
                        ml_seq = row["peptide"][:100]
                        Xr     = pd.DataFrame([model_features(ml_seq)])
                        t      = le_taste.inverse_transform(taste_model.predict(Xr))[0]
                        s      = le_sol.inverse_transform(sol_model.predict(Xr))[0]
                        d      = round(dock_model.predict(Xr)[0], 3)
                        tc     = round(max(taste_model.predict_proba(Xr)[0]) * 100, 1)
                    except Exception:
                        t, s, d, tc = "Error", "Error", None, None
                    tastes.append(t); sols.append(s); docks.append(d); taste_confs.append(tc)
                    progress.progress(min(int((i + 1) / total * 100), 100),
                                      text=f"Processing {i+1}/{total}…")

                progress.progress(100, text="Done!")
                batch_df["Predicted Taste"]         = tastes
                batch_df["Predicted Solubility"]    = sols
                batch_df["Predicted Docking Score"] = docks
                batch_df["Taste Confidence (%)"]    = taste_confs

                # Summary
                st.markdown("### 📊 Batch Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Peptides", total)
                s2.metric("Unique Tastes",  batch_df["Predicted Taste"].nunique())
                sol_pct = 100 * batch_df["Predicted Solubility"].str.contains(
                    "oluble", case=False, na=False).mean()
                s3.metric("Soluble (%)", f"{sol_pct:.1f}%")
                valid_d = batch_df["Predicted Docking Score"].dropna()
                s4.metric("Avg Docking", f"{valid_d.mean():.2f} kcal/mol" if len(valid_d) else "N/A")

                # Taste distribution
                C      = get_plot_colors()
                fig_b, ax_b = plt.subplots(figsize=(8, 3))
                apply_plot_style(fig_b, [ax_b])
                tc = batch_df["Predicted Taste"].value_counts()
                clrs_b = plt.cm.get_cmap("tab20", len(tc))(np.linspace(0, 1, len(tc)))
                ax_b.barh(tc.index, tc.values, color=clrs_b, edgecolor=C["grid"])
                ax_b.set_xlabel("Count", color=C["text"])
                ax_b.set_title("Batch Taste Distribution", color=C["text"], fontweight="bold")
                plt.tight_layout()
                save_fig(fig_b, "batch_taste_distribution.png")
                st.pyplot(fig_b)
                plt.close(fig_b)

                # Docking distribution
                if len(valid_d) > 1:
                    fig_d2, ax_d2 = plt.subplots(figsize=(8, 3))
                    apply_plot_style(fig_d2, [ax_d2])
                    ax_d2.hist(valid_d, bins=20, color=C["accent1"], edgecolor=C["grid"], alpha=0.85)
                    ax_d2.axvline(valid_d.mean(), color=C["red"], linestyle="--", lw=2,
                                  label=f"Mean = {valid_d.mean():.2f}")
                    ax_d2.set_xlabel("Docking Score (kcal/mol)"); ax_d2.set_ylabel("Count")
                    ax_d2.set_title("Batch Docking Score Distribution", fontweight="bold")
                    leg_d = ax_d2.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
                    for t in leg_d.get_texts(): t.set_color(C["text"])
                    plt.tight_layout()
                    save_fig(fig_d2, "batch_docking_distribution.png")
                    st.pyplot(fig_d2)
                    plt.close(fig_d2)

                st.markdown("### ✅ Batch Results")
                st.dataframe(batch_df, use_container_width=True)
                st.download_button(
                    "⬇️ Download Batch Predictions (CSV)",
                    batch_df.to_csv(index=False),
                    file_name="batch_predictions.csv",
                )
                st.session_state.show_analytics = True

                # PDF
                if len(st.session_state.pdf_figures) > 0:
                    st.markdown("### 📄 Download Report")
                    pdf_path = generate_pdf(metrics, {}, st.session_state.pdf_figures)
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "📥 Download Batch PDF Report", f,
                                file_name="PepTastePredictor_Batch_Report.pdf",
                                mime="application/pdf",
                            )


# ==========================================================
# SECTION 23 - PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

elif mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyze PDB Structure")
    st.info(
        "Generate your structure using [ESM Atlas](https://esmatlas.com/resources?action=fold) "
        "or [AlphaFold Server](https://alphafoldserver.com), then upload the PDB file below.",
        icon="🌐",
    )
    uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])

    if uploaded_pdb is not None:
        try:
            pdb_text = uploaded_pdb.read().decode("utf-8")
        except Exception as e:
            st.error(f"Could not read PDB: {e}")
            pdb_text = ""

        if pdb_text and pdb_text.strip():
            st.session_state.pdb_text       = pdb_text
            st.session_state.pdb_source     = "Uploaded Structure"
            st.session_state.show_analytics = True

            n_atoms    = sum(1 for l in pdb_text.splitlines() if l.startswith("ATOM"))
            n_residues = len({l[22:26].strip() for l in pdb_text.splitlines() if l.startswith("ATOM")})
            plddt_vals = _extract_plddt_from_pdb(pdb_text)
            has_plddt  = len(plddt_vals) > 0 and max(plddt_vals) > 1.0

            c1, c2, c3 = st.columns(3)
            c1.metric("ATOM records", n_atoms)
            c2.metric("Residues",     n_residues)
            c3.metric("File size",    f"{max(1, len(pdb_text)//1024)} KB")

            st.markdown("### 🧬 3D Structure Viewer")
            try:
                st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)
            except Exception as e:
                st.warning(f"3D viewer error: {e}")

            rmsd_val = ca_rmsd(pdb_text)
            if rmsd_val is not None:
                st.success(f"Cα RMSD: **{rmsd_val:.3f} Å**")

            render_structural_analysis(
                pdb_text, prefix="pdb_",
                plddt_vals=plddt_vals if has_plddt else None,
                source_label="Uploaded Structure",
            )

            # Analytics expander
            with st.expander("📊 Dataset Analytics", expanded=False):
                fig_dist = plot_distributions(df_all)
                save_fig(fig_dist, "distributions.png")
                st.pyplot(fig_dist)
                plt.close(fig_dist)
                show_caption(caption_distributions(df_all))

            # PDF download
            if len(st.session_state.pdf_figures) > 0:
                st.markdown("### 📄 Download Report")
                pdf_path = generate_pdf(metrics, {}, st.session_state.pdf_figures)
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            "📥 Download Structural Analysis PDF", f,
                            file_name="PepTastePredictor_Structural_Report.pdf",
                            mime="application/pdf",
                        )
        else:
            st.error("Uploaded PDB file is empty or could not be decoded.")


# ==========================================================
# SECTION 24 - FOOTER
# ==========================================================

st.markdown(f"""
<div class="footer">
&copy; {date.today().year} &nbsp; <b>PepTastePredictor v3</b><br>
AI + Structural Bioinformatics platform for peptide analysis<br>
Local PeptideBuilder &nbsp;|&nbsp; External: ESM Atlas &amp; AlphaFold Server<br>
For academic, educational, and research use only
</div>
""", unsafe_allow_html=True)
