# ==========================================================
# PepTastePredictor — app.py  (Code 4 — Full Dynamic Structural Overhaul)
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

.hero {
    background: linear-gradient(135deg, #1f3c88 0%, #0b7285 60%, #12b886 100%);
    padding: 40px 44px; border-radius: 20px; margin-bottom: 36px;
    box-shadow: 0 8px 32px rgba(31,60,136,0.18);
}
.hero h1 { font-size: 2.4rem !important; font-weight: 800 !important;
    margin-bottom: 10px; color: #ffffff !important; letter-spacing: -0.5px; }
.hero p { font-size: 1.08rem !important; line-height: 1.8; color: #dce8ff !important; margin: 0; }

.card { border: 1px solid rgba(128,128,180,0.3); padding: 28px 32px;
    border-radius: 16px; margin-bottom: 28px; background: rgba(128,128,180,0.05);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); }
.live-card { border: 2px solid rgba(26,143,209,0.35); padding: 22px 28px;
    border-radius: 14px; margin-bottom: 18px; background: rgba(26,143,209,0.05); }
.ext-card { border: 2px solid rgba(18,184,134,0.4); padding: 26px 32px;
    border-radius: 16px; margin-bottom: 24px; background: rgba(18,184,134,0.05); }
.quality-card { border: 2px solid rgba(74,111,165,0.4); padding: 24px 28px;
    border-radius: 16px; margin-bottom: 24px; background: rgba(74,111,165,0.06); }

.step-row { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 14px; }
.step-num { min-width: 32px; height: 32px; border-radius: 50%; background: #1a8fd1;
    color: #fff !important; font-weight: 800; font-size: 14px;
    display: flex; align-items: center; justify-content: center; }
.step-text { font-size: 15px; line-height: 1.6; color: var(--text-color) !important; padding-top: 4px; }

.metric-label { font-size: 12px !important; font-weight: 700 !important;
    text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.6;
    margin-bottom: 4px; margin-top: 18px; color: var(--text-color) !important; }
.metric-label:first-child { margin-top: 0; }
.metric-value { font-size: 24px !important; font-weight: 800 !important;
    color: #1a8fd1 !important; margin-bottom: 2px; }

.progress-bar-wrap { background: rgba(128,128,180,0.15); border-radius: 8px;
    height: 10px; margin: 6px 0 16px 0; overflow: hidden; }
.progress-bar-fill { height: 10px; border-radius: 8px; }

.aa-badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 13px; font-weight: 700; margin: 3px 2px;
    background: rgba(26,143,209,0.15); color: #1a8fd1 !important;
    border: 1px solid rgba(26,143,209,0.3); }
.seq-counter { font-size: 13px; font-weight: 600; opacity: 0.65;
    margin-bottom: 6px; color: var(--text-color) !important; }

.graph-caption { border-left: 5px solid #4a6fa5; border-radius: 0 10px 10px 0;
    padding: 18px 24px; margin-top: 12px; margin-bottom: 40px;
    font-size: 15px !important; line-height: 1.9; background: rgba(74,111,165,0.08);
    color: var(--text-color) !important; }
.graph-caption strong { font-weight: 700; color: var(--text-color) !important; }
.graph-caption em { font-style: italic; color: var(--text-color) !important; opacity: 0.85; }

.section-gap { margin-top: 40px; margin-bottom: 6px; }
.structure-badge { display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-size: 13px; font-weight: 700; margin-bottom: 12px; }

.footer { text-align: center; font-size: 14px !important; padding: 44px 20px 20px;
    margin-top: 60px; line-height: 2.2;
    border-top: 1px solid rgba(128,128,180,0.25);
    color: var(--text-color) !important; opacity: 0.7; }

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

st.sidebar.markdown("### 🧬 PepTastePredictor v4")
st.sidebar.write("AI-driven peptide analysis platform")
st.sidebar.write("• Taste prediction")
st.sidebar.write("• Solubility prediction")
st.sidebar.write("• Docking estimation")
st.sidebar.write("• Structural bioinformatics")
st.sidebar.write("• Batch screening")
st.sidebar.write("• Dynamic structural analysis")

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
    "pdb_source":      None,
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
    except Exception:
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
    helix_ratio = n_h / total if total > 0 else 0.0
    sheet_ratio = n_e / total if total > 0 else 0.0
    coil_ratio  = n_c / total if total > 0 else 1.0
    hs_ratio    = (n_h / max(n_e, 1)) if n_e > 0 else float(n_h)
    # secondary structure entropy
    probs = [p for p in [helix_ratio, sheet_ratio, coil_ratio] if p > 0]
    ss_entropy = -sum(p * np.log2(p) for p in probs)
    return {
        "helix":       round(helix_ratio * 100, 1),
        "sheet":       round(sheet_ratio * 100, 1),
        "coil":        round(coil_ratio  * 100, 1),
        "raw":         raw,
        "hs_ratio":    round(hs_ratio, 2),
        "ss_entropy":  round(ss_entropy, 3),
        "n_helix":     n_h,
        "n_sheet":     n_e,
        "n_coil":      n_c,
        "total":       total,
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
    sasa_data = per_residue_sasa(pdb_text)
    if not sasa_data:
        return {}
    vals = list(sasa_data.values())
    threshold = np.median(vals) if vals else 30.0
    return {k: ("Exposed" if v >= threshold else "Buried") for k, v in sasa_data.items()}


# ==========================================================
# SECTION 12 - DYNAMIC SCORING ENGINE
# ==========================================================

def compute_rama_stats(phi_psi: list) -> dict:
    """Fully dynamic Ramachandran statistics."""
    if not phi_psi:
        return {
            "total": 0, "helix_count": 0, "sheet_count": 0, "lhelix_count": 0,
            "allowed_count": 0, "outlier_count": 0,
            "helix_pct": 0.0, "sheet_pct": 0.0, "lhelix_pct": 0.0,
            "allowed_pct": 0.0, "outlier_pct": 0.0,
            "backbone_strain": 0.0, "conformational_flexibility": 0.0,
            "outlier_indices": [], "dominant_region": "Undefined",
        }
    total = len(phi_psi)
    helix_count  = sum(1 for p, s in phi_psi if -180<=p<=-45 and -75<=s<=-15)
    sheet_count  = sum(1 for p, s in phi_psi if -180<=p<=-45 and (90<=s<=180 or -180<=s<=-150))
    lhelix_count = sum(1 for p, s in phi_psi if 45<=p<=90 and 0<=s<=90)
    allowed_count = helix_count + sheet_count + lhelix_count
    outlier_count = total - allowed_count

    helix_pct  = helix_count / total * 100
    sheet_pct  = sheet_count / total * 100
    lhelix_pct = lhelix_count / total * 100
    allowed_pct = allowed_count / total * 100
    outlier_pct = outlier_count / total * 100

    # backbone strain: fraction of outliers (inverted → higher = more strain)
    backbone_strain = outlier_pct
    # conformational flexibility: dispersion of phi/psi angles
    phi_vals = [p for p, s in phi_psi]
    psi_vals = [s for p, s in phi_psi]
    conformational_flexibility = round((np.std(phi_vals) + np.std(psi_vals)) / 2, 1)

    # outlier indices
    outlier_indices = [i for i, (p, s) in enumerate(phi_psi)
                       if not ((-180<=p<=-45 and -75<=s<=-15) or
                               (-180<=p<=-45 and (90<=s<=180 or -180<=s<=-150)) or
                               (45<=p<=90 and 0<=s<=90))]

    # dominant region
    counts = {"Alpha-helix": helix_count, "Beta-sheet": sheet_count,
              "Left-handed helix": lhelix_count, "Coil/Other": outlier_count}
    dominant_region = max(counts, key=counts.get)

    return {
        "total": total, "helix_count": helix_count, "sheet_count": sheet_count,
        "lhelix_count": lhelix_count, "allowed_count": allowed_count, "outlier_count": outlier_count,
        "helix_pct": round(helix_pct, 1), "sheet_pct": round(sheet_pct, 1),
        "lhelix_pct": round(lhelix_pct, 1), "allowed_pct": round(allowed_pct, 1),
        "outlier_pct": round(outlier_pct, 1), "backbone_strain": round(backbone_strain, 1),
        "conformational_flexibility": conformational_flexibility,
        "outlier_indices": outlier_indices, "dominant_region": dominant_region,
    }


def compute_distance_stats(dist_matrix: np.ndarray) -> dict:
    """Dynamic distance map statistics."""
    n = dist_matrix.shape[0]
    if n < 2:
        return {
            "n": n, "mean_dist": 0.0, "median_dist": 0.0,
            "max_dist": 0.0, "min_dist": 0.0, "std_dist": 0.0,
            "long_range_contacts": 0, "contact_density": 0.0,
            "compactness_score": 0.0, "folding_tendency": "Undefined",
            "clustering_score": 0.0,
        }
    mask = ~np.eye(n, dtype=bool)
    od = dist_matrix[mask]
    mean_dist   = float(np.mean(od))
    median_dist = float(np.median(od))
    max_dist    = float(np.max(od))
    min_dist    = float(np.min(od))
    std_dist    = float(np.std(od))
    long_range  = sum(1 for i in range(n) for j in range(n) if abs(i-j)>3 and dist_matrix[i,j]<8.0)
    total_possible_lr = max(sum(1 for i in range(n) for j in range(n) if abs(i-j)>3), 1)
    contact_density   = long_range / total_possible_lr * 100
    # compactness score: lower max distance relative to n → more compact
    expected_linear_dist = n * 3.8  # ~3.8 Å per residue in extended chain
    compactness_score = max(0.0, min(100.0, (1 - max_dist / max(expected_linear_dist, 1)) * 100))
    # clustering
    clustering_score = round(contact_density, 1)
    # folding tendency
    if contact_density > 30:
        folding_tendency = "Compact / Well-folded"
    elif contact_density > 15:
        folding_tendency = "Semi-compact / Partially folded"
    else:
        folding_tendency = "Extended / Loosely packed"

    return {
        "n": n, "mean_dist": round(mean_dist, 2), "median_dist": round(median_dist, 2),
        "max_dist": round(max_dist, 2), "min_dist": round(min_dist, 2),
        "std_dist": round(std_dist, 2), "long_range_contacts": long_range,
        "contact_density": round(contact_density, 1),
        "compactness_score": round(compactness_score, 1),
        "folding_tendency": folding_tendency, "clustering_score": clustering_score,
    }


def compute_contact_stats(cm_matrix: np.ndarray, threshold: float = 8.0) -> dict:
    """Dynamic contact map statistics."""
    n = cm_matrix.shape[0]
    if n < 2:
        return {"n": n, "total_contacts": 0, "contact_density": 0.0,
                "local_contacts": 0, "long_range_contacts": 0,
                "connectivity_score": 0.0, "hub_residues": []}
    mask = ~np.eye(n, dtype=bool)
    total_contacts = int(cm_matrix[mask].sum()) // 2
    total_possible = (n * (n-1)) // 2
    contact_density = total_contacts / max(total_possible, 1) * 100
    local_contacts = sum(1 for i in range(n) for j in range(n)
                         if 0 < abs(i-j) <= 3 and cm_matrix[i,j]==1) // 2
    long_range_contacts = sum(1 for i in range(n) for j in range(n)
                              if abs(i-j) > 3 and cm_matrix[i,j]==1) // 2
    # hub residues: top 3 most connected
    row_sums = cm_matrix.sum(axis=1) - 1  # exclude self
    hub_threshold = np.percentile(row_sums, 75) if len(row_sums) > 3 else 0
    hub_residues = [int(i) for i in np.where(row_sums >= hub_threshold)[0]][:5]
    connectivity_score = round(contact_density, 1)

    return {
        "n": n, "total_contacts": total_contacts, "contact_density": round(contact_density, 1),
        "local_contacts": local_contacts, "long_range_contacts": long_range_contacts,
        "connectivity_score": connectivity_score, "hub_residues": hub_residues,
    }


def compute_sasa_stats(pr_sasa: dict, sasa_total, n_residues: int) -> dict:
    """Dynamic SASA statistics."""
    if not pr_sasa or sasa_total is None:
        return {
            "total_sasa": 0.0, "avg_sasa": 0.0, "median_sasa": 0.0,
            "sasa_variance": 0.0, "sasa_std": 0.0,
            "exposed_pct": 0.0, "buried_pct": 0.0, "exposure_index": 0.0,
            "classification": "Undetermined",
        }
    vals = list(pr_sasa.values())
    avg_sasa    = sasa_total / max(n_residues, 1)
    median_sasa = float(np.median(vals))
    sasa_var    = float(np.var(vals))
    sasa_std    = float(np.std(vals))
    exposed     = sum(1 for v in vals if v >= median_sasa)
    buried      = len(vals) - exposed
    exposed_pct = exposed / len(vals) * 100 if vals else 0.0
    buried_pct  = buried  / len(vals) * 100 if vals else 0.0
    exposure_index = round(avg_sasa / max(median_sasa, 1), 2)
    # classify based on avg SASA per residue
    if avg_sasa > 120:
        classification = "Highly solvent-exposed"
    elif avg_sasa > 80:
        classification = "Moderately exposed"
    elif avg_sasa > 40:
        classification = "Partially buried"
    else:
        classification = "Compact / Buried core"

    return {
        "total_sasa": round(sasa_total, 1), "avg_sasa": round(avg_sasa, 1),
        "median_sasa": round(median_sasa, 1), "sasa_variance": round(sasa_var, 1),
        "sasa_std": round(sasa_std, 1), "exposed_pct": round(exposed_pct, 1),
        "buried_pct": round(buried_pct, 1), "exposure_index": exposure_index,
        "classification": classification,
    }


def compute_hydrophobicity_stats(seq: str) -> dict:
    """Dynamic hydrophobicity statistics with cluster detection."""
    if not seq:
        return {
            "gravy": 0.0, "hydrophobic_clusters": [], "hydrophilic_stretches": [],
            "max_cluster_length": 0, "membrane_affinity": "Low",
            "solubility_tendency": "Unknown", "aggregation_risk": "Unknown",
            "hydrophobic_moment": 0.0,
        }
    vals = [KD_SCALE.get(a, 0) for a in seq]
    gravy = sum(vals) / len(vals)

    # Cluster detection: runs of >=3 consecutive hydrophobic residues
    hydrophobic_clusters = []
    hydrophilic_stretches = []
    i = 0
    while i < len(seq):
        if vals[i] > 1.0:
            j = i
            while j < len(seq) and vals[j] > 1.0:
                j += 1
            if j - i >= 2:
                hydrophobic_clusters.append((i, j-1, seq[i:j]))
            i = j
        elif vals[i] < -1.0:
            j = i
            while j < len(seq) and vals[j] < -1.0:
                j += 1
            if j - i >= 2:
                hydrophilic_stretches.append((i, j-1, seq[i:j]))
            i = j
        else:
            i += 1

    max_cluster_length = max((c[1]-c[0]+1 for c in hydrophobic_clusters), default=0)

    # Hydrophobic moment (simplified, using alpha-helix periodicity ~100°)
    if len(seq) >= 4:
        angle = 100.0
        hm_x = sum(vals[i] * np.cos(np.radians(angle * i)) for i in range(len(vals)))
        hm_y = sum(vals[i] * np.sin(np.radians(angle * i)) for i in range(len(vals)))
        hydrophobic_moment = round(np.sqrt(hm_x**2 + hm_y**2) / len(vals), 3)
    else:
        hydrophobic_moment = 0.0

    # Classifications based on actual values
    if gravy > 1.0 or max_cluster_length >= 5:
        membrane_affinity = "High"
    elif gravy > 0.3 or max_cluster_length >= 3:
        membrane_affinity = "Moderate"
    else:
        membrane_affinity = "Low"

    if gravy < -0.5 and len(hydrophobic_clusters) == 0:
        solubility_tendency = "Highly soluble"
    elif gravy < 0.0:
        solubility_tendency = "Likely soluble"
    elif gravy < 0.5:
        solubility_tendency = "Borderline solubility"
    else:
        solubility_tendency = "Potential solubility issues"

    if len(hydrophobic_clusters) >= 3 or max_cluster_length >= 6:
        aggregation_risk = "High"
    elif len(hydrophobic_clusters) >= 1 or max_cluster_length >= 3:
        aggregation_risk = "Moderate"
    else:
        aggregation_risk = "Low"

    return {
        "gravy": round(gravy, 3), "hydrophobic_clusters": hydrophobic_clusters,
        "hydrophilic_stretches": hydrophilic_stretches,
        "max_cluster_length": max_cluster_length,
        "membrane_affinity": membrane_affinity,
        "solubility_tendency": solubility_tendency,
        "aggregation_risk": aggregation_risk,
        "hydrophobic_moment": hydrophobic_moment,
    }


def compute_charge_stats(seq: str) -> dict:
    """Dynamic charge statistics."""
    pos = sum(1 for a in seq if a in "KRH")
    neg = sum(1 for a in seq if a in "DE")
    neu = len(seq) - pos - neg
    net = pos - neg
    charge_density = net / len(seq) if seq else 0.0
    charge_asymmetry = abs(pos - neg) / max(pos + neg, 1)

    # charge clustering
    charge_seq = [1 if a in "KRH" else (-1 if a in "DE" else 0) for a in seq]
    clusters = []
    i = 0
    while i < len(charge_seq):
        if charge_seq[i] != 0:
            j = i
            sign = charge_seq[i]
            while j < len(charge_seq) and charge_seq[j] == sign:
                j += 1
            if j - i >= 2:
                clusters.append((i, j-1, "+" if sign > 0 else "-"))
            i = j
        else:
            i += 1

    if net > 2 or (pos > 0 and net / max(pos+neg, 1) > 0.5):
        classification = "Cationic"
        membrane_interaction = "Strong affinity for anionic membranes"
    elif net < -2 or (neg > 0 and -net / max(pos+neg, 1) > 0.5):
        classification = "Anionic"
        membrane_interaction = "Repelled by anionic membranes; may interact with cationic surfaces"
    elif abs(net) <= 1:
        classification = "Near-neutral"
        membrane_interaction = "Minimal electrostatic-driven membrane interaction"
    else:
        classification = "Weakly charged"
        membrane_interaction = "Moderate electrostatic interactions"

    return {
        "pos": pos, "neg": neg, "neu": neu, "net": net,
        "charge_density": round(charge_density, 3),
        "charge_asymmetry": round(charge_asymmetry, 3),
        "clusters": clusters, "classification": classification,
        "membrane_interaction": membrane_interaction,
    }


def compute_aa_composition_stats(seq: str) -> dict:
    """Dynamic amino acid composition statistics."""
    c = Counter(seq)
    L = len(seq)
    dominant_aa = max(c, key=c.get) if c else "N/A"
    dominant_count = c.get(dominant_aa, 0)

    aromatic_richness = sum(c[a] for a in "FWY") / L * 100
    polar_richness    = sum(c[a] for a in "STNQ") / L * 100
    hydrophobic_richness = sum(c[a] for a in "AILMFWV") / L * 100

    # compositional entropy
    probs = [c[a]/L for a in c if c[a] > 0]
    comp_entropy = -sum(p * np.log2(p) for p in probs)

    # dominant group
    groups = {
        "Hydrophobic": sum(c[a] for a in "AILMFWV"),
        "Polar": sum(c[a] for a in "STNQ"),
        "Charged": sum(c[a] for a in "DEKRH"),
        "Aromatic": sum(c[a] for a in "FWY"),
        "Tiny": sum(c[a] for a in "AGSC"),
    }
    dominant_group = max(groups, key=groups.get)

    # biological implication
    if aromatic_richness > 20:
        bio_note = "High aromatic content may promote π-stacking and structural rigidity."
    elif hydrophobic_richness > 60:
        bio_note = "Hydrophobic-dominated sequence; may prefer membrane or buried environments."
    elif polar_richness > 40:
        bio_note = "Polar-rich composition supports aqueous solubility and receptor interactions."
    else:
        bio_note = "Balanced composition; versatile functional potential."

    return {
        "dominant_aa": dominant_aa, "dominant_count": dominant_count,
        "dominant_pct": round(dominant_count / L * 100, 1),
        "aromatic_richness": round(aromatic_richness, 1),
        "polar_richness": round(polar_richness, 1),
        "hydrophobic_richness": round(hydrophobic_richness, 1),
        "comp_entropy": round(comp_entropy, 3),
        "dominant_group": dominant_group, "bio_note": bio_note,
        "group_counts": groups,
    }


def compute_plddt_stats(plddt_vals: list) -> dict:
    """Dynamic pLDDT statistics."""
    if not plddt_vals or max(plddt_vals) <= 1.0:
        return {}
    vals = np.array(plddt_vals)
    mean_pl   = float(np.mean(vals))
    median_pl = float(np.median(vals))
    min_pl    = float(np.min(vals))
    max_pl    = float(np.max(vals))
    conf_var  = float(np.var(vals))
    conf_std  = float(np.std(vals))
    vhigh = int(np.sum(vals >= 90))
    high  = int(np.sum((vals >= 70) & (vals < 90)))
    med   = int(np.sum((vals >= 50) & (vals < 70)))
    low   = int(np.sum(vals < 50))

    # hotspot detection: residues with pLDDT > mean + std
    hotspot_threshold = mean_pl + conf_std
    hotspot_indices = [int(i) for i in np.where(vals >= hotspot_threshold)[0]]

    # confidence drops: residues with pLDDT < mean - std
    drop_threshold = mean_pl - conf_std
    drop_indices = [int(i) for i in np.where(vals < drop_threshold)[0]]

    # flexible / uncertain regions: runs of low pLDDT
    flexible_regions = []
    i = 0
    while i < len(vals):
        if vals[i] < 50:
            j = i
            while j < len(vals) and vals[j] < 50:
                j += 1
            if j - i >= 2:
                flexible_regions.append((i, j-1))
            i = j
        else:
            i += 1

    # overall confidence label
    lbl, col = plddt_label(mean_pl)

    return {
        "mean": round(mean_pl, 1), "median": round(median_pl, 1),
        "min": round(min_pl, 1), "max": round(max_pl, 1),
        "variance": round(conf_var, 1), "std": round(conf_std, 1),
        "very_high": vhigh, "high": high, "medium": med, "low": low,
        "hotspot_indices": hotspot_indices, "drop_indices": drop_indices,
        "flexible_regions": flexible_regions, "label": lbl, "color": col,
        "hotspot_threshold": round(hotspot_threshold, 1),
        "drop_threshold": round(drop_threshold, 1),
    }


def compute_rg_stats(rg: float, n_residues: int) -> dict:
    """Dynamic radius of gyration analysis."""
    if rg is None or n_residues < 2:
        return {"rg": None, "normalized_rg": None, "compactness_score": 0.0,
                "packing_efficiency": 0.0, "classification": "Undetermined"}
    # Expected Rg for random coil: Rg ~ 2.2 * N^0.60
    expected_rg_coil = 2.2 * (n_residues ** 0.60)
    # Expected Rg for globular: Rg ~ 2.2 * N^(1/3)
    expected_rg_glob = 2.2 * (n_residues ** (1/3))
    normalized_rg = rg / max(expected_rg_coil, 1)
    compactness_score = max(0.0, min(100.0, (1 - normalized_rg) * 100))
    packing_efficiency = max(0.0, min(100.0,
        (expected_rg_coil - rg) / max(expected_rg_coil - expected_rg_glob, 0.01) * 100))

    if rg <= expected_rg_glob * 1.1:
        classification = "Globular / Highly compact"
    elif rg <= expected_rg_coil * 0.7:
        classification = "Semi-compact"
    elif rg <= expected_rg_coil * 0.9:
        classification = "Partially extended"
    else:
        classification = "Extended / Unstructured"

    return {
        "rg": round(rg, 2), "normalized_rg": round(normalized_rg, 3),
        "compactness_score": round(compactness_score, 1),
        "packing_efficiency": round(packing_efficiency, 1),
        "expected_rg_coil": round(expected_rg_coil, 2),
        "expected_rg_glob": round(expected_rg_glob, 2),
        "classification": classification,
    }


def compute_hbond_stats(hbonds: int, n_residues: int) -> dict:
    """Dynamic hydrogen bond analysis."""
    if n_residues < 1:
        return {"count": 0, "per_residue": 0.0, "density": 0.0,
                "stabilization_index": 0.0, "interpretation": "Insufficient data"}
    per_residue = hbonds / max(n_residues, 1)
    density     = per_residue
    # stabilization: >1.5 H-bonds/residue is well-stabilized
    stabilization_index = min(100.0, per_residue / 2.0 * 100)
    if per_residue >= 2.0:
        interpretation = "Highly stabilized by H-bond network; likely well-folded helical or sheet content."
    elif per_residue >= 1.0:
        interpretation = "Moderate H-bond density; partial secondary structure stabilization."
    elif per_residue >= 0.5:
        interpretation = "Low H-bond density; predominantly coil or flexible regions."
    else:
        interpretation = "Minimal H-bonds detected; possibly disordered or very short peptide."
    return {
        "count": hbonds, "per_residue": round(per_residue, 2),
        "density": round(density, 2), "stabilization_index": round(stabilization_index, 1),
        "interpretation": interpretation,
    }


def compute_disulfide_stats(ss_bonds: int, seq: str) -> dict:
    """Dynamic disulfide bond analysis."""
    cys_count = seq.count("C")
    cys_utilization = (ss_bonds * 2 / max(cys_count, 1)) * 100 if cys_count > 0 else 0.0
    stability_contribution = min(100.0, ss_bonds * 25.0)
    if ss_bonds == 0:
        interpretation = "No disulfide bonds; stability depends on non-covalent interactions."
    elif ss_bonds == 1:
        interpretation = "Single disulfide bond provides moderate structural constraint."
    else:
        interpretation = f"{ss_bonds} disulfide bonds form a robust covalent scaffold; high structural rigidity."
    return {
        "count": ss_bonds, "cys_count": cys_count,
        "cys_utilization": round(cys_utilization, 1),
        "stability_contribution": round(stability_contribution, 1),
        "interpretation": interpretation,
    }


def detect_sequence_motifs(seq: str) -> dict:
    """Detect motifs, repeats, and clusters in sequence."""
    # Charge clusters
    charge_clusters = []
    i = 0
    while i < len(seq):
        if seq[i] in "KRH":
            j = i
            while j < len(seq) and seq[j] in "KRH":
                j += 1
            if j - i >= 2:
                charge_clusters.append(("positive", i, j-1, seq[i:j]))
            i = j
        elif seq[i] in "DE":
            j = i
            while j < len(seq) and seq[j] in "DE":
                j += 1
            if j - i >= 2:
                charge_clusters.append(("negative", i, j-1, seq[i:j]))
            i = j
        else:
            i += 1

    # Aromatic clusters
    aromatic_clusters = []
    i = 0
    while i < len(seq):
        if seq[i] in "FWY":
            j = i
            while j < len(seq) and seq[j] in "FWY":
                j += 1
            if j - i >= 2:
                aromatic_clusters.append((i, j-1, seq[i:j]))
            i = j
        else:
            i += 1

    # Simple repeat detection (di/tripeptide repeats)
    repeats = []
    for motif_len in [2, 3]:
        motif_counts = Counter(seq[i:i+motif_len] for i in range(len(seq)-motif_len+1))
        for motif, count in motif_counts.items():
            if count >= 3:
                repeats.append((motif, count))
    repeats.sort(key=lambda x: -x[1])

    return {
        "charge_clusters": charge_clusters,
        "aromatic_clusters": aromatic_clusters,
        "repeats": repeats[:5],
    }


# ==========================================================
# SECTION 12B - ADVANCED COMPOSITE SCORES
# ==========================================================

def compute_folding_propensity(contact_stats: dict, dssp: dict, hyd_stats: dict, rg_stats: dict) -> dict:
    """Folding propensity score 0–100."""
    score = 0.0
    details = {}
    # Contact density contribution (0–30)
    cd = contact_stats.get("contact_density", 0)
    cd_pts = min(30.0, cd * 0.6)
    score += cd_pts
    details["Contact density"] = round(cd_pts, 1)
    # DSSP structured content (0–30)
    struct = dssp.get("helix", 0) + dssp.get("sheet", 0)
    dssp_pts = min(30.0, struct * 0.30)
    score += dssp_pts
    details["Secondary structure"] = round(dssp_pts, 1)
    # Hydrophobicity (0–20) — moderate GRAVY is optimal for folding
    gv = hyd_stats.get("gravy", 0)
    hyd_pts = max(0.0, 20.0 - abs(gv - 0.5) * 10)
    score += hyd_pts
    details["Hydrophobicity balance"] = round(hyd_pts, 1)
    # Compactness (0–20)
    cs = rg_stats.get("compactness_score", 0) if rg_stats.get("rg") else 10.0
    rg_pts = cs * 0.20
    score += rg_pts
    details["Compactness"] = round(rg_pts, 1)
    score = min(100.0, score)
    if score >= 75: label = "High folding propensity"
    elif score >= 50: label = "Moderate folding propensity"
    elif score >= 25: label = "Low folding propensity"
    else: label = "Likely disordered"
    return {"score": round(score, 1), "label": label, "breakdown": details}


def compute_solubility_propensity(sasa_stats: dict, charge_stats: dict, hyd_stats: dict) -> dict:
    """Solubility propensity score 0–100."""
    score = 0.0
    details = {}
    # SASA contribution (0–40)
    avg_sasa = sasa_stats.get("avg_sasa", 60)
    sasa_pts = min(40.0, avg_sasa * 0.25)
    score += sasa_pts
    details["Solvent accessibility"] = round(sasa_pts, 1)
    # Charge (0–30) — charged residues improve solubility
    abs_charge = abs(charge_stats.get("net", 0))
    charge_pts = min(30.0, abs_charge * 5.0)
    score += charge_pts
    details["Charge contribution"] = round(charge_pts, 1)
    # Hydrophobicity (0–30) — negative GRAVY improves solubility
    gv = hyd_stats.get("gravy", 0)
    hyd_pts = max(0.0, 30.0 - max(gv, 0) * 15.0)
    score += hyd_pts
    details["Hydrophilicity"] = round(hyd_pts, 1)
    score = min(100.0, score)
    if score >= 70: label = "Likely soluble"
    elif score >= 45: label = "Borderline solubility"
    else: label = "Potential insolubility"
    return {"score": round(score, 1), "label": label, "breakdown": details}


def compute_aggregation_risk(hyd_stats: dict, sasa_stats: dict, aa_stats: dict) -> dict:
    """Aggregation risk score 0–100 (higher = higher risk)."""
    score = 0.0
    details = {}
    n_clusters = len(hyd_stats.get("hydrophobic_clusters", []))
    cluster_pts = min(40.0, n_clusters * 10.0)
    score += cluster_pts
    details["Hydrophobic clusters"] = round(cluster_pts, 1)
    exposed_pct = sasa_stats.get("exposed_pct", 50)
    exp_pts = max(0.0, 30.0 - exposed_pct * 0.3)
    score += exp_pts
    details["Surface burial"] = round(exp_pts, 1)
    hyd_rich = aa_stats.get("hydrophobic_richness", 40)
    hyd_pts = min(30.0, hyd_rich * 0.40)
    score += hyd_pts
    details["Hydrophobic richness"] = round(hyd_pts, 1)
    score = min(100.0, score)
    if score >= 65: label = "High aggregation risk"
    elif score >= 35: label = "Moderate aggregation risk"
    else: label = "Low aggregation risk"
    return {"score": round(score, 1), "label": label, "breakdown": details}


def compute_stability_score(hbond_stats: dict, ss_stats: dict, rg_stats: dict, dssp: dict) -> dict:
    """Structural stability score 0–100."""
    score = 0.0
    details = {}
    hb_pts = min(30.0, hbond_stats.get("stabilization_index", 0) * 0.30)
    score += hb_pts
    details["H-bond network"] = round(hb_pts, 1)
    ss_pts = min(20.0, ss_stats.get("stability_contribution", 0) * 0.20)
    score += ss_pts
    details["Disulfide bonds"] = round(ss_pts, 1)
    cs = rg_stats.get("compactness_score", 0) if rg_stats.get("rg") else 10.0
    rg_pts = cs * 0.25
    score += rg_pts
    details["Compactness"] = round(rg_pts, 1)
    struct = dssp.get("helix", 0) + dssp.get("sheet", 0)
    dssp_pts = min(25.0, struct * 0.25)
    score += dssp_pts
    details["Secondary structure"] = round(dssp_pts, 1)
    score = min(100.0, score)
    if score >= 75: label = "Highly stable"
    elif score >= 50: label = "Moderately stable"
    elif score >= 25: label = "Marginally stable"
    else: label = "Likely unstable"
    return {"score": round(score, 1), "label": label, "breakdown": details}


def compute_flexibility_score(plddt_stats: dict, dssp: dict, rama_stats: dict) -> dict:
    """Flexibility score 0–100 (higher = more flexible)."""
    score = 0.0
    details = {}
    # pLDDT: lower confidence → more flexibility
    if plddt_stats:
        flex_from_plddt = max(0.0, 100.0 - plddt_stats.get("mean", 70))
        plddt_pts = min(40.0, flex_from_plddt * 0.40)
    else:
        plddt_pts = 20.0  # neutral
    score += plddt_pts
    details["pLDDT uncertainty"] = round(plddt_pts, 1)
    coil_pct = dssp.get("coil", 0)
    coil_pts = min(35.0, coil_pct * 0.35)
    score += coil_pts
    details["Coil content"] = round(coil_pts, 1)
    conf_flex = rama_stats.get("conformational_flexibility", 0)
    rama_pts = min(25.0, conf_flex * 0.10)
    score += rama_pts
    details["Backbone dispersion"] = round(rama_pts, 1)
    score = min(100.0, score)
    if score >= 70: label = "Highly flexible"
    elif score >= 45: label = "Moderately flexible"
    elif score >= 20: label = "Relatively rigid"
    else: label = "Rigid"
    return {"score": round(score, 1), "label": label, "breakdown": details}


def compute_complexity_score(contact_stats: dict, dssp: dict, distance_stats: dict) -> dict:
    """Structural complexity score 0–100."""
    score = 0.0
    details = {}
    cd = contact_stats.get("contact_density", 0)
    net_pts = min(40.0, cd * 0.80)
    score += net_pts
    details["Contact network"] = round(net_pts, 1)
    ss_ent = dssp.get("ss_entropy", 0)
    ent_pts = min(35.0, ss_ent * 25.0)
    score += ent_pts
    details["SS diversity (entropy)"] = round(ent_pts, 1)
    lr = distance_stats.get("long_range_contacts", 0)
    lr_pts = min(25.0, lr * 1.5)
    score += lr_pts
    details["Long-range contacts"] = round(lr_pts, 1)
    score = min(100.0, score)
    if score >= 70: label = "Complex / Highly connected"
    elif score >= 45: label = "Moderately complex"
    elif score >= 20: label = "Simple topology"
    else: label = "Minimal structure"
    return {"score": round(score, 1), "label": label, "breakdown": details}


def structural_quality_score_v2(
    rama_stats: dict, dssp: dict, sasa_stats: dict, rg_stats: dict,
    contact_stats: dict, hbond_stats: dict, ss_stats: dict,
    plddt_stats: dict, hyd_stats: dict,
    folding_prop: dict, stability: dict, n_residues: int = 1,
) -> tuple:
    """Fully dynamic weighted structural quality score (0–100)."""
    score = 0.0
    breakdown = {}

    # Ramachandran (0–20)
    allowed_pct = rama_stats.get("allowed_pct", 0) if rama_stats.get("total", 0) > 0 else 50.0
    rama_pts = allowed_pct * 0.20
    score += rama_pts
    breakdown["Ramachandran"] = round(rama_pts, 1)

    # DSSP (0–15)
    struct_pct = dssp.get("helix", 0) + dssp.get("sheet", 0)
    dssp_pts = min(15.0, struct_pct * 0.15)
    score += dssp_pts
    breakdown["Secondary Structure"] = round(dssp_pts, 1)

    # SASA (0–10)
    avg_sasa = sasa_stats.get("avg_sasa", 80) if sasa_stats.get("total_sasa", 0) > 0 else 80.0
    sasa_pts = max(0.0, 10.0 - abs(avg_sasa - 80) * 0.08)
    score += sasa_pts
    breakdown["SASA Distribution"] = round(sasa_pts, 1)

    # Radius of gyration (0–10)
    compactness = rg_stats.get("compactness_score", 0) if rg_stats.get("rg") else 5.0
    rg_pts = min(10.0, compactness * 0.10)
    score += rg_pts
    breakdown["Compactness (Rg)"] = round(rg_pts, 1)

    # Contact density (0–10)
    cd = contact_stats.get("contact_density", 0)
    cd_pts = min(10.0, cd * 0.20)
    score += cd_pts
    breakdown["Contact Density"] = round(cd_pts, 1)

    # H-bonds (0–10)
    hb_pts = min(10.0, hbond_stats.get("stabilization_index", 0) * 0.10)
    score += hb_pts
    breakdown["H-Bond Network"] = round(hb_pts, 1)

    # Disulfide bonds (0–5)
    ss_pts = min(5.0, ss_stats.get("stability_contribution", 0) * 0.05)
    score += ss_pts
    breakdown["Disulfide Bonds"] = round(ss_pts, 1)

    # pLDDT (0–10)
    if plddt_stats and plddt_stats.get("mean"):
        plddt_pts = min(10.0, plddt_stats["mean"] * 0.10)
    else:
        plddt_pts = 5.0  # neutral
    score += plddt_pts
    breakdown["pLDDT Confidence"] = round(plddt_pts, 1)

    # Hydrophobic core (0–5)
    gv = hyd_stats.get("gravy", 0)
    hyd_pts = max(0.0, 5.0 - abs(gv) * 2.0)
    score += hyd_pts
    breakdown["Hydrophobic Balance"] = round(hyd_pts, 1)

    # Folding propensity (0–5)
    fp_pts = min(5.0, folding_prop.get("score", 0) * 0.05)
    score += fp_pts
    breakdown["Folding Propensity"] = round(fp_pts, 1)

    score = min(100.0, score)
    if score >= 80: label = "Excellent"
    elif score >= 60: label = "Good"
    elif score >= 40: label = "Moderate"
    else: label = "Poor"

    return round(score, 1), label, breakdown


# ==========================================================
# SECTION 13 - DYNAMIC CAPTION SYSTEM
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


def caption_ramachandran_v2(rama_stats: dict, seq: str = "") -> str:
    if rama_stats["total"] == 0:
        return "No φ/ψ angles extracted — peptide needs ≥3 residues with complete backbone atoms."
    n = rama_stats["total"]
    dom = rama_stats["dominant_region"]
    outlier_note = (
        f"<strong style='color:#c0392b;'>{rama_stats['outlier_count']} outliers "
        f"({rama_stats['outlier_pct']:.0f}%)</strong> detected — backbone strain present."
        if rama_stats["outlier_count"] > 0
        else "No outliers detected — all residues in allowed regions."
    )
    flex_note = (
        "High conformational flexibility (φ/ψ dispersion: "
        f"<strong>{rama_stats['conformational_flexibility']:.1f}°</strong>) — "
        "peptide backbone is geometrically diverse."
        if rama_stats["conformational_flexibility"] > 60
        else f"Moderate conformational rigidity (dispersion: {rama_stats['conformational_flexibility']:.1f}°)."
    )
    return (
        f"Backbone torsion analysis of <strong>{n} residue(s)</strong>"
        f"{f' from sequence <em>{seq[:12]}…</em>' if seq else ''}.<br><br>"
        f"<strong>α-Helix region:</strong> {rama_stats['helix_pct']}% ({rama_stats['helix_count']} res) | "
        f"<strong>β-Sheet region:</strong> {rama_stats['sheet_pct']}% ({rama_stats['sheet_count']} res) | "
        f"<strong>L-Helix:</strong> {rama_stats['lhelix_pct']}% | "
        f"<strong>Outliers:</strong> {rama_stats['outlier_pct']}%<br><br>"
        f"Dominant backbone conformation: <strong>{dom}</strong>.<br>"
        f"Allowed region occupancy: <strong>{rama_stats['allowed_pct']}%</strong>. {outlier_note}<br>"
        f"Backbone strain score: <strong>{rama_stats['backbone_strain']:.1f}/100</strong> (lower = better). {flex_note}"
    )


def caption_distance_map_v2(dist_stats: dict, seq: str = "") -> str:
    n = dist_stats["n"]
    if n < 2:
        return "Distance map unavailable — fewer than 2 Cα atoms detected."
    fold = dist_stats["folding_tendency"]
    return (
        f"Cα–Cα pairwise distance matrix for <strong>{n} residues</strong> "
        f"(mean: <strong>{dist_stats['mean_dist']:.1f} Å</strong>, "
        f"range: {dist_stats['min_dist']:.1f}–{dist_stats['max_dist']:.1f} Å).<br><br>"
        f"<strong>Contact density:</strong> {dist_stats['contact_density']:.1f}% of residue pairs "
        f"within 8 Å. <strong>Long-range contacts:</strong> {dist_stats['long_range_contacts']}.<br>"
        f"Distance variability (σ = {dist_stats['std_dist']:.1f} Å) — "
        f"{'heterogeneous spacing' if dist_stats['std_dist'] > 5 else 'uniform spacing'}.<br><br>"
        f"<strong>Folding assessment:</strong> <em>{fold}</em>. "
        f"Compactness score: <strong>{dist_stats['compactness_score']:.1f}/100</strong>."
    )


def caption_contact_map_v2(cs: dict, threshold: float = 8.0) -> str:
    n = cs["n"]
    if n < 2:
        return "Contact map unavailable — fewer than 2 residues."
    label = "compact / well-folded" if cs["contact_density"] > 30 else ("partially folded" if cs["contact_density"] > 15 else "extended / loosely packed")
    hub_note = (
        f"Hub residues (highest connectivity): positions {', '.join(str(h+1) for h in cs['hub_residues'][:3])}."
        if cs["hub_residues"] else "No hub residues identified."
    )
    return (
        f"Binary contact map at <strong>{threshold} Å threshold</strong> for {n} residues.<br><br>"
        f"<strong>Total contacts:</strong> {cs['total_contacts']} | "
        f"<strong>Local (|i-j|≤3):</strong> {cs['local_contacts']} | "
        f"<strong>Long-range (|i-j|>3):</strong> {cs['long_range_contacts']}<br>"
        f"<strong>Contact density:</strong> {cs['contact_density']:.1f}% → structure is "
        f"<strong>{label}</strong>.<br>"
        f"<strong>Connectivity score:</strong> {cs['connectivity_score']:.1f}. {hub_note}"
    )


def caption_sasa_v2(sasa_stats: dict, pr_sasa: dict) -> str:
    if sasa_stats.get("total_sasa", 0) == 0:
        return "SASA could not be computed for this structure."
    cls = sasa_stats["classification"]
    top_exposed = sorted(pr_sasa.items(), key=lambda x: -x[1])[:3] if pr_sasa else []
    top_buried  = sorted(pr_sasa.items(), key=lambda x: x[1])[:3] if pr_sasa else []
    exp_note = ", ".join(f"{k} ({v:.0f} Å²)" for k, v in top_exposed) if top_exposed else "N/A"
    bur_note = ", ".join(f"{k} ({v:.0f} Å²)" for k, v in top_buried) if top_buried else "N/A"
    return (
        f"Solvent-accessible surface area analysis.<br><br>"
        f"<strong>Total SASA:</strong> {sasa_stats['total_sasa']:.1f} Å² | "
        f"<strong>Average/residue:</strong> {sasa_stats['avg_sasa']:.1f} Å² | "
        f"<strong>Median:</strong> {sasa_stats['median_sasa']:.1f} Å² | "
        f"<strong>σ:</strong> {sasa_stats['sasa_std']:.1f} Å²<br>"
        f"<strong>Exposed:</strong> {sasa_stats['exposed_pct']:.1f}% | "
        f"<strong>Buried:</strong> {sasa_stats['buried_pct']:.1f}% | "
        f"<strong>Exposure index:</strong> {sasa_stats['exposure_index']}<br><br>"
        f"<strong>Classification:</strong> <em>{cls}</em>.<br>"
        f"Most exposed: {exp_note}.<br>Most buried: {bur_note}."
    )


def caption_dssp_v2(dssp: dict) -> str:
    total = dssp.get("total", 0)
    if total == 0:
        return "Secondary structure could not be estimated — insufficient backbone angles."
    dom = ("α-helical" if dssp["helix"] > dssp["sheet"] and dssp["helix"] > dssp["coil"]
           else "β-sheet" if dssp["sheet"] > dssp["helix"] and dssp["sheet"] > dssp["coil"]
           else "random coil / mixed")
    hs = dssp.get("hs_ratio", 0)
    hs_note = (
        f"Helix-to-sheet ratio: <strong>{hs:.2f}</strong> — "
        + ("helix-dominant." if hs > 1.5 else "sheet-dominant." if hs < 0.7 else "balanced α/β.")
    )
    ent = dssp.get("ss_entropy", 0)
    ent_note = (
        f"SS entropy: <strong>{ent:.3f} bits</strong> — "
        + ("diverse secondary structure composition." if ent > 1.0 else "homogeneous secondary structure.")
    )
    return (
        f"Secondary structure estimated from backbone φ/ψ angles ({total} residues).<br><br>"
        f"<strong>α-Helix:</strong> {dssp['helix']}% ({dssp['n_helix']} res) | "
        f"<strong>β-Sheet:</strong> {dssp['sheet']}% ({dssp['n_sheet']} res) | "
        f"<strong>Coil:</strong> {dssp['coil']}% ({dssp['n_coil']} res)<br><br>"
        f"Dominant structure: <strong>{dom}</strong>. {hs_note}<br>{ent_note}"
    )


def caption_hydrophobicity_v2(hyd_stats: dict, seq: str) -> str:
    gv = hyd_stats["gravy"]
    n_clusters = len(hyd_stats["hydrophobic_clusters"])
    cluster_detail = (
        f"<strong>{n_clusters} hydrophobic cluster(s)</strong> detected "
        + (f"(longest: {hyd_stats['max_cluster_length']} residues): "
           + ", ".join(f"pos {c[0]+1}–{c[1]+1} ({c[2]})" for c in hyd_stats["hydrophobic_clusters"][:3])
           if n_clusters > 0 else "")
    )
    return (
        f"Per-residue hydrophobicity (Kyte–Doolittle scale) for {len(seq)}-aa sequence.<br><br>"
        f"<strong>GRAVY:</strong> {gv:+.3f} "
        f"({'strongly hydrophobic' if gv>1.0 else 'mildly hydrophobic' if gv>0 else 'hydrophilic'} character)<br>"
        f"<strong>Hydrophobic moment:</strong> {hyd_stats['hydrophobic_moment']:.3f} (amphipathicity proxy)<br>"
        f"{cluster_detail}<br><br>"
        f"<strong>Membrane affinity:</strong> {hyd_stats['membrane_affinity']} | "
        f"<strong>Solubility tendency:</strong> {hyd_stats['solubility_tendency']} | "
        f"<strong>Aggregation risk:</strong> {hyd_stats['aggregation_risk']}"
    )


def caption_charge_v2(charge_stats: dict, seq: str) -> str:
    cls = charge_stats["classification"]
    n_clusters = len(charge_stats["clusters"])
    cluster_note = (
        f"{n_clusters} charge cluster(s): "
        + ", ".join(f"{'(+)' if c[0]=='positive' else '(-)'} pos {c[1]+1}–{c[2]+1} ({c[3]})"
                    for c in charge_stats["clusters"][:3])
        if n_clusters > 0 else "No charge clusters detected."
    )
    return (
        f"Charge distribution for {len(seq)}-aa sequence at pH 7.<br><br>"
        f"<strong>Positive (K/R/H):</strong> {charge_stats['pos']} | "
        f"<strong>Negative (D/E):</strong> {charge_stats['neg']} | "
        f"<strong>Net charge:</strong> {charge_stats['net']:+d}<br>"
        f"<strong>Charge density:</strong> {charge_stats['charge_density']:+.3f} per residue | "
        f"<strong>Charge asymmetry:</strong> {charge_stats['charge_asymmetry']:.3f}<br><br>"
        f"<strong>Classification:</strong> {cls}. {charge_stats['membrane_interaction']}.<br>"
        f"{cluster_note}"
    )


def caption_aa_composition_v2(aa_stats: dict, seq: str) -> str:
    return (
        f"Amino acid composition of {len(seq)}-aa sequence.<br><br>"
        f"<strong>Dominant residue:</strong> {aa_stats['dominant_aa']} "
        f"({aa_stats['dominant_count']} occurrences, {aa_stats['dominant_pct']:.1f}%)<br>"
        f"<strong>Dominant group:</strong> {aa_stats['dominant_group']}<br>"
        f"<strong>Hydrophobic:</strong> {aa_stats['hydrophobic_richness']:.1f}% | "
        f"<strong>Polar:</strong> {aa_stats['polar_richness']:.1f}% | "
        f"<strong>Aromatic:</strong> {aa_stats['aromatic_richness']:.1f}%<br>"
        f"<strong>Compositional entropy:</strong> {aa_stats['comp_entropy']:.3f} bits "
        f"({'high diversity' if aa_stats['comp_entropy'] > 3.5 else 'moderate diversity' if aa_stats['comp_entropy'] > 2.5 else 'low diversity'})<br><br>"
        f"<em>{aa_stats['bio_note']}</em>"
    )

def caption_plddt_v2(plddt_stats: dict, seq: str = "") -> str:
    if not plddt_stats:
        return (
            "pLDDT data not available for this structure "
            "(B-factor column does not contain AlphaFold confidence scores)."
        )

    lbl = plddt_stats["label"]
    col = plddt_stats["color"]

    n_flex = sum(
        r[1] - r[0] + 1
        for r in plddt_stats["flexible_regions"]
    )

    flex_note = (
        f"<strong>{n_flex} residues</strong> in "
        f"{len(plddt_stats['flexible_regions'])} flexible region(s)."
        if plddt_stats["flexible_regions"]
        else "No extended flexible regions detected."
    )

    hotspot_note = (
        f"<strong>{len(plddt_stats['hotspot_indices'])} confidence hotspot(s)</strong> "
        f"(pLDDT ≥ {plddt_stats['hotspot_threshold']:.0f})."
        if plddt_stats["hotspot_indices"]
        else "No high-confidence hotspots above threshold."
    )

    residue_count_text = ""
    if seq:
        residue_count_text = f" ({len(seq)} residues)"

    return (
        f"Per-residue pLDDT confidence profile{residue_count_text}.<br><br>"
        f"<strong>Mean pLDDT:</strong> "
        f"<span style='color:{col}'>{plddt_stats['mean']:.1f} — {lbl}</span> | "
        f"<strong>Median:</strong> {plddt_stats['median']:.1f} | "
        f"<strong>Range:</strong> {plddt_stats['min']:.1f}–{plddt_stats['max']:.1f} | "
        f"<strong>σ:</strong> {plddt_stats['std']:.1f}<br>"
        f"<strong>Very High (≥90):</strong> {plddt_stats['very_high']} | "
        f"<strong>High (70–90):</strong> {plddt_stats['high']} | "
        f"<strong>Medium (50–70):</strong> {plddt_stats['medium']} | "
        f"<strong>Low (&lt;50):</strong> {plddt_stats['low']}<br><br>"
        f"{hotspot_note} {flex_note}"
    )
def caption_rg_v2(rg_stats: dict, seq: str = "") -> str:
    if rg_stats.get("rg") is None:
        return "Radius of gyration could not be computed."
    cls = rg_stats["classification"]
    return (
        f"Radius of gyration analysis"
        f"{f' for {len(seq)}-aa structure' if seq else ''}.<br><br>"
        f"<strong>Rg:</strong> {rg_stats['rg']:.2f} Å | "
        f"<strong>Normalized Rg:</strong> {rg_stats['normalized_rg']:.3f} "
        f"(vs expected coil: {rg_stats['expected_rg_coil']:.2f} Å, "
        f"globular: {rg_stats['expected_rg_glob']:.2f} Å)<br>"
        f"<strong>Compactness score:</strong> {rg_stats['compactness_score']:.1f}/100 | "
        f"<strong>Packing efficiency:</strong> {rg_stats['packing_efficiency']:.1f}/100<br><br>"
        f"<strong>Classification:</strong> <em>{cls}</em>."
    )


def caption_hbond_v2(hbond_stats: dict) -> str:
    return (
        f"Hydrogen bond estimation (N···O distance ≤ 3.5 Å).<br><br>"
        f"<strong>Total H-bonds:</strong> {hbond_stats['count']} | "
        f"<strong>Per residue:</strong> {hbond_stats['per_residue']:.2f} | "
        f"<strong>Stabilization index:</strong> {hbond_stats['stabilization_index']:.1f}/100<br><br>"
        f"<em>{hbond_stats['interpretation']}</em>"
    )


def caption_disulfide_v2(ss_stats: dict) -> str:
    return (
        f"Disulfide bond analysis.<br><br>"
        f"<strong>Disulfide bonds:</strong> {ss_stats['count']} | "
        f"<strong>Cysteine residues:</strong> {ss_stats['cys_count']} | "
        f"<strong>Cys utilization:</strong> {ss_stats['cys_utilization']:.1f}%<br>"
        f"<strong>Stability contribution:</strong> {ss_stats['stability_contribution']:.1f}/100<br><br>"
        f"<em>{ss_stats['interpretation']}</em>"
    )


def caption_sequence_logo_v2(hyd_stats: dict, motifs: dict, seq: str) -> str:
    n_motifs = len(motifs.get("repeats", []))
    n_arom   = len(motifs.get("aromatic_clusters", []))
    n_chrg   = len(motifs.get("charge_clusters", []))
    motif_note = (
        f"<strong>{n_motifs} repeated motif(s)</strong>: "
        + ", ".join(f"{m} (×{c})" for m, c in motifs["repeats"][:3])
        if n_motifs > 0 else "No repeated dipeptide/tripeptide motifs detected."
    )
    return (
        f"Hydrophobicity-weighted sequence visualization for {len(seq)}-aa peptide.<br><br>"
        f"Bar height = |Kyte–Doolittle value|. Blue = hydrophobic, red = hydrophilic.<br>"
        f"<strong>GRAVY:</strong> {hyd_stats['gravy']:+.3f} | "
        f"<strong>Hydrophobic clusters:</strong> {len(hyd_stats['hydrophobic_clusters'])} | "
        f"<strong>Aromatic clusters:</strong> {n_arom} | "
        f"<strong>Charge clusters:</strong> {n_chrg}<br><br>"
        f"{motif_note}"
    )


# ==========================================================
# SECTION 14 - UPGRADED PLOT FUNCTIONS
# ==========================================================

def plot_plddt_v2(plddt_vals: list, plddt_stats: dict, seq: str = ""):
    C   = get_plot_colors()
    n   = len(plddt_vals)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.2), 5))
    apply_plot_style(fig, [ax])
    bar_colors = ["#12b886" if v>=90 else "#1a8fd1" if v>=70 else
                  "#f39c12" if v>=50 else "#c0392b" for v in plddt_vals]
    ax.bar(range(n), plddt_vals, color=bar_colors, width=0.85)

    # Dynamic threshold lines
    mean_pl = plddt_stats.get("mean", 0)
    ax.axhline(mean_pl, color=C["orange"], linestyle="-.", lw=2, alpha=0.9,
               label=f"Mean = {mean_pl:.1f}")
    for thresh, col, lbl in [(90,"#12b886","Very High (≥90)"),
                              (70,"#1a8fd1","High (≥70)"),
                              (50,"#f39c12","Medium (≥50)")]:
        ax.axhline(thresh, color=col, linestyle="--", lw=1.2, alpha=0.6, label=lbl)

    # Highlight hotspots
    for idx in plddt_stats.get("hotspot_indices", []):
        if idx < n:
            ax.bar(idx, plddt_vals[idx], color="#12b886", width=0.85,
                   edgecolor="white", linewidth=1.5)

    # Highlight flexible regions
    for start, end in plddt_stats.get("flexible_regions", []):
        ax.axvspan(start - 0.5, min(end + 0.5, n - 0.5),
                   color="#c0392b", alpha=0.12)
        ax.annotate("Flexible", xy=((start+end)/2, 15), ha="center",
                    fontsize=7, color=C["red"], style="italic")

    # Annotate confidence drops
    drops = plddt_stats.get("drop_indices", [])
    if drops and len(drops) <= 10:
        for idx in drops:
            if idx < n:
                ax.annotate("↓", xy=(idx, plddt_vals[idx]+2), ha="center",
                             fontsize=9, color=C["red"])

    ax.set_ylim(0, 105)
    ax.set_xlabel("Residue Index", fontsize=11, labelpad=8)
    ax.set_ylabel("pLDDT", fontsize=11, labelpad=8)
    lbl, lcol = plddt_label(mean_pl)
    ax.set_title(f"pLDDT Confidence — Mean: {mean_pl:.1f} ({lbl})", fontsize=13,
                 fontweight="bold", pad=12, color=C["text"])
    if seq and len(seq) == n and n <= 60:
        ax.set_xticks(range(n))
        ax.set_xticklabels(list(seq), fontsize=8)
    leg = ax.legend(fontsize=8, loc="lower right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_ramachandran_v2(phi_psi: list, rama_stats: dict, seq: str = ""):
    C   = get_plot_colors()
    fig, ax = plt.subplots(figsize=(7, 7))
    apply_plot_style(fig, [ax])

    # Static reference regions (scientific standard)
    ax.fill([-180,-180,-45,-45,-180], [-75,-45,-45,-75,-75], color="#4CAF50", alpha=0.25, label="α-helix region")
    ax.fill([-180,-180,-90,-90,-180], [90,180,180,90,90],    color="#2196F3", alpha=0.25, label="β-sheet region")
    ax.fill([45,45,90,90,45],         [0,90,90,0,0],         color="#FF9800", alpha=0.20, label="L-helix region")

    if phi_psi:
        phi_vals = [p for p, s in phi_psi]
        psi_vals = [s for p, s in phi_psi]
        outlier_set = set(rama_stats.get("outlier_indices", []))
        # color points: outliers in red, others by region
        colors_pts = []
        for i, (p, s) in enumerate(phi_psi):
            if i in outlier_set:
                colors_pts.append(C["red"])
            elif -180<=p<=-45 and -75<=s<=-15:
                colors_pts.append("#4CAF50")
            elif -180<=p<=-45 and (90<=s<=180 or -180<=s<=-150):
                colors_pts.append("#2196F3")
            elif 45<=p<=90 and 0<=s<=90:
                colors_pts.append("#FF9800")
            else:
                colors_pts.append(C["red"])
        ax.scatter(phi_vals, psi_vals, s=60, c=colors_pts, zorder=5,
                   edgecolors="white", linewidths=0.5)

        # Annotate outliers if few
        if 0 < len(outlier_set) <= 5:
            for idx in outlier_set:
                if idx < len(phi_psi):
                    p, s = phi_psi[idx]
                    label_txt = seq[idx] if seq and idx < len(seq) else str(idx+1)
                    ax.annotate(label_txt, (p, s), fontsize=8, color=C["red"],
                                xytext=(6, 4), textcoords="offset points", fontweight="bold")

        # Dynamic annotation box
        stats_text = (
            f"α: {rama_stats['helix_pct']:.0f}% "
            f"β: {rama_stats['sheet_pct']:.0f}% "
            f"Out: {rama_stats['outlier_pct']:.0f}%"
        )
        ax.annotate(stats_text, xy=(0.02, 0.97), xycoords="axes fraction",
                    fontsize=9, color=C["text"], va="top",
                    bbox=dict(boxstyle="round,pad=0.4", fc=C["fig_bg"], ec=C["grid"], alpha=0.90))

    ax.axhline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.axvline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi φ (°)", fontsize=12, labelpad=10)
    ax.set_ylabel("Psi ψ (°)", fontsize=12, labelpad=10)
    dom = rama_stats.get("dominant_region", "Undefined")
    ax.set_title(f"Ramachandran Plot — Dominant: {dom}", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, loc="upper right", facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    ax.set_xticks(range(-180, 181, 60)); ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    return fig


def plot_distance_map_v2(dist_matrix: np.ndarray, dist_stats: dict, seq: str = ""):
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
    im = sns.heatmap(dist_matrix, cmap="viridis", ax=ax,
                xticklabels=show_labels, yticklabels=show_labels,
                linewidths=0, cbar_kws={"label": "Distance (Å)"})
    # Annotate compactness
    fold = dist_stats.get("folding_tendency", "")
    cs   = dist_stats.get("compactness_score", 0)
    ax.annotate(f"Compactness: {cs:.1f}/100\n{fold}",
                xy=(0.01, 0.99), xycoords="axes fraction", va="top",
                fontsize=8, color=C["text"],
                bbox=dict(boxstyle="round,pad=0.35", fc=C["fig_bg"], ec=C["grid"], alpha=0.90))
    ax.set_title(f"Cα Distance Map — {dist_stats['folding_tendency']}", fontsize=12,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Residue", fontsize=12, labelpad=10)
    ax.set_ylabel("Residue", fontsize=12, labelpad=10)
    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.label.set_color(C["text"])
        cbar.ax.tick_params(colors=C["text"])
    plt.xticks(rotation=45, ha="right", fontsize=8, color=C["tick"])
    plt.yticks(rotation=0,  fontsize=8, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_contact_map_v2(cm_matrix: np.ndarray, cs: dict, seq: str = "", threshold: float = 8.0):
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

    # Highlight hub residues
    for hub in cs.get("hub_residues", [])[:3]:
        if hub < n:
            ax.axhline(hub, color=C["orange"], lw=1.0, alpha=0.6)
            ax.axvline(hub, color=C["orange"], lw=1.0, alpha=0.6)

    ax.set_xticks(range(0, n, tick_step))
    ax.set_xticklabels([show_labels[i] for i in range(0, n, tick_step)],
                       rotation=45, ha="right", fontsize=8, color=C["tick"])
    ax.set_yticks(range(0, n, tick_step))
    ax.set_yticklabels([show_labels[i] for i in range(0, n, tick_step)],
                       fontsize=8, color=C["tick"])
    ax.annotate(f"Contacts: {cs['total_contacts']}\nDensity: {cs['contact_density']:.1f}%",
                xy=(0.01, 0.99), xycoords="axes fraction", va="top", fontsize=8, color=C["text"],
                bbox=dict(boxstyle="round,pad=0.35", fc=C["fig_bg"], ec=C["grid"], alpha=0.90))
    ax.set_title(f"Contact Map ({threshold} Å) — Density: {cs['contact_density']:.1f}%",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Residue", fontsize=12, labelpad=10)
    ax.set_ylabel("Residue", fontsize=12, labelpad=10)
    plt.tight_layout()
    return fig


def plot_dssp_pie_v2(dssp: dict):
    C   = get_plot_colors()
    labels  = ["α-Helix", "β-Sheet", "Coil/Other"]
    sizes   = [dssp.get("helix", 0), dssp.get("sheet", 0), dssp.get("coil", 100)]
    colors  = ["#4CAF50", "#2196F3", "#9E9E9E"]
    # explode dominant slice
    max_idx = sizes.index(max(sizes))
    explode = [0.08 if i == max_idx else 0 for i in range(3)]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    apply_plot_style(fig, [ax])
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, textprops={"color": C["text"], "fontsize": 12},
        explode=explode,
    )
    for at in autotexts:
        at.set_color(C["text"])
        at.set_fontsize(11)
        at.set_fontweight("bold")
    dom = labels[max_idx]
    ent = dssp.get("ss_entropy", 0)
    ax.set_title(f"Secondary Structure — {dom} Dominant\nEntropy: {ent:.3f} bits",
                 fontsize=12, fontweight="bold", pad=14, color=C["text"])
    plt.tight_layout()
    return fig


def plot_sasa_per_residue_v2(pr_sasa: dict, sasa_stats: dict, seq: str = ""):
    C = get_plot_colors()
    if not pr_sasa:
        return None
    keys = list(pr_sasa.keys())
    vals = list(pr_sasa.values())
    median_val = np.median(vals)
    mean_val   = np.mean(vals)
    std_val    = np.std(vals)
    high_thresh = mean_val + std_val
    low_thresh  = mean_val - std_val
    bar_colors = []
    for v in vals:
        if v >= high_thresh:
            bar_colors.append("#1a8fd1")  # highly exposed
        elif v <= low_thresh:
            bar_colors.append("#e67e22")  # buried
        else:
            bar_colors.append("#9e9e9e")  # moderate
    fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.35), 5))
    apply_plot_style(fig, [ax])
    bars = ax.bar(range(len(keys)), vals, color=bar_colors, width=0.85)
    ax.axhline(median_val, color=C["red"],    linestyle="--", lw=1.5, label=f"Median = {median_val:.1f} Å²")
    ax.axhline(mean_val,   color=C["orange"], linestyle="-.", lw=1.5, label=f"Mean = {mean_val:.1f} Å²")
    ax.axhline(high_thresh, color="#1a8fd1",  linestyle=":",  lw=1.2, alpha=0.7, label=f"Exposed threshold ({high_thresh:.1f})")
    ax.axhline(low_thresh,  color="#e67e22",  linestyle=":",  lw=1.2, alpha=0.7, label=f"Buried threshold ({low_thresh:.1f})")
    # Annotate top 3 exposed
    top3_idx = sorted(range(len(vals)), key=lambda i: -vals[i])[:3]
    for idx in top3_idx:
        ax.annotate(keys[idx], xy=(idx, vals[idx]), ha="center",
                    fontsize=7, color=C["text"], xytext=(0, 4), textcoords="offset points")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8, color=C["tick"])
    ax.set_xlabel("Residue", fontsize=11, labelpad=8)
    ax.set_ylabel("SASA (Å²)", fontsize=11, labelpad=8)
    cls = sasa_stats.get("classification", "")
    ax.set_title(f"Per-Residue SASA — {cls}", fontsize=13, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_hydrophobicity_v2(seq: str, hyd_stats: dict):
    C    = get_plot_colors()
    vals = [KD_SCALE.get(a, 0) for a in seq]
    win  = min(5, len(seq))
    if len(seq) >= win:
        smooth = [np.mean(vals[max(0, i - win//2):i + win//2 + 1]) for i in range(len(seq))]
    else:
        smooth = vals
    fig, ax = plt.subplots(figsize=(max(8, len(seq) * 0.25), 5))
    apply_plot_style(fig, [ax])
    bar_colors = [C["accent1"] if v > 0 else C["red"] for v in vals]
    ax.bar(range(len(seq)), vals, color=bar_colors, alpha=0.6, width=0.85)
    ax.plot(range(len(seq)), smooth, color=C["orange"], lw=2.5, label=f"Sliding avg (w={win})")
    # Highlight hydrophobic clusters
    for start, end, _ in hyd_stats.get("hydrophobic_clusters", []):
        ax.axvspan(start - 0.5, min(end + 0.5, len(seq) - 0.5),
                   color="#1a8fd1", alpha=0.18)
        ax.annotate("HC", xy=((start+end)/2, max(vals)*0.9),
                    ha="center", fontsize=7, color=C["accent1"], fontweight="bold")
    ax.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels(list(seq), fontsize=9, color=C["tick"])
    ax.set_xlabel("Residue", fontsize=11, labelpad=8)
    ax.set_ylabel("Hydrophobicity (KD)", fontsize=11, labelpad=8)
    gv = hyd_stats.get("gravy", 0)
    mem = hyd_stats.get("membrane_affinity", "")
    ax.set_title(f"Hydrophobicity (GRAVY={gv:+.2f}, Membrane affinity: {mem})",
                 fontsize=12, fontweight="bold", pad=12)
    leg = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_charge_distribution_v2(seq: str, charge_stats: dict):
    C    = get_plot_colors()
    charges = [1 if a in "KRH" else (-1 if a in "DE" else 0) for a in seq]
    cumulative = np.cumsum(charges)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(seq) * 0.25), 7), sharex=True)
    apply_plot_style(fig, [ax1, ax2])
    bar_colors = [C["green"] if c > 0 else C["red"] if c < 0 else C["grid"] for c in charges]
    ax1.bar(range(len(seq)), charges, color=bar_colors, width=0.85)
    # Highlight charge clusters
    for ctype, start, end, subseq in charge_stats.get("clusters", []):
        col = C["green"] if ctype == "positive" else C["red"]
        ax1.axvspan(start-0.5, end+0.5, color=col, alpha=0.15)
        ax1.annotate(subseq, xy=((start+end)/2, 0.5 if ctype=="positive" else -0.8),
                     ha="center", fontsize=7, color=col, fontweight="bold")
    ax1.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax1.set_ylabel("Charge", fontsize=11)
    cls = charge_stats.get("classification", "")
    ax1.set_title(f"Charge Distribution — {cls} (net={charge_stats.get('net',0):+d})",
                  fontsize=12, fontweight="bold", pad=12)
    ax2.plot(range(len(seq)), cumulative, color=C["accent1"], lw=2.5)
    ax2.fill_between(range(len(seq)), cumulative, 0,
                     where=[c > 0 for c in cumulative], color=C["green"], alpha=0.15)
    ax2.fill_between(range(len(seq)), cumulative, 0,
                     where=[c <= 0 for c in cumulative], color=C["red"], alpha=0.15)
    ax2.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax2.set_ylabel("Cumulative Charge", fontsize=11)
    ax2.set_xlabel("Residue", fontsize=11)
    ax2.set_xticks(range(len(seq)))
    ax2.set_xticklabels(list(seq), fontsize=9, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_aa_composition_v2(seq: str, aa_stats: dict):
    C   = get_plot_colors()
    cnt = Counter(seq)
    L   = len(seq)
    sorted_aa  = sorted(AA)
    freqs      = [cnt.get(a, 0) / L * 100 for a in sorted_aa]
    bar_colors = [AA_COLORS.get(a, C["accent1"]) for a in sorted_aa]
    mean_freq  = 100.0 / len(AA)  # uniform baseline
    fig, ax    = plt.subplots(figsize=(12, 5))
    apply_plot_style(fig, [ax])
    bars = ax.bar(sorted_aa, freqs, color=bar_colors, edgecolor=C["grid"], width=0.75)
    ax.axhline(mean_freq, color=C["red"], linestyle="--", lw=1.5, alpha=0.7,
               label=f"Uniform baseline ({mean_freq:.1f}%)")
    # Annotate dominant residue
    dom_aa = aa_stats.get("dominant_aa", "")
    if dom_aa in sorted_aa:
        dom_idx = sorted_aa.index(dom_aa)
        ax.annotate(f"★ {dom_aa}\n{freqs[dom_idx]:.1f}%",
                    xy=(dom_idx, freqs[dom_idx]), ha="center",
                    fontsize=8, color=C["text"], fontweight="bold",
                    xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("Amino Acid", fontsize=11, labelpad=8)
    ax.set_ylabel("Frequency (%)", fontsize=11, labelpad=8)
    dom_group = aa_stats.get("dominant_group", "")
    ent = aa_stats.get("comp_entropy", 0)
    ax.set_title(f"AA Composition — Dominant group: {dom_group}, Entropy: {ent:.3f} bits",
                 fontsize=12, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=10, colors=C["tick"])
    leg = ax.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg.get_texts():
        t.set_color(C["text"])
    plt.tight_layout()
    return fig


def plot_sequence_logo_style_v2(seq: str, hyd_stats: dict, motifs: dict):
    C   = get_plot_colors()
    fig, ax = plt.subplots(figsize=(max(8, len(seq) * 0.4), 3.5))
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
    # Annotate hydrophobic cluster regions
    for start, end, subseq in hyd_stats.get("hydrophobic_clusters", []):
        ax.annotate("", xy=(end+0.4, 4.6), xytext=(start-0.4, 4.6),
                    arrowprops=dict(arrowstyle="<->", color=C["accent1"], lw=1.5))
        ax.annotate(f"HC:{subseq}", xy=((start+end)/2, 4.8), ha="center",
                    fontsize=7, color=C["accent1"])
    # Annotate aromatic clusters
    for start, end, subseq in motifs.get("aromatic_clusters", []):
        ax.annotate("⬟", xy=((start+end)/2, -4.6), ha="center",
                    fontsize=10, color="#b3de69")
    ax.axhline(0, color=C["grid"], lw=1, linestyle="--")
    ax.set_xlim(-0.5, len(seq) - 0.5)
    ax.set_xlabel("Position", fontsize=11, labelpad=8)
    ax.set_ylabel("|Hydrophobicity|", fontsize=11, labelpad=8)
    n_clusters = len(hyd_stats.get("hydrophobic_clusters", []))
    ax.set_title(f"Sequence Logo Visualization — {n_clusters} hydrophobic cluster(s)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(range(len(seq)))
    ax.set_xticklabels([f"{a}{i+1}" for i, a in enumerate(seq)],
                       rotation=45, ha="right", fontsize=8, color=C["tick"])
    plt.tight_layout()
    return fig


def plot_quality_dashboard_v2(
    q_score: float, q_label: str, breakdown: dict,
    folding_prop: dict, stability: dict, flexibility: dict,
    agg_risk: dict, complexity: dict, sol_prop: dict,
):
    C   = get_plot_colors()
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.5, 1.5], wspace=0.35)
    apply_plot_style(fig, [])
    fig.patch.set_facecolor(C["fig_bg"])

    color_map = {"Excellent": "#12b886", "Good": "#1a8fd1", "Moderate": "#f39c12", "Poor": "#c0392b"}
    col = color_map.get(q_label, "#9e9e9e")

    # Main gauge
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(C["ax_bg"])
    ax1.pie([q_score, 100 - q_score], colors=[col, C["grid"]],
            startangle=90, counterclock=False, wedgeprops=dict(width=0.4))
    ax1.text(0,  0.1, f"{q_score:.0f}", ha="center", va="center",
             fontsize=34, fontweight="bold", color=col)
    ax1.text(0, -0.25, q_label, ha="center", va="center",
             fontsize=12, fontweight="bold", color=C["text"])
    ax1.set_title("Overall Quality", fontsize=12, fontweight="bold",
                  color=C["text"], pad=8)

    # Score breakdown bars
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(C["ax_bg"])
    cats = list(breakdown.keys())
    vals = list(breakdown.values())
    max_per_cat = {
        "Ramachandran": 20, "Secondary Structure": 15, "SASA Distribution": 10,
        "Compactness (Rg)": 10, "Contact Density": 10, "H-Bond Network": 10,
        "Disulfide Bonds": 5, "pLDDT Confidence": 10, "Hydrophobic Balance": 5,
        "Folding Propensity": 5,
    }
    max_vals_cats = [max_per_cat.get(c, 10) for c in cats]
    pcts     = [v / m * 100 for v, m in zip(vals, max_vals_cats)]
    bcolors  = [("#12b886" if p >= 80 else "#1a8fd1" if p >= 60 else
                 "#f39c12" if p >= 40 else "#c0392b") for p in pcts]
    ax2.barh(cats, vals, color=bcolors, edgecolor=C["grid"], height=0.55)
    ax2.barh(cats, max_vals_cats, color=C["grid"], height=0.55, alpha=0.20)
    for i, (v, mv) in enumerate(zip(vals, max_vals_cats)):
        ax2.text(v + 0.1, i, f"{v:.1f}/{mv}", va="center", fontsize=8, color=C["text"])
    ax2.set_xlabel("Score", fontsize=10, color=C["text"])
    ax2.set_title("Quality Breakdown", fontsize=11, fontweight="bold", color=C["text"], pad=8)
    ax2.tick_params(axis="y", labelsize=7, colors=C["tick"])
    ax2.tick_params(axis="x", labelsize=8, colors=C["tick"])
    for sp in ax2.spines.values(): sp.set_edgecolor(C["grid"])

    # Composite scores radar-style bar chart
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(C["ax_bg"])
    composite_labels = ["Folding", "Stability", "Flexibility", "Agg. Risk", "Complexity", "Solubility"]
    composite_vals   = [
        folding_prop.get("score", 0), stability.get("score", 0),
        flexibility.get("score", 0),  agg_risk.get("score", 0),
        complexity.get("score", 0),   sol_prop.get("score", 0),
    ]
    composite_colors = ["#1a8fd1", "#12b886", "#f39c12", "#c0392b", "#9b59b6", "#2ecc71"]
    bars3 = ax3.barh(composite_labels, composite_vals,
                     color=composite_colors, edgecolor=C["grid"], height=0.55)
    ax3.barh(composite_labels, [100]*len(composite_labels),
             color=C["grid"], height=0.55, alpha=0.15)
    for i, v in enumerate(composite_vals):
        ax3.text(v + 0.8, i, f"{v:.0f}", va="center", fontsize=9,
                 color=C["text"], fontweight="bold")
    ax3.set_xlim(0, 105)
    ax3.set_xlabel("Score (0–100)", fontsize=10, color=C["text"])
    ax3.set_title("Composite Functional Scores", fontsize=11, fontweight="bold",
                  color=C["text"], pad=8)
    ax3.tick_params(axis="y", labelsize=9, colors=C["tick"])
    ax3.tick_params(axis="x", labelsize=8, colors=C["tick"])
    for sp in ax3.spines.values(): sp.set_edgecolor(C["grid"])

    plt.suptitle("Structural Quality Dashboard", fontsize=14, fontweight="bold",
                 color=C["text"], y=1.01)
    plt.tight_layout()
    return fig


def plot_advanced_scores_summary(
    folding_prop: dict, stability: dict, flexibility: dict,
    agg_risk: dict, complexity: dict, sol_prop: dict, seq: str = "",
):
    """Spider/radar-like breakdown for all composite scores."""
    C = get_plot_colors()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    apply_plot_style(fig, axes.flatten())
    score_dicts = [folding_prop, stability, flexibility, agg_risk, complexity, sol_prop]
    titles      = ["Folding Propensity", "Structural Stability",
                   "Flexibility", "Aggregation Risk",
                   "Structural Complexity", "Solubility Propensity"]
    colors_list = ["#1a8fd1", "#12b886", "#f39c12", "#c0392b", "#9b59b6", "#2ecc71"]
    for ax, sd, title, col in zip(axes.flatten(), score_dicts, titles, colors_list):
        bd = sd.get("breakdown", {})
        if not bd:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=C["text"])
            ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
            continue
        cats = list(bd.keys())
        vals = list(bd.values())
        max_v = max(vals) if vals else 1
        bar_cs = [col] * len(vals)
        ax.barh(cats, vals, color=bar_cs, edgecolor=C["grid"], height=0.5, alpha=0.85)
        for i, v in enumerate(vals):
            ax.text(v + 0.2, i, f"{v:.1f}", va="center", fontsize=8, color=C["text"])
        score = sd.get("score", 0)
        label = sd.get("label", "")
        ax.set_title(f"{title}\n{score:.0f}/100 — {label}", fontsize=10,
                     fontweight="bold", pad=6, color=C["text"])
        ax.tick_params(axis="y", labelsize=8, colors=C["tick"])
        ax.tick_params(axis="x", labelsize=8, colors=C["tick"])
        for sp in ax.spines.values(): sp.set_edgecolor(C["grid"])
    plt.suptitle(f"Advanced Structural Scores{f' — {seq[:15]}…' if len(seq)>15 else (f' — {seq}' if seq else '')}",
                 fontsize=13, fontweight="bold", color=C["text"], y=1.01)
    plt.tight_layout()
    return fig


# Keep original plot functions for analytics dashboard (unchanged)

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


# ==========================================================
# SECTION 15 - SEQUENCE ANALYSIS PANEL (upgraded)
# ==========================================================

def render_sequence_analysis(seq: str, prefix: str = ""):
    if not seq:
        return
    hyd_stats = compute_hydrophobicity_stats(seq)
    charge_stats = compute_charge_stats(seq)
    aa_stats = compute_aa_composition_stats(seq)
    motifs = detect_sequence_motifs(seq)

    st.markdown("### 🔠 Amino Acid Composition")
    fig_aa = plot_aa_composition_v2(seq, aa_stats)
    save_fig(fig_aa, f"{prefix}aa_composition.png")
    st.pyplot(fig_aa)
    plt.close(fig_aa)
    show_caption(caption_aa_composition_v2(aa_stats, seq))

    st.markdown("### 💧 Per-Residue Hydrophobicity")
    fig_hyd = plot_hydrophobicity_v2(seq, hyd_stats)
    save_fig(fig_hyd, f"{prefix}hydrophobicity.png")
    st.pyplot(fig_hyd)
    plt.close(fig_hyd)
    show_caption(caption_hydrophobicity_v2(hyd_stats, seq))

    st.markdown("### ⚡ Charge Distribution")
    fig_chg = plot_charge_distribution_v2(seq, charge_stats)
    save_fig(fig_chg, f"{prefix}charge_distribution.png")
    st.pyplot(fig_chg)
    plt.close(fig_chg)
    show_caption(caption_charge_v2(charge_stats, seq))

    st.markdown("### 🔤 Sequence Logo-Style Visualization")
    fig_logo = plot_sequence_logo_style_v2(seq, hyd_stats, motifs)
    save_fig(fig_logo, f"{prefix}sequence_logo.png")
    st.pyplot(fig_logo)
    plt.close(fig_logo)
    show_caption(caption_sequence_logo_v2(hyd_stats, motifs, seq))


# ==========================================================
# SECTION 16 - FULL STRUCTURAL ANALYSIS PANEL (v2 — fully dynamic)
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

    # ── Pre-compute all stats ─────────────────────────────
    phi_psi      = ramachandran(pdb_text)
    dssp         = run_dssp_fallback(pdb_text)
    n_residues   = len({l[22:26].strip() for l in pdb_text.splitlines() if l.startswith("ATOM")})
    rama_stats   = compute_rama_stats(phi_psi)
    rg_val       = radius_of_gyration(pdb_text)
    sasa_val     = compute_sasa(pdb_text)
    hbonds       = count_hbonds(pdb_text)
    ss_bonds     = count_disulfide_bonds(pdb_text)
    pr_sasa      = per_residue_sasa(pdb_text)
    dist_matrix  = ca_distance_map(pdb_text)
    cm_matrix    = contact_map(pdb_text, threshold=8.0)
    dist_stats   = compute_distance_stats(dist_matrix)
    contact_stats_d = compute_contact_stats(cm_matrix, threshold=8.0)
    sasa_stats   = compute_sasa_stats(pr_sasa, sasa_val, n_residues)
    rg_stats     = compute_rg_stats(rg_val, n_residues)
    hbond_stats  = compute_hbond_stats(hbonds, n_residues)
    ss_stats     = compute_disulfide_stats(ss_bonds, seq)
    plddt_stats  = compute_plddt_stats(plddt_vals if plddt_vals else [])
    hyd_stats    = compute_hydrophobicity_stats(seq) if seq else compute_hydrophobicity_stats("")
    charge_stats = compute_charge_stats(seq) if seq else compute_charge_stats("")
    aa_stats     = compute_aa_composition_stats(seq) if seq else {"dominant_aa": "N/A", "dominant_count": 0, "dominant_pct": 0, "aromatic_richness": 0, "polar_richness": 0, "hydrophobic_richness": 0, "comp_entropy": 0, "dominant_group": "N/A", "bio_note": "", "group_counts": {}}
    motifs       = detect_sequence_motifs(seq) if seq else {"charge_clusters": [], "aromatic_clusters": [], "repeats": []}

    # ── Composite scores ─────────────────────────────────
    folding_prop = compute_folding_propensity(contact_stats_d, dssp, hyd_stats, rg_stats)
    sol_prop     = compute_solubility_propensity(sasa_stats, charge_stats, hyd_stats)
    agg_risk     = compute_aggregation_risk(hyd_stats, sasa_stats, aa_stats)
    stability    = compute_stability_score(hbond_stats, ss_stats, rg_stats, dssp)
    flexibility  = compute_flexibility_score(plddt_stats, dssp, rama_stats)
    complexity   = compute_complexity_score(contact_stats_d, dssp, dist_stats)

    # ── Overall quality score ─────────────────────────────
    q_score, q_label, q_breakdown = structural_quality_score_v2(
        rama_stats, dssp, sasa_stats, rg_stats, contact_stats_d,
        hbond_stats, ss_stats, plddt_stats, hyd_stats,
        folding_prop, stability, n_residues,
    )

    # ── DSSP ──────────────────────────────────────────────
    st.markdown("### 🔩 Secondary Structure")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("α-Helix",    f"{dssp['helix']}%")
    c2.metric("β-Sheet",    f"{dssp['sheet']}%")
    c3.metric("Coil/Other", f"{dssp['coil']}%")
    c4.metric("H:S Ratio",  f"{dssp.get('hs_ratio', 0):.2f}")
    c5.metric("SS Entropy", f"{dssp.get('ss_entropy', 0):.3f}")
    fig_pie = plot_dssp_pie_v2(dssp)
    save_fig(fig_pie, f"{prefix}dssp_pie.png")
    st.pyplot(fig_pie)
    plt.close(fig_pie)
    show_caption(caption_dssp_v2(dssp))

    # ── Structural Metrics ────────────────────────────────
    st.markdown("### 🔬 Structural Metrics")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Radius of Gyration",   f"{rg_val:.2f} Å" if rg_val else "N/A")
    m2.metric("Total SASA",           f"{sasa_val:.1f} Å²" if sasa_val else "N/A")
    m3.metric("H-Bonds (est.)",       str(hbonds))
    m4.metric("Disulfide Bonds",      str(ss_bonds))
    m5.metric("H-Bonds/Residue",      f"{hbond_stats['per_residue']:.2f}")
    m6.metric("Rg Classification",    rg_stats.get("classification", "N/A")[:12])

    # ── Structural Quality Dashboard ──────────────────────
    st.markdown("### 🏅 Structural Quality Dashboard")
    fig_qual = plot_quality_dashboard_v2(
        q_score, q_label, q_breakdown,
        folding_prop, stability, flexibility,
        agg_risk, complexity, sol_prop,
    )
    save_fig(fig_qual, f"{prefix}quality_dashboard.png")
    st.pyplot(fig_qual)
    plt.close(fig_qual)
    qual_color = {"Excellent": "#12b886", "Good": "#1a8fd1",
                  "Moderate": "#f39c12", "Poor": "#c0392b"}.get(q_label, "#9e9e9e")
    breakdown_str = " | ".join(f"{k}: {v:.1f}" for k, v in q_breakdown.items())
    show_caption(
        f"Overall structural quality: "
        f"<strong style='color:{qual_color};'>{q_score}/100 — {q_label}</strong><br>"
        f"<em>{breakdown_str}</em>"
    )

    # ── Advanced Composite Scores ─────────────────────────
    st.markdown("### 🧪 Advanced Composite Scores")
    ac1, ac2, ac3, ac4, ac5, ac6 = st.columns(6)
    ac1.metric("Folding Propensity",  f"{folding_prop['score']:.0f}/100",  folding_prop["label"])
    ac2.metric("Stability",           f"{stability['score']:.0f}/100",     stability["label"])
    ac3.metric("Flexibility",         f"{flexibility['score']:.0f}/100",   flexibility["label"])
    ac4.metric("Aggregation Risk",    f"{agg_risk['score']:.0f}/100",      agg_risk["label"])
    ac5.metric("Complexity",          f"{complexity['score']:.0f}/100",    complexity["label"])
    ac6.metric("Solubility",          f"{sol_prop['score']:.0f}/100",      sol_prop["label"])
    fig_adv = plot_advanced_scores_summary(
        folding_prop, stability, flexibility, agg_risk, complexity, sol_prop, seq=seq)
    save_fig(fig_adv, f"{prefix}advanced_scores.png")
    st.pyplot(fig_adv)
    plt.close(fig_adv)

    # ── pLDDT ─────────────────────────────────────────────
    if plddt_vals and len(plddt_vals) > 0 and max(plddt_vals) > 1.0:
        st.markdown("### 📊 pLDDT Confidence Profile")
        pc1, pc2, pc3, pc4, pc5 = st.columns(5)
        pc1.metric("Mean pLDDT",    f"{plddt_stats['mean']:.1f}")
        pc2.metric("Median pLDDT", f"{plddt_stats['median']:.1f}")
        pc3.metric("Min pLDDT",    f"{plddt_stats['min']:.1f}")
        pc4.metric("Max pLDDT",    f"{plddt_stats['max']:.1f}")
        pc5.metric("σ",            f"{plddt_stats['std']:.1f}")
        fig_plddt = plot_plddt_v2(plddt_vals, plddt_stats, seq=seq)
        save_fig(fig_plddt, f"{prefix}plddt.png")
        st.pyplot(fig_plddt)
        plt.close(fig_plddt)
        show_caption(caption_plddt_v2(plddt_stats, seq=seq))

    # ── Ramachandran ──────────────────────────────────────
    st.markdown("### 📐 Ramachandran Plot")
    if not phi_psi:
        st.info("No φ/ψ angles — peptide needs ≥3 residues.")
    else:
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Allowed %",     f"{rama_stats['allowed_pct']}%")
        rc2.metric("Outliers",      f"{rama_stats['outlier_count']}")
        rc3.metric("Backbone Strain", f"{rama_stats['backbone_strain']:.1f}")
        rc4.metric("Flexibility",   f"{rama_stats['conformational_flexibility']:.1f}°")
    fig_rama = plot_ramachandran_v2(phi_psi, rama_stats, seq=seq)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.pyplot(fig_rama)
    plt.close(fig_rama)
    show_caption(caption_ramachandran_v2(rama_stats, seq=seq))

    # ── Distance Map ──────────────────────────────────────
    st.markdown("### 🗺️ Cα Distance Map")
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Mean Distance",    f"{dist_stats['mean_dist']:.1f} Å")
    dc2.metric("Compactness",      f"{dist_stats['compactness_score']:.1f}/100")
    dc3.metric("Folding Tendency", dist_stats["folding_tendency"][:20])
    try:
        fig_dist = plot_distance_map_v2(dist_matrix, dist_stats, seq=seq)
        save_fig(fig_dist, f"{prefix}ca_distance_map.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        show_caption(caption_distance_map_v2(dist_stats, seq=seq))
    except Exception as e:
        st.warning(f"Distance map error: {e}")

    # ── Contact Map ───────────────────────────────────────
    st.markdown("### 🔗 Contact Map (8 Å threshold)")
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Total Contacts",     contact_stats_d["total_contacts"])
    cc2.metric("Contact Density",    f"{contact_stats_d['contact_density']:.1f}%")
    cc3.metric("Long-Range Contacts",contact_stats_d["long_range_contacts"])
    try:
        fig_cm = plot_contact_map_v2(cm_matrix, contact_stats_d, seq=seq, threshold=8.0)
        save_fig(fig_cm, f"{prefix}contact_map.png")
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        show_caption(caption_contact_map_v2(contact_stats_d, threshold=8.0))
    except Exception as e:
        st.warning(f"Contact map error: {e}")

    # ── Per-Residue SASA ──────────────────────────────────
    st.markdown("### 💧 Per-Residue SASA")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total SASA",    f"{sasa_stats['total_sasa']:.1f} Å²")
    sc2.metric("Avg/Residue",   f"{sasa_stats['avg_sasa']:.1f} Å²")
    sc3.metric("Exposed %",     f"{sasa_stats['exposed_pct']:.1f}%")
    sc4.metric("Classification", sasa_stats["classification"][:18])
    if pr_sasa:
        fig_sasa = plot_sasa_per_residue_v2(pr_sasa, sasa_stats, seq=seq)
        if fig_sasa:
            save_fig(fig_sasa, f"{prefix}sasa_per_residue.png")
            st.pyplot(fig_sasa)
            plt.close(fig_sasa)
            show_caption(caption_sasa_v2(sasa_stats, pr_sasa))
    else:
        st.info("Per-residue SASA could not be computed.")

    # ── Radius of Gyration ────────────────────────────────
    st.markdown("### 📏 Radius of Gyration")
    if rg_stats.get("rg"):
        rg1, rg2, rg3, rg4 = st.columns(4)
        rg1.metric("Rg",                   f"{rg_stats['rg']:.2f} Å")
        rg2.metric("Expected (coil)",       f"{rg_stats['expected_rg_coil']:.2f} Å")
        rg3.metric("Expected (globular)",   f"{rg_stats['expected_rg_glob']:.2f} Å")
        rg4.metric("Compactness Score",     f"{rg_stats['compactness_score']:.1f}/100")
        show_caption(caption_rg_v2(rg_stats, seq=seq))
    else:
        st.info("Radius of gyration could not be computed.")

    # ── H-Bond & Disulfide ────────────────────────────────
    st.markdown("### 🔗 Hydrogen Bonds & Disulfide Bonds")
    hb1, hb2, hb3, hb4 = st.columns(4)
    hb1.metric("Total H-Bonds",       hbond_stats["count"])
    hb2.metric("H-Bonds/Residue",     f"{hbond_stats['per_residue']:.2f}")
    hb3.metric("Stabilization Index", f"{hbond_stats['stabilization_index']:.1f}/100")
    hb4.metric("Disulfide Bonds",     ss_stats["count"])
    show_caption(caption_hbond_v2(hbond_stats) + "<br><br>" + caption_disulfide_v2(ss_stats))

    # ── Residue Exposure ──────────────────────────────────
    st.markdown("### 🔍 Residue Exposure Classification")
    exposure = residue_exposure(pdb_text)
    if exposure:
        exp_df = pd.DataFrame([(k, v) for k, v in exposure.items()], columns=["Residue", "Exposure"])
        n_exp = (exp_df["Exposure"] == "Exposed").sum()
        n_bur = (exp_df["Exposure"] == "Buried").sum()
        ec1, ec2 = st.columns(2)
        ec1.metric("Exposed Residues", n_exp)
        ec2.metric("Buried Residues",  n_bur)
        st.dataframe(exp_df, use_container_width=True)
    else:
        st.info("Residue exposure could not be computed.")


# ==========================================================
# SECTION 17 - MODEL TRAINING
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
# SECTION 18 - LOAD MODELS
# ==========================================================

(
    df_all, X_all, X_test, yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics,
) = train_models()


# ==========================================================
# SECTION 19 - PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics: dict, prediction: dict, image_paths: list) -> str:
    file_name = "PepTastePredictor_Report.pdf"
    styles    = getSampleStyleSheet()
    doc       = SimpleDocTemplate(file_name, pagesize=A4,
                                  topMargin=40, bottomMargin=40,
                                  leftMargin=50, rightMargin=50)
    story = []
    story.append(Paragraph("<b>PepTastePredictor — Comprehensive Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking &amp; dynamic structural analysis. "
        "Fully data-driven structural scoring system.",
        styles["Normal"]))
    story.append(Spacer(1, 14))
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
    if prediction:
        story.append(Paragraph("<b>Prediction Results</b>", styles["Heading2"]))
        for k, v in prediction.items():
            story.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
        story.append(Spacer(1, 14))
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
        "advanced_scores.png":     "Advanced Composite Scores",
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
        f"<i>Generated by PepTastePredictor v4 · {date.today().strftime('%d %B %Y')}</i>",
        styles["Normal"],
    ))
    doc.build(story)
    return file_name


# ==========================================================
# SECTION 20 - HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🧬 PepTastePredictor</h1>
<p>
An integrated machine learning &amp; structural bioinformatics platform for peptide
taste, solubility, docking, and fully dynamic 3D structural analysis.<br>
Supports <strong>local PeptideBuilder</strong> and external
<strong>ESM Atlas</strong> / <strong>AlphaFold Server</strong> workflows.
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 21 - MODE SELECTION
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
# SECTION 22 - SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📝 Sequence", "🤖 ML Predictions", "🧬 Structure Viewer",
        "📐 Structural Analysis", "🔬 Advanced Bioinformatics",
        "📊 Analytics Dashboard", "📄 Report",
    ])

    # ── TAB 1 ─────────────────────────────────────────────
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
            d1, d2 = st.columns(2)
            d1.download_button("⬇️ Download as FASTA",
                               f">peptide\n{seq_clean}\n",
                               file_name="peptide.fasta", mime="text/plain")
            d2.download_button("⬇️ Download sequence (.txt)",
                               seq_clean, file_name="peptide_sequence.txt", mime="text/plain")

    # ── TAB 2 ─────────────────────────────────────────────
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

            st.markdown("### 🎯 Prediction Confidence")
            taste_proba  = taste_model.predict_proba(Xp)[0]
            taste_conf   = max(taste_proba) * 100
            sol_proba    = sol_model.predict_proba(Xp)[0]
            sol_conf     = max(sol_proba) * 100
            conf1, conf2 = st.columns(2)
            conf1.metric("Taste Confidence",      f"{taste_conf:.1f}%")
            conf2.metric("Solubility Confidence", f"{sol_conf:.1f}%")

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

    # ── TAB 3 ─────────────────────────────────────────────
    with tab3:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not seq_clean:
            st.info("Enter a sequence in the Sequence tab first.")
        else:
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
                    st.error("PeptideBuilder failed.")

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

    # ── TAB 4 ─────────────────────────────────────────────
    with tab4:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not st.session_state.pdb_text:
            st.info("Generate or upload a structure in the **Structure Viewer** tab first.")
        else:
            pdb_text   = st.session_state.pdb_text
            plddt_vals = _extract_plddt_from_pdb(pdb_text)
            has_plddt  = len(plddt_vals) > 0 and max(plddt_vals) > 1.0
            render_structural_analysis(
                pdb_text,
                prefix="single_",
                seq=seq_clean,
                plddt_vals=plddt_vals if has_plddt else None,
                source_label=st.session_state.pdb_source or "",
            )

    # ── TAB 5 ─────────────────────────────────────────────
    with tab5:
        seq_clean = clean_sequence(st.session_state.get("single_seq_input", ""))
        if not seq_clean:
            st.info("Enter a sequence in the Sequence tab first.")
        else:
            st.markdown("## 🔬 Advanced Bioinformatics")
            with st.expander("🧬 Sequence Properties Summary"):
                phys = physicochemical_features(seq_clean[:100])
                for k, v in phys.items():
                    st.write(f"**{k}**: {v}")

            # Dipeptide heatmap
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

            # Sliding-window hydrophobicity
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
                max_win = win_vals.index(max(win_vals))
                ax_win.annotate(f"Peak: pos {max_win+1}",
                                xy=(max_win, max(win_vals)), ha="center",
                                fontsize=9, color=C["text"],
                                xytext=(0, 8), textcoords="offset points")
                ax_win.set_title(
                    f"Sliding-Window Hydrophobicity (w={win_size}) — "
                    f"Peak at position {max_win+1}",
                    fontsize=12, fontweight="bold")
                leg_w = ax_win.legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
                for t in leg_w.get_texts(): t.set_color(C["text"])
                plt.tight_layout()
                save_fig(fig_win, "single_window_hydrophobicity.png")
                st.pyplot(fig_win)
                plt.close(fig_win)
                show_caption(
                    f"Hydrophobicity averaged over window of {win_size} residues.<br>"
                    f"Peak hydrophobic window at position <strong>{max_win+1}</strong> "
                    f"(score: {max(win_vals):.2f})."
                )
            else:
                st.info(f"Sequence must be ≥{win_size} residues for sliding-window analysis.")

            # Motif detection
            st.markdown("### 🔍 Sequence Motif & Cluster Detection")
            motifs = detect_sequence_motifs(seq_clean)
            if motifs["charge_clusters"]:
                st.write(f"**Charge clusters ({len(motifs['charge_clusters'])}):**")
                for ctype, start, end, subseq in motifs["charge_clusters"]:
                    st.write(f"  {'(+)' if ctype=='positive' else '(-)'} pos {start+1}–{end+1}: `{subseq}`")
            if motifs["aromatic_clusters"]:
                st.write(f"**Aromatic clusters ({len(motifs['aromatic_clusters'])}):**")
                for start, end, subseq in motifs["aromatic_clusters"]:
                    st.write(f"  pos {start+1}–{end+1}: `{subseq}`")
            if motifs["repeats"]:
                st.write(f"**Repeated motifs:**")
                for motif, count in motifs["repeats"]:
                    st.write(f"  `{motif}` × {count}")
            if not any([motifs["charge_clusters"], motifs["aromatic_clusters"], motifs["repeats"]]):
                st.info("No significant motifs or clusters detected.")

    # ── TAB 6 ─────────────────────────────────────────────
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

    # ── TAB 7 ─────────────────────────────────────────────
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
# SECTION 23 - BATCH PREDICTION MODE
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

                st.markdown("### 📊 Batch Summary")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Total Peptides", total)
                s2.metric("Unique Tastes",  batch_df["Predicted Taste"].nunique())
                sol_pct = 100 * batch_df["Predicted Solubility"].str.contains(
                    "oluble", case=False, na=False).mean()
                s3.metric("Soluble (%)", f"{sol_pct:.1f}%")
                valid_d = batch_df["Predicted Docking Score"].dropna()
                s4.metric("Avg Docking", f"{valid_d.mean():.2f} kcal/mol" if len(valid_d) else "N/A")

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
# SECTION 24 - PDB UPLOAD & STRUCTURAL ANALYSIS MODE
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

            with st.expander("📊 Dataset Analytics", expanded=False):
                fig_dist = plot_distributions(df_all)
                save_fig(fig_dist, "distributions.png")
                st.pyplot(fig_dist)
                plt.close(fig_dist)
                show_caption(caption_distributions(df_all))

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
# SECTION 25 - FOOTER
# ==========================================================

st.markdown(f"""
<div class="footer">
&copy; {date.today().year} &nbsp; <b>PepTastePredictor v4</b><br>
AI + Fully Dynamic Structural Bioinformatics platform for peptide analysis<br>
Local PeptideBuilder &nbsp;|&nbsp; External: ESM Atlas &amp; AlphaFold Server<br>
For academic, educational, and research use only
</div>
""", unsafe_allow_html=True)
