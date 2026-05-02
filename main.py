# ==========================================================
# PepTastePredictor — app.py
# A complete end-to-end peptide analysis platform
# Light + Dark Mode Compatible Version
# ==========================================================

# ==========================================================
# SECTION 1 - IMPORTS
# ==========================================================

import os
import tempfile
from datetime import date
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder
import PeptideBuilder
from PeptideBuilder import Geometry

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

DATASET_PATH = "AIML.xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}


# ==========================================================
# SECTION 3 - FRONTEND STYLING (Light + Dark Mode)
# ==========================================================

st.markdown(
    """
<style>

/* ══════════════════════════════════════════════════════════
   UNIVERSAL FIX — var(--text-color) is injected by
   Streamlit itself and is always correct for the current
   theme (dark or light). Using it everywhere means we never
   fight against Streamlit's own theme system.
══════════════════════════════════════════════════════════ */

/* Every text element in the app */
.stApp p, .stApp span, .stApp label,
.stApp li, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp div {
    color: var(--text-color) !important;
}
.stMarkdown, .stMarkdown * { color: var(--text-color) !important; }
h1, h2, h3, h4 { color: var(--text-color) !important; }

/* Radio */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label p { color: var(--text-color) !important; }

/* Text input label */
div[data-testid="stTextInput"] label,
div[data-testid="stTextInput"] label p { color: var(--text-color) !important; }

/* Text input VALUE — the typed text inside the box */
div[data-testid="stTextInput"] input,
div[data-testid="stTextInput"] input::placeholder { color: var(--text-color) !important; }

/* File uploader */
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploaderDropzone"] span,
div[data-testid="stFileUploaderDropzone"] p { color: var(--text-color) !important; }

/* Selectbox */
div[data-testid="stSelectbox"] label,
div[data-testid="stSelectbox"] label p { color: var(--text-color) !important; }

/* Expander */
details summary p, details summary span,
button[data-testid="stExpanderToggleButton"] p,
button[data-testid="stExpanderToggleButton"] span,
button[data-testid="stExpanderToggleButton"] div { color: var(--text-color) !important; }

/* Sidebar */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--text-color) !important; }

/* DataFrames */
.stDataFrame, .stDataFrame td, .stDataFrame th,
[data-testid="stDataFrame"] * { color: var(--text-color) !important; }

/* Metric widgets */
[data-testid="stMetric"] label,
[data-testid="stMetric"] div { color: var(--text-color) !important; }

/* Buttons — inherit so Streamlit's own colour applies correctly */
.stButton button p,
div[data-testid="stDownloadButton"] button p { color: inherit !important; }

/* Alert boxes — never override */
div[data-testid="stAlert"] *,
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span { color: inherit !important; }

/* ══════════════════════════════════════════════════════════
   CUSTOM HTML COMPONENTS
══════════════════════════════════════════════════════════ */

.hero {
    background: linear-gradient(90deg, #1f3c88, #0b7285);
    padding: 36px 40px;
    border-radius: 16px;
    margin-bottom: 36px;
}
.hero h1 {
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    margin-bottom: 10px;
    color: #ffffff !important;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.05rem !important;
    line-height: 1.7;
    color: #dce8ff !important;
    margin: 0;
}

.card {
    border: 1px solid rgba(128,128,180,0.3);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 28px;
    background: rgba(128,128,180,0.05);
}

.metric-label {
    font-size: 13px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.7;
    margin-bottom: 4px;
    margin-top: 18px;
    color: var(--text-color) !important;
}
.metric-label:first-child { margin-top: 0; }

.metric-value {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #1a8fd1 !important;
    margin-bottom: 2px;
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
.graph-caption strong {
    font-weight: 700;
    color: var(--text-color) !important;
}
.graph-caption em {
    font-style: italic;
    color: var(--text-color) !important;
    opacity: 0.85;
}

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

</style>
""",
    unsafe_allow_html=True,
)



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
st.sidebar.info("For academic & educational use only")


# ==========================================================
# SECTION 5 - SESSION STATE INITIALISATION
# ==========================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.pdb_text = None
    st.session_state.last_prediction = {}
    st.session_state.show_analytics = False
    st.session_state.pdf_figures = []


# ==========================================================
# SECTION 6 - UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename):
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    if filename not in st.session_state.pdf_figures:
        st.session_state.pdf_figures.append(filename)


def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


def model_features(seq):
    ana = ProteinAnalysis(seq)
    features = {
        "length":      len(seq),
        "mw":          ana.molecular_weight(),
        "pI":          ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability": ana.instability_index(),
        "gravy":       ana.gravy(),
        "charge":      ana.charge_at_pH(7.0),
    }
    for aa in AA:
        features[f"AA_{aa}"] = seq.count(aa) / len(seq)
    denom = max(len(seq) - 1, 1)
    for dp in ALL_DIPEPTIDES:
        features[f"DPC_{dp}"] = seq.count(dp) / denom
    L = len(seq)
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
    ana = ProteinAnalysis(seq)
    h, t, s = ana.secondary_structure_fraction()
    return {
        "Length":                len(seq),
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
    return sum(KD_SCALE.get(a, 0) for a in seq) / len(seq)


def show_caption(html_text: str):
    """Render a styled, theme-aware caption box below each graph."""
    st.markdown(
        f'<div class="graph-caption">{html_text}</div>',
        unsafe_allow_html=True,
    )


# ==========================================================
# SECTION 6B - DYNAMIC CAPTION HELPERS
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
        f"with a mean of <strong>{mean_len:.1f} aa</strong>. "
        f"Most peptides are short — very long sequences are rare in this dataset.<br><br>"

        f"<strong>Taste classes (centre):</strong> "
        f"&ldquo;{dominant_taste}&rdquo; is the most represented class "
        f"(<strong>{dominant_count}</strong> peptides out of {len(df)} total, "
        f"across {n_classes} classes). "
        f"Class imbalances like this are why the model uses balanced class weighting during training.<br><br>"

        f"<strong>GRAVY distribution (right):</strong> "
        f"The dataset's mean GRAVY score is <strong>{mean_gravy:.2f}</strong>, "
        f"indicating the average peptide is <strong>{gravy_label}</strong>. "
        f"Scores above 0 lean hydrophobic; below 0 lean hydrophilic."
    )


def caption_pca(pca_model, class_names):
    var1  = pca_model.explained_variance_ratio_[0] * 100
    var2  = pca_model.explained_variance_ratio_[1] * 100
    total = var1 + var2
    return (
        f"Each dot represents one peptide, compressed from hundreds of physicochemical features "
        f"down to just 2 dimensions for visualisation.<br><br>"

        f"<strong>PC1</strong> captures <strong>{var1:.1f}%</strong> of the total variance and "
        f"<strong>PC2</strong> captures <strong>{var2:.1f}%</strong> — "
        f"together accounting for <strong>{total:.1f}%</strong> of all variation in the dataset.<br><br>"

        f"<strong>Tight, well-separated colour clusters</strong> mean the model can reliably distinguish "
        f"those taste classes. "
        f"<strong>Overlapping clusters</strong> indicate classes that share similar amino acid profiles — "
        f"expect higher confusion between those classes in the confusion matrix below."
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
        f"The taste model achieved <strong>{acc:.1f}% overall accuracy</strong> "
        f"on the held-out test set.<br><br>"

        f"<strong>How to read this chart:</strong> Rows = actual taste class, "
        f"Columns = what the model predicted. "
        f"Numbers on the <em>diagonal</em> are correct predictions. "
        f"Off-diagonal numbers are mistakes.<br><br>"

        f"<strong>Most common confusion:</strong> Actual &ldquo;{true_cls}&rdquo; was predicted as "
        f"&ldquo;{pred_cls}&rdquo; <strong>{worst_n} time(s)</strong> — "
        f"likely because these two classes share similar physicochemical profiles.<br><br>"

        f"<strong>Best-classified class:</strong> &ldquo;{best_cls}&rdquo; "
        f"(highest per-class accuracy) &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"<strong>Hardest class:</strong> &ldquo;{worst_cls}&rdquo; (most off-diagonal errors)."
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
        f"The solubility model achieved <strong>{acc:.1f}% overall accuracy</strong> "
        f"on the held-out test set.<br><br>"

        f"<strong>Most common misclassification:</strong> "
        f"Actual &ldquo;{true_cls}&rdquo; was predicted as "
        f"&ldquo;{pred_cls}&rdquo; <strong>{worst_n} time(s)</strong>.<br><br>"

        f"These errors typically occur for peptides with borderline hydrophobicity scores "
        f"that fall close to the solubility decision boundary — "
        f"neither clearly soluble nor clearly insoluble."
    )


def caption_feature_importance(model, feature_names, top_n=20):
    imp = pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    top3        = [prettify_feature(f) for f in imp["Feature"].iloc[:3]]
    top3_scores = imp["Importance"].iloc[:3].tolist()
    n_dpc       = sum(1 for f in imp["Feature"] if f.startswith("DPC_"))
    n_aa        = sum(1 for f in imp["Feature"] if f.startswith("AA_"))
    dpc_note    = (
        "Dipeptide patterns dominate — meaning <em>sequential context</em> "
        "matters more than raw composition alone."
        if n_dpc > n_aa else
        "Single amino acid composition is the stronger predictor here."
    )
    return (
        f"The top {top_n} features driving the taste model's decisions, "
        f"ranked by importance score.<br><br>"

        f"<strong>#1 &mdash; {top3[0]}</strong> &nbsp; (score: {top3_scores[0]:.4f})<br>"
        f"<strong>#2 &mdash; {top3[1]}</strong> &nbsp; (score: {top3_scores[1]:.4f})<br>"
        f"<strong>#3 &mdash; {top3[2]}</strong> &nbsp; (score: {top3_scores[2]:.4f})<br><br>"

        f"Of the top {top_n}: <strong>{n_dpc}</strong> are dipeptide frequency features (DPC) and "
        f"<strong>{n_aa}</strong> are single amino acid frequency features. "
        f"{dpc_note}"
    )


def caption_docking(y_true, y_pred):
    r2        = r2_score(y_true, y_pred)
    rmse      = np.sqrt(mean_squared_error(y_true, y_pred))
    min_score = y_true.min()
    max_score = y_true.max()
    quality   = "strong" if r2 >= 0.75 else ("moderate" if r2 >= 0.5 else "weak")
    return (
        f"Each point is one peptide from the held-out test set. "
        f"The <strong>red dashed line</strong> is the ideal reference — "
        f"a perfect model would place all points exactly on this line.<br><br>"

        f"<strong>R² = {r2:.3f}</strong> — indicates a <strong>{quality} fit</strong>. "
        f"The model explains <strong>{r2*100:.1f}%</strong> of variance in docking scores.<br>"
        f"<strong>RMSE = {rmse:.2f} kcal/mol</strong> — this is the typical prediction error per peptide.<br>"
        f"True docking scores range from <strong>{min_score:.2f}</strong> to "
        f"<strong>{max_score:.2f} kcal/mol</strong> in this test set.<br><br>"

        f"Points <strong>above</strong> the diagonal = model over-estimated binding affinity. "
        f"Points <strong>below</strong> the diagonal = model under-estimated it."
    )


def caption_ramachandran(phi_psi, seq=""):
    if not phi_psi:
        return (
            "No φ/ψ angles could be extracted — the peptide may be too short "
            "(fewer than 3 residues) or composed entirely of proline."
        )
    n_total      = len(phi_psi)
    n_helix      = sum(1 for p, s in phi_psi if -180 <= p <= -45 and -75 <= s <= -15)
    n_sheet      = sum(1 for p, s in phi_psi if -180 <= p <= -45 and 90  <= s <= 180)
    n_disallowed = n_total - n_helix - n_sheet
    pct_helix    = n_helix      / n_total * 100
    pct_sheet    = n_sheet      / n_total * 100
    pct_dis      = n_disallowed / n_total * 100
    dominant     = "α-helix" if n_helix >= n_sheet else "β-sheet"
    length_note  = f" for this <strong>{len(seq)}-residue peptide</strong>" if seq else ""
    return (
        f"Each dot is one backbone torsion angle pair (φ, ψ){length_note}. "
        f"The <strong>coloured shaded regions</strong> mark conformations typical "
        f"of known secondary structures.<br><br>"

        f"<strong>α-helix region (green):</strong> &nbsp; ~{pct_helix:.0f}% of residues<br>"
        f"<strong>β-sheet region (blue):</strong> &nbsp; ~{pct_sheet:.0f}% of residues<br>"
        f"<strong>Outside allowed regions:</strong> &nbsp; ~{pct_dis:.0f}% of residues "
        f"— strained conformations, common in short or proline-rich peptides.<br><br>"

        f"This peptide's backbone geometry primarily favours "
        f"<strong>{dominant}</strong> character."
    )


def caption_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if n < 2:
        return "Distance map could not be computed — fewer than 2 Cα atoms detected."
    mask       = ~np.eye(n, dtype=bool)
    off_diag   = dist_matrix[mask]
    max_dist   = off_diag.max()
    min_dist   = off_diag.min()
    long_range = sum(
        1 for i in range(n) for j in range(n)
        if abs(i - j) > 3 and dist_matrix[i, j] < 8.0
    )
    fold_note  = (
        "suggesting the peptide <strong>folds back on itself</strong>"
        if long_range > 0 else
        "consistent with an <strong>extended / linear conformation</strong>"
    )
    seq_label  = (
        f"for <strong>{seq}</strong> ({n} residues)"
        if seq else
        f"for this <strong>{n}-residue peptide</strong>"
    )
    return (
        f"Pairwise Cα–Cα distances {seq_label} — "
        f"<strong>darker = closer in 3D space</strong>.<br><br>"

        f"<strong>Nearest non-adjacent residues:</strong> &nbsp; {min_dist:.1f} Å apart<br>"
        f"<strong>Furthest residue pair:</strong> &nbsp; {max_dist:.1f} Å apart<br>"
        f"<strong>Long-range contacts</strong> (|i−j| &gt; 3, distance &lt; 8 Å): "
        f"<strong>{long_range} pair(s)</strong> — {fold_note}.<br><br>"

        f"The bright diagonal band is expected — adjacent residues are always ~3.8 Å apart."
    )


# ==========================================================
# SECTION 6C - MATPLOTLIB THEME
# Detects Streamlit theme and adjusts plot colours
# ==========================================================

def _is_dark_mode():
    """Best-effort dark mode detection via Streamlit's theme config."""
    try:
        theme = st.get_option("theme.base")
        return theme == "dark"
    except Exception:
        return False


def get_plot_colors():
    dark = _is_dark_mode()
    if dark:
        return {
            "fig_bg":  "#1a1d2e",
            "ax_bg":   "#1e2140",
            "text":    "#e8edf8",
            "grid":    "#2e3560",
            "accent1": "#5c7cfa",
            "accent2": "#748ffc",
            "accent3": "#4dd0e1",
            "red":     "#ff6b6b",
            "orange":  "#ffa94d",
            "tick":    "#c5cff0",
        }
    else:
        return {
            "fig_bg":  "#f8f9fc",
            "ax_bg":   "#ffffff",
            "text":    "#1a1d2e",
            "grid":    "#d0d5e8",
            "accent1": "#1a56db",
            "accent2": "#4361ee",
            "accent3": "#0b7285",
            "red":     "#c0392b",
            "orange":  "#e67e22",
            "tick":    "#4a5170",
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
# SECTION 7 - STRUCTURE GENERATION & ANALYSIS
# ==========================================================

def build_peptide_pdb(seq):
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        PeptideBuilder.add_residue(structure, Geometry.geometry(aa))
    io = PDBIO()
    io.set_structure(structure)
    io.save("predicted_peptide.pdb")
    with open("predicted_peptide.pdb") as f:
        return f.read()


def show_structure(pdb_text):
    view = py3Dmol.view(width=800, height=450)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view


def _write_temp_pdb(pdb_text):
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False)
    tmp.write(pdb_text)
    tmp.close()
    return tmp.name


def ramachandran(pdb_text):
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp_path)[0]
        pts = []
        for pp in PPBuilder().build_peptides(structure):
            for phi, psi in pp.get_phi_psi_list():
                if phi and psi:
                    pts.append((np.degrees(phi), np.degrees(psi)))
        return pts
    finally:
        os.unlink(tmp_path)


def ca_distance_map(pdb_text):
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp_path)
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
    finally:
        os.unlink(tmp_path)


def ca_rmsd(pdb_text):
    tmp_path = _write_temp_pdb(pdb_text)
    try:
        structure = PDBParser(QUIET=True).get_structure("x", tmp_path)
        cas = [
            r["CA"].get_vector()
            for r in structure.get_residues()
            if "CA" in r
        ]
        if len(cas) < 2:
            return None
        ref = cas[0]
        return np.sqrt(np.mean([(v - ref).norm() ** 2 for v in cas]))
    finally:
        os.unlink(tmp_path)


# ==========================================================
# SECTION 8 - PLOT FUNCTIONS
# ==========================================================

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
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=cls, alpha=0.75, s=35,
            color=palette(i), edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12, labelpad=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    legend = ax.legend(
        fontsize=8, bbox_to_anchor=(1.02, 1),
        loc="upper left", borderaxespad=0,
        title="Taste class", title_fontsize=9,
        facecolor=C["fig_bg"],
        edgecolor=C["grid"],
    )
    legend.get_title().set_color(C["text"])
    for text in legend.get_texts():
        text.set_color(C["text"])
    plt.tight_layout()
    return fig, pca


def plot_confusion(y_true, y_pred, class_names, title, cmap):
    C = get_plot_colors()
    cm  = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    n   = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.75), max(5, n * 0.6)))
    apply_plot_style(fig, [ax])
    annot_color = "#111122" if not _is_dark_mode() else "#ffffff"
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, linewidths=0.4, linecolor=C["grid"],
        annot_kws={"size": 11, "color": annot_color},
    )
    ax.set_title(
        f"{title}  —  Accuracy: {acc * 100:.1f}%",
        fontsize=14, fontweight="bold", pad=14,
    )
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
    C = get_plot_colors()
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lims = [
        min(y_true.min(), y_pred.min()) - 1,
        max(y_true.max(), y_pred.max()) + 1,
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.scatter(
        y_true, y_pred, alpha=0.65,
        edgecolors="none",
        color=C["accent1"], s=45,
    )
    ax.plot(lims, lims, color=C["red"], linestyle="--", lw=1.8, label="Perfect fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    box_bg = C["fig_bg"]
    ax.annotate(
        f"R² = {r2:.3f}\nRMSE = {rmse:.2f} kcal/mol",
        xy=(0.05, 0.87), xycoords="axes fraction", fontsize=11,
        color=C["text"],
        bbox=dict(boxstyle="round,pad=0.5", fc=box_bg, ec=C["grid"], alpha=0.95),
    )
    ax.set_xlabel("True Docking Score (kcal/mol)",      fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted Docking Score (kcal/mol)", fontsize=12, labelpad=10)
    ax.set_title("Docking: True vs Predicted (test set)", fontsize=13,
                 fontweight="bold", pad=12)
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
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1],
            color=colors, edgecolor=C["grid"])
    ax.set_xlabel("Importance Score", fontsize=12, labelpad=10)
    ax.set_title(f"Top {top_n} Features — Taste Model", fontsize=13,
                 fontweight="bold", pad=12)
    ax.tick_params(axis="y", labelsize=9, colors=C["tick"])
    ax.tick_params(axis="x", labelsize=9, colors=C["tick"])
    plt.tight_layout()
    return fig


def plot_distributions(df):
    C = get_plot_colors()
    seq_lengths  = [len(s) for s in df["peptide"]]
    taste_counts = df["taste"].value_counts()
    gravy_vals   = [gravy_score(s) for s in df["peptide"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    apply_plot_style(fig, axes)

    # Length histogram
    axes[0].hist(seq_lengths, bins=20,
                 color=C["accent1"], edgecolor=C["grid"], alpha=0.85)
    mean_len = np.mean(seq_lengths)
    axes[0].axvline(mean_len, color=C["red"], linestyle="--", lw=2,
                    label=f"Mean = {mean_len:.1f} aa")
    axes[0].set_xlabel("Sequence Length (aa)", fontsize=11, labelpad=8)
    axes[0].set_ylabel("Count",                fontsize=11, labelpad=8)
    axes[0].set_title("Peptide Length Distribution", fontsize=12,
                      fontweight="bold", pad=10)
    leg0 = axes[0].legend(fontsize=9, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg0.get_texts(): t.set_color(C["text"])

    # Taste bar chart
    n_cls      = len(taste_counts)
    bar_colors = plt.cm.get_cmap("tab20", n_cls)(np.linspace(0, 1, n_cls))
    axes[1].barh(taste_counts.index, taste_counts.values,
                 color=bar_colors, edgecolor=C["grid"], alpha=0.9)
    axes[1].set_xlabel("Number of Peptides", fontsize=11, labelpad=8)
    axes[1].set_title("Taste Class Distribution", fontsize=12,
                      fontweight="bold", pad=10)
    axes[1].tick_params(axis="y", labelsize=9, colors=C["tick"])
    axes[1].tick_params(axis="x", labelsize=9, colors=C["tick"])
    for i, v in enumerate(taste_counts.values):
        axes[1].text(v + 0.3, i, str(v), va="center",
                     fontsize=9, color=C["text"])

    # GRAVY histogram
    axes[2].hist(gravy_vals, bins=20,
                 color=C["accent2"], edgecolor=C["grid"], alpha=0.85)
    axes[2].axvline(0, color=C["red"], linestyle="--", lw=2,
                    label="Hydrophilic | Hydrophobic")
    axes[2].axvline(np.mean(gravy_vals), color=C["orange"], linestyle="--", lw=2,
                    label=f"Mean = {np.mean(gravy_vals):.2f}")
    axes[2].set_xlabel("GRAVY Score", fontsize=11, labelpad=8)
    axes[2].set_ylabel("Count",       fontsize=11, labelpad=8)
    axes[2].set_title("GRAVY Hydrophobicity Distribution", fontsize=12,
                      fontweight="bold", pad=10)
    leg2 = axes[2].legend(fontsize=8, facecolor=C["fig_bg"], edgecolor=C["grid"])
    for t in leg2.get_texts(): t.set_color(C["text"])

    plt.tight_layout(pad=2.5)
    return fig


def plot_ramachandran(phi_psi):
    C = get_plot_colors()
    fig, ax = plt.subplots(figsize=(6, 6))
    apply_plot_style(fig, [ax])
    ax.fill([-180, -180, -45, -45, -180], [-75, -45, -45, -75, -75],
            color="#4CAF50", alpha=0.25, label="α-helix (allowed)")
    ax.fill([-180, -180, -90, -90, -180], [90, 180, 180, 90, 90],
            color="#2196F3", alpha=0.25, label="β-sheet (allowed)")
    ax.fill([45, 45, 90, 90, 45], [0, 90, 90, 0, 0],
            color="#FF9800", alpha=0.2, label="L-helix (allowed)")
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=50, color=C["red"], zorder=5,
                   edgecolors="white", linewidths=0.5)
    ax.axhline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.axvline(0, color=C["grid"], lw=0.8, linestyle="--")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
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
        labels = [str(i + 1) for i in range(n)]
    tick_step   = max(1, n // 15)
    show_labels = [labels[i] if i % tick_step == 0 else "" for i in range(n)]
    size        = max(5, n * 0.3 + 2)
    fig, ax     = plt.subplots(figsize=(size, size))
    apply_plot_style(fig, [ax])
    sns.heatmap(
        dist_matrix, cmap="viridis", ax=ax,
        xticklabels=show_labels, yticklabels=show_labels,
        linewidths=0, cbar_kws={"label": "Distance (Å)"},
    )
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
# SECTION 9 - STRUCTURAL ANALYSIS HELPER
# ==========================================================

def render_structural_analysis(pdb_text, prefix="", seq=""):
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 📐 Ramachandran Plot")
    phi_psi = ramachandran(pdb_text)
    if not phi_psi:
        st.info("No phi/psi angles found — peptide may be too short.")
    fig_rama = plot_ramachandran(phi_psi)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.pyplot(fig_rama)
    plt.close(fig_rama)
    show_caption(caption_ramachandran(phi_psi, seq=seq))

    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
    st.markdown("### 🗺️ Cα Distance Map")
    dist_map = ca_distance_map(pdb_text)
    fig_dist = plot_distance_map(dist_map, seq=seq)
    save_fig(fig_dist, f"{prefix}ca_distance_map.png")
    st.pyplot(fig_dist)
    plt.close(fig_dist)
    show_caption(caption_distance_map(dist_map, seq=seq))


# ==========================================================
# SECTION 10 - MODEL TRAINING
# ==========================================================

@st.cache_data
def train_models():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found: {DATASET_PATH}")
        st.stop()

    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()
    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2].reset_index(drop=True)
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
# SECTION 11 - LOAD MODELS
# ==========================================================

(
    df_all, X_all, X_test, yt_test, ys_test, yd_test,
    taste_model, sol_model, dock_model,
    le_taste, le_sol, metrics,
) = train_models()


# ==========================================================
# SECTION 12 - PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics, prediction, image_paths):
    file_name = "PepTastePredictor_Full_Report.pdf"
    styles    = getSampleStyleSheet()
    doc       = SimpleDocTemplate(file_name, pagesize=A4)
    story     = []

    story.append(Paragraph("<b>PepTastePredictor</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking, and structural analysis platform.",
        styles["Normal"],
    ))
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
# SECTION 13 - HERO HEADER
# ==========================================================

st.markdown("""
<div class="hero">
<h1>🧬 PepTastePredictor</h1>
<p>
An integrated machine learning and structural bioinformatics platform
for peptide taste, solubility, docking, and structural analysis.
</p>
</div>
""", unsafe_allow_html=True)


# ==========================================================
# SECTION 14 - MODE SELECTION
# ==========================================================

st.markdown("## 🔧 Prediction & Analysis Mode")

mode = st.radio(
    "Choose the analysis mode",
    [
        "Single Peptide Prediction",
        "Batch Peptide Prediction",
        "PDB Upload & Structural Analysis",
    ],
    horizontal=True,
)

if "current_mode" not in st.session_state or st.session_state.current_mode != mode:
    st.session_state.pdf_figures     = []
    st.session_state.show_analytics  = False
    st.session_state.last_prediction = {}
    st.session_state.current_mode    = mode


# ==========================================================
# SECTION 15 - SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("## 🔬 Single Peptide Prediction")
    seq = st.text_input(
        "Enter peptide sequence (FASTA single-letter code)",
        help="Example: AGLWFK",
    )

    if st.button("Run Prediction"):
        st.session_state.pdf_figures = []
        seq = clean_sequence(seq)

        if len(seq) < 2:
            st.error("Peptide sequence must be at least 2 amino acids long.")
        else:
            Xp    = pd.DataFrame([model_features(seq)])
            taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
            sol   = le_sol.inverse_transform(sol_model.predict(Xp))[0]
            dock  = dock_model.predict(Xp)[0]

            st.session_state.last_prediction = {
                "Sequence":                 seq,
                "Predicted taste":          taste,
                "Predicted solubility":     sol,
                "Docking score (kcal/mol)": round(dock, 3),
            }
            st.session_state.show_analytics = True

            st.markdown(f"""
            <div class="card">
                <div class="metric-label">Taste</div>
                <div class="metric-value">{taste}</div>
                <div class="metric-label">Solubility</div>
                <div class="metric-value">{sol}</div>
                <div class="metric-label">Docking Score</div>
                <div class="metric-value">{dock:.3f} kcal/mol</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 📌 Physicochemical Properties")
            for k, v in physicochemical_features(seq).items():
                st.write(f"**{k}**: {v}")

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("### 🧪 Amino Acid Composition")
            for k, v in composition_features(seq).items():
                st.write(f"**{k}**: {v}")

            st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
            st.markdown("## 🧬 Predicted 3D Peptide Structure")
            pdb_text = build_peptide_pdb(seq)
            st.session_state.pdb_text = pdb_text

            st.download_button("⬇️ Download Predicted PDB", pdb_text,
                               file_name="predicted_peptide.pdb")
            st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)

            rmsd_val = ca_rmsd(pdb_text)
            if rmsd_val is not None:
                st.success(f"Cα RMSD: {rmsd_val:.3f} Å")

            render_structural_analysis(pdb_text, prefix="single_", seq=seq)


# ==========================================================
# SECTION 16 - BATCH PEPTIDE PREDICTION MODE
# ==========================================================

elif mode == "Batch Peptide Prediction":

    st.markdown("## 📦 Batch Peptide Prediction")
    batch_file = st.file_uploader(
        "Upload CSV file with a column named 'peptide'", type=["csv"])

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)
        if "peptide" not in batch_df.columns:
            st.error("CSV must contain a column named 'peptide'")
        else:
            batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
            batch_df = batch_df[batch_df["peptide"].str.len() >= 2].reset_index(drop=True)
            X_batch  = build_feature_table(batch_df["peptide"])
            batch_df["Predicted Taste"]         = le_taste.inverse_transform(taste_model.predict(X_batch))
            batch_df["Predicted Solubility"]    = le_sol.inverse_transform(sol_model.predict(X_batch))
            batch_df["Predicted Docking Score"] = dock_model.predict(X_batch)

            st.markdown("### ✅ Batch Results")
            st.dataframe(batch_df)
            st.download_button(
                "⬇️ Download Batch Predictions",
                batch_df.to_csv(index=False),
                file_name="batch_predictions.csv",
            )
            st.session_state.show_analytics = True


# ==========================================================
# SECTION 17 - PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

elif mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyze PDB Structure")
    uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])

    if uploaded_pdb is not None:
        pdb_text = uploaded_pdb.read().decode()
        st.session_state.pdb_text       = pdb_text
        st.session_state.show_analytics = True

        st.markdown("### 🧬 3D Structure Viewer")
        st.components.v1.html(show_structure(pdb_text)._make_html(), height=520)

        rmsd_val = ca_rmsd(pdb_text)
        if rmsd_val is not None:
            st.success(f"Cα RMSD: {rmsd_val:.3f} Å")

        render_structural_analysis(pdb_text, prefix="pdb_")


# ==========================================================
# SECTION 18 - MODEL & DATASET ANALYTICS
# ==========================================================

if st.session_state.show_analytics:

    st.markdown("---")

    with st.expander("📊 Model Performance & Dataset Analytics", expanded=False):

        st.markdown("### 📈 Model Performance Metrics")
        for k, v in metrics.items():
            st.write(f"**{k}**: {round(v, 4)}")

        # Dataset Distributions
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Dataset Distributions")
        fig_dist = plot_distributions(df_all)
        save_fig(fig_dist, "distributions.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)
        show_caption(caption_distributions(df_all))

        # PCA
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 PCA: Peptide Feature Space (coloured by taste)")
        fig_pca, pca_model = plot_pca(
            X_all, le_taste.transform(df_all["taste"]), le_taste.classes_,
            title="PCA — Peptide Feature Space (coloured by taste class)",
        )
        save_fig(fig_pca, "pca_overall.png")
        st.pyplot(fig_pca)
        plt.close(fig_pca)
        show_caption(caption_pca(pca_model, le_taste.classes_))

        # Confusion Matrix — Taste
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Confusion Matrix — Taste (test set)")
        taste_preds  = taste_model.predict(X_test)
        fig_cm_taste = plot_confusion(
            yt_test, taste_preds,
            le_taste.classes_, title="Taste Confusion Matrix", cmap="Blues",
        )
        save_fig(fig_cm_taste, "confusion_taste.png")
        st.pyplot(fig_cm_taste)
        plt.close(fig_cm_taste)
        show_caption(caption_confusion_taste(yt_test, taste_preds, le_taste.classes_))

        # Confusion Matrix — Solubility
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Confusion Matrix — Solubility (test set)")
        sol_preds  = sol_model.predict(X_test)
        fig_cm_sol = plot_confusion(
            ys_test, sol_preds,
            le_sol.classes_, title="Solubility Confusion Matrix", cmap="Greens",
        )
        save_fig(fig_cm_sol, "confusion_solubility.png")
        st.pyplot(fig_cm_sol)
        plt.close(fig_cm_sol)
        show_caption(caption_confusion_sol(ys_test, sol_preds, le_sol.classes_))

        # Feature Importance
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Feature Importance — Taste Model")
        fig_imp = plot_feature_importance(taste_model, X_all.columns, top_n=20)
        save_fig(fig_imp, "feature_importance_taste.png")
        st.pyplot(fig_imp)
        plt.close(fig_imp)
        show_caption(caption_feature_importance(taste_model, X_all.columns, top_n=20))

        # Docking Scatter
        st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)
        st.markdown("### 🔹 Docking Score: True vs Predicted (test set)")
        dock_preds = dock_model.predict(X_test)
        fig_dock   = plot_docking(yd_test, dock_preds)
        save_fig(fig_dock, "docking_scatter.png")
        st.pyplot(fig_dock)
        plt.close(fig_dock)
        show_caption(caption_docking(yd_test, dock_preds))


# ==========================================================
# SECTION 19 - PDF DOWNLOAD
# ==========================================================

if st.session_state.show_analytics and len(st.session_state.pdf_figures) > 0:

    st.markdown("## 📄 Download Complete PDF Report")
    pdf_path = generate_pdf(
        metrics, st.session_state.last_prediction, st.session_state.pdf_figures)
    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Download Full Analytics PDF", f,
            file_name="PepTastePredictor_Full_Report.pdf",
            mime="application/pdf",
        )


# ==========================================================
# SECTION 20 - FOOTER (dynamic year)
# ==========================================================

st.markdown(f"""
<div class="footer">
&copy; {date.today().year} &nbsp; <b>PepTastePredictor</b><br>
An AI + Structural Bioinformatics platform for peptide analysis<br>
For academic, educational, and research use
</div>
""", unsafe_allow_html=True)
