# ==========================================================
# PepTastePredictor — app.py
# A complete end-to-end peptide analysis platform
# ==========================================================
#
# Integrates:
#   • Machine Learning  (Taste, Solubility, Docking)
#   • Structural Bioinformatics (PDB, RMSD, Ramachandran)
#   • Visualization (3D, PCA, Heatmaps, Distributions)
#   • Batch Screening
#   • Automated PDF Report Generation
#
# All fixes applied:
#   1.  Solubility label noise cleaned
#   2.  Rare taste classes merged (< 5 samples)
#   3.  ExtraTrees + class_weight="balanced"
#   4.  train_test_split fixed (single index reused)
#   5.  Confusion matrix on held-out test set only
#   6.  All file handles use with-open
#   7.  Temp PDB files use tempfile (no session collisions)
#   8.  logo.png guarded with os.path.exists
#   9.  Missing dataset -> graceful st.stop()
#  10.  if mode -> elif
#  11.  Duplicate structural analysis extracted to helper
#  12.  Section numbering fixed
#  13.  CSS dark-mode safe (no hardcoded heading colour)
#  14.  PDF figures cleared on mode switch
#  15.  ProteinAnalysis created once per sequence
#  16.  ALL_DIPEPTIDES pre-enumerated (fixes ValueError)
#  17.  PCA coloured by taste class + variance % on axes
#  18.  Docking plot with R2 + RMSE annotated on chart
#  19.  Confusion matrices show accuracy % in title
#  20.  Ramachandran plot with shaded allowed regions
#  21.  Distance map with residue number/letter labels
#  22.  Feature importance with readable human names
#  23.  Distribution plots (length, taste classes, GRAVY)
#  24.  Footer year dynamic (date.today().year)
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

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"

ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]

KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}


# ==========================================================
# SECTION 3 - FRONTEND STYLING
# ==========================================================

st.markdown(
    """
<style>
.card {
    background-color: var(--secondary-background-color);
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.hero {
    background: linear-gradient(90deg, #1f3c88, #0b7285);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}
.metric {
    font-size: 20px;
    font-weight: 600;
    color: #0b7285;
}
.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    padding: 30px;
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
        diff = coords[:, None, :] - coords[None, :, :]
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
# SECTION 8 - ENHANCED PLOT FUNCTIONS (all dynamic)
# ==========================================================

def plot_pca(X, y_labels, class_names, title="PCA"):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    palette = plt.cm.get_cmap("tab20", len(class_names))
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, cls in enumerate(class_names):
        mask = y_labels == i
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=cls, alpha=0.75, s=35,
            color=palette(i), edgecolors="white", linewidths=0.3,
        )
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(
        fontsize=7, bbox_to_anchor=(1.02, 1),
        loc="upper left", borderaxespad=0,
        title="Taste class", title_fontsize=8,
    )
    plt.tight_layout()
    return fig


def plot_confusion(y_true, y_pred, class_names, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.55)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax, linewidths=0.4, linecolor="white",
    )
    ax.set_title(f"{title}  —  Accuracy: {acc * 100:.1f}%", fontsize=13)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    return fig


def plot_docking(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    lims = [
        min(y_true.min(), y_pred.min()) - 1,
        max(y_true.max(), y_pred.max()) + 1,
    ]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        y_true, y_pred, alpha=0.65,
        edgecolors="k", linewidths=0.3, color="#1f3c88", s=40,
    )
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect fit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.annotate(
        f"R\u00b2 = {r2:.3f}\nRMSE = {rmse:.2f} kcal/mol",
        xy=(0.05, 0.88), xycoords="axes fraction", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.9),
    )
    ax.set_xlabel("True Docking Score (kcal/mol)", fontsize=11)
    ax.set_ylabel("Predicted Docking Score (kcal/mol)", fontsize=11)
    ax.set_title("Docking: True vs Predicted (test set)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20):
    imp = pd.DataFrame({
        "Feature":    [prettify_feature(f) for f in feature_names],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(imp))[::-1])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(imp["Feature"][::-1], imp["Importance"][::-1], color=colors)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Features — Taste Model", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    return fig


def plot_distributions(df):
    seq_lengths  = [len(s) for s in df["peptide"]]
    taste_counts = df["taste"].value_counts()
    gravy_vals   = [gravy_score(s) for s in df["peptide"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Length
    axes[0].hist(seq_lengths, bins=20, color="#1f3c88", edgecolor="white", alpha=0.85)
    mean_len = np.mean(seq_lengths)
    axes[0].axvline(mean_len, color="red", linestyle="--", lw=1.5,
                    label=f"Mean = {mean_len:.1f} aa")
    axes[0].set_xlabel("Sequence Length (aa)", fontsize=10)
    axes[0].set_ylabel("Count", fontsize=10)
    axes[0].set_title("Peptide Length Distribution", fontsize=11)
    axes[0].legend(fontsize=8)

    # Taste classes
    n_cls = len(taste_counts)
    bar_colors = plt.cm.get_cmap("tab20", n_cls)(np.linspace(0, 1, n_cls))
    axes[1].barh(taste_counts.index, taste_counts.values,
                 color=bar_colors, edgecolor="white", alpha=0.9)
    axes[1].set_xlabel("Number of Peptides", fontsize=10)
    axes[1].set_title("Taste Class Distribution", fontsize=11)
    axes[1].tick_params(axis="y", labelsize=8)
    for i, v in enumerate(taste_counts.values):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=8)

    # GRAVY
    axes[2].hist(gravy_vals, bins=20, color="#5c6bc0", edgecolor="white", alpha=0.85)
    axes[2].axvline(0, color="red", linestyle="--", lw=1.5,
                    label="Hydrophilic | Hydrophobic")
    axes[2].axvline(np.mean(gravy_vals), color="orange", linestyle="--", lw=1.5,
                    label=f"Mean = {np.mean(gravy_vals):.2f}")
    axes[2].set_xlabel("GRAVY Score", fontsize=10)
    axes[2].set_ylabel("Count", fontsize=10)
    axes[2].set_title("GRAVY Hydrophobicity Distribution", fontsize=11)
    axes[2].legend(fontsize=7)

    plt.tight_layout()
    return fig


def plot_ramachandran(phi_psi):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#f8f8f8")
    # Shaded allowed regions
    ax.fill([-180, -180, -45, -45, -180], [-75, -45, -45, -75, -75],
            color="#4CAF50", alpha=0.2, label="\u03b1-helix (allowed)")
    ax.fill([-180, -180, -90, -90, -180], [90, 180, 180, 90, 90],
            color="#2196F3", alpha=0.2, label="\u03b2-sheet (allowed)")
    ax.fill([45, 45, 90, 90, 45], [0, 90, 90, 0, 0],
            color="#FF9800", alpha=0.15, label="L-helix (allowed)")
    if phi_psi:
        phi, psi = zip(*phi_psi)
        ax.scatter(phi, psi, s=35, color="#c0392b", zorder=5,
                   edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--")
    ax.axvline(0, color="gray", lw=0.5, linestyle="--")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi \u03c6 (\u00b0)", fontsize=11)
    ax.set_ylabel("Psi \u03c8 (\u00b0)", fontsize=11)
    ax.set_title("Ramachandran Plot", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-180, 181, 60))
    plt.tight_layout()
    return fig


def plot_distance_map(dist_matrix, seq=""):
    n = dist_matrix.shape[0]
    if seq and len(seq) == n:
        labels = [f"{aa}{i+1}" for i, aa in enumerate(seq)]
    else:
        labels = [str(i + 1) for i in range(n)]
    tick_step = max(1, n // 15)
    show_labels = [labels[i] if i % tick_step == 0 else "" for i in range(n)]
    size = max(5, n * 0.3 + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        dist_matrix, cmap="viridis", ax=ax,
        xticklabels=show_labels, yticklabels=show_labels,
        linewidths=0, cbar_kws={"label": "Distance (\u00c5)"},
    )
    ax.set_title("C\u03b1 Distance Map", fontsize=12)
    ax.set_xlabel("Residue", fontsize=10)
    ax.set_ylabel("Residue", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    return fig


# ==========================================================
# SECTION 9 - STRUCTURAL ANALYSIS HELPER
# ==========================================================

def render_structural_analysis(pdb_text, prefix="", seq=""):
    st.markdown("### 📐 Ramachandran Plot")
    phi_psi = ramachandran(pdb_text)
    if not phi_psi:
        st.info("No phi/psi angles found — peptide may be too short.")
    fig_rama = plot_ramachandran(phi_psi)
    save_fig(fig_rama, f"{prefix}ramachandran.png")
    st.pyplot(fig_rama)
    plt.close(fig_rama)

    st.markdown("### 🗺️ Cα Distance Map")
    dist_map = ca_distance_map(pdb_text)
    fig_dist = plot_distance_map(dist_map, seq=seq)
    save_fig(fig_dist, f"{prefix}ca_distance_map.png")
    st.pyplot(fig_dist)
    plt.close(fig_dist)


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
    df["taste"] = simplify_taste(df["taste"])

    X = build_feature_table(df["peptide"])
    le_taste = LabelEncoder()
    le_sol   = LabelEncoder()
    y_taste  = le_taste.fit_transform(df["taste"])
    y_sol    = le_sol.fit_transform(df["solubility"])
    y_dock   = df["docking score (kcal/mol)"].values

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y_taste
    )
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    yt_tr, yt_te = y_taste[tr_idx], y_taste[te_idx]
    ys_tr, ys_te = y_sol[tr_idx],   y_sol[te_idx]
    yd_tr, yd_te = y_dock[tr_idx],  y_dock[te_idx]

    taste_model = ExtraTreesClassifier(
        n_estimators=500, class_weight="balanced", random_state=42)
    sol_model = ExtraTreesClassifier(
        n_estimators=300, class_weight="balanced", random_state=42)
    dock_model = RandomForestRegressor(n_estimators=400, random_state=42)

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
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_name, pagesize=A4)
    story = []

    story.append(Paragraph("<b>PepTastePredictor</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking, and structural analysis platform.",
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
    st.session_state.pdf_figures = []
    st.session_state.show_analytics = False
    st.session_state.last_prediction = {}
    st.session_state.current_mode = mode


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
            Xp = pd.DataFrame([model_features(seq)])
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
                <div class="metric">Taste</div><p>{taste}</p>
                <div class="metric">Solubility</div><p>{sol}</p>
                <div class="metric">Docking Score</div><p>{dock:.3f} kcal/mol</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📌 Physicochemical Properties")
            for k, v in physicochemical_features(seq).items():
                st.write(f"**{k}**: {v}")

            st.markdown("### 🧪 Amino Acid Composition")
            for k, v in composition_features(seq).items():
                st.write(f"**{k}**: {v}")

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
            X_batch = build_feature_table(batch_df["peptide"])
            batch_df["Predicted Taste"] = le_taste.inverse_transform(
                taste_model.predict(X_batch))
            batch_df["Predicted Solubility"] = le_sol.inverse_transform(
                sol_model.predict(X_batch))
            batch_df["Predicted Docking Score"] = dock_model.predict(X_batch)

            st.markdown("### ✅ Batch Results")
            st.dataframe(batch_df)
            st.download_button("⬇️ Download Batch Predictions",
                               batch_df.to_csv(index=False),
                               file_name="batch_predictions.csv")
            st.session_state.show_analytics = True


# ==========================================================
# SECTION 17 - PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

elif mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyze PDB Structure")
    uploaded_pdb = st.file_uploader("Upload a PDB file", type=["pdb"])

    if uploaded_pdb is not None:
        pdb_text = uploaded_pdb.read().decode()
        st.session_state.pdb_text = pdb_text
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

        st.markdown("### 📊 Dataset Distributions")
        fig_dist = plot_distributions(df_all)
        save_fig(fig_dist, "distributions.png")
        st.pyplot(fig_dist)
        plt.close(fig_dist)

        st.markdown("### 🔹 PCA: Peptide Feature Space (coloured by taste)")
        fig_pca = plot_pca(
            X_all, le_taste.transform(df_all["taste"]), le_taste.classes_,
            title="PCA — Peptide Feature Space (coloured by taste class)",
        )
        save_fig(fig_pca, "pca_overall.png")
        st.pyplot(fig_pca)
        plt.close(fig_pca)

        st.markdown("### 🔹 Confusion Matrix — Taste (test set)")
        fig_cm_taste = plot_confusion(
            yt_test, taste_model.predict(X_test),
            le_taste.classes_, title="Taste Confusion Matrix", cmap="Blues",
        )
        save_fig(fig_cm_taste, "confusion_taste.png")
        st.pyplot(fig_cm_taste)
        plt.close(fig_cm_taste)

        st.markdown("### 🔹 Confusion Matrix — Solubility (test set)")
        fig_cm_sol = plot_confusion(
            ys_test, sol_model.predict(X_test),
            le_sol.classes_, title="Solubility Confusion Matrix", cmap="Greens",
        )
        save_fig(fig_cm_sol, "confusion_solubility.png")
        st.pyplot(fig_cm_sol)
        plt.close(fig_cm_sol)

        st.markdown("### 🔹 Feature Importance — Taste Model")
        fig_imp = plot_feature_importance(taste_model, X_all.columns, top_n=20)
        save_fig(fig_imp, "feature_importance_taste.png")
        st.pyplot(fig_imp)
        plt.close(fig_imp)

        st.markdown("### 🔹 Docking Score: True vs Predicted (test set)")
        fig_dock = plot_docking(yd_test, dock_model.predict(X_test))
        save_fig(fig_dock, "docking_scatter.png")
        st.pyplot(fig_dock)
        plt.close(fig_dock)


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
&copy; {date.today().year} <b>PepTastePredictor</b><br>
An AI + Structural Bioinformatics platform for peptide analysis<br>
For academic, educational, and research use
</div>
""", unsafe_allow_html=True)
