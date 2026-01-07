# ==========================================================
# PepTastePredictor
# A COMPLETE END-TO-END PEPTIDE ANALYSIS PLATFORM
# ==========================================================
#
# This application integrates:
#   • Machine Learning (Taste, Solubility, Docking)
#   • Structural Bioinformatics (PDB, RMSD, Ramachandran)
#   • Visualization (3D, PCA, Heatmaps)
#   • Batch Screening
#   • Automated PDF Report Generation
#
# ==========================================================


# ==========================================================
# SECTION 1 — IMPORTS
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import py3Dmol
import os

from collections import Counter

from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBIO, PDBParser, PPBuilder
import PeptideBuilder
from PeptideBuilder import Geometry

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix
)
from sklearn.decomposition import PCA

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image as RLImage,
    Spacer
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


# ==========================================================
# SECTION 2 — GLOBAL CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="PepTastePredictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_PATH = "AIML (4).xlsx"
AA = "ACDEFGHIKLMNPQRSTVWY"


# ==========================================================
# SECTION 3 — FRONTEND STYLING (AUTO LIGHT / DARK)
# ==========================================================

st.markdown(
    """
<style>

/* ===============================
   BASE THEME (AUTO-ADAPTIVE)
   =============================== */

.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

/* Card */
.card {
    background-color: var(--secondary-background-color);
    padding: 22px;
    border-radius: 14px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Hero banner */
.hero {
    background: linear-gradient(90deg, #1f3c88, #0b7285);
    padding: 30px;
    border-radius: 16px;
    color: white;
    margin-bottom: 30px;
}

/* Titles */
h1, h2, h3 {
    color: #1f3c88;
}

/* Metric text */
.metric {
    font-size: 20px;
    font-weight: 600;
    color: #0b7285;
}

/* Footer */
.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    padding: 30px;
}

</style>
""",
    unsafe_allow_html=True
)


# ==========================================================
# SECTION 4 — SIDEBAR
# ==========================================================

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
# SECTION 5 — SESSION STATE INITIALIZATION
# ==========================================================

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.pdb_text = None
    st.session_state.last_prediction = {}
    st.session_state.show_analytics = False
    st.session_state.pdf_figures = []


# ==========================================================
# SECTION 6 — UTILITY FUNCTIONS
# ==========================================================

def save_fig(fig, filename):
    """
    Save matplotlib figure AND register it for PDF export
    """
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    st.session_state.pdf_figures.append(filename)


def clean_sequence(seq):
    """
    Clean peptide sequence:
    • uppercase
    • remove whitespace
    • keep valid amino acids only
    """
    if not isinstance(seq, str):
        return ""
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)


def model_features(seq):
    """
    Convert peptide sequence into ML feature vector
    """
    ana = ProteinAnalysis(seq)

    features = {
        "length": len(seq),
        "mw": ana.molecular_weight(),
        "pI": ana.isoelectric_point(),
        "aromaticity": ana.aromaticity(),
        "instability": ana.instability_index(),
        "gravy": ana.gravy(),
        "charge": ana.charge_at_pH(7.0)
    }

    for aa in AA:
        features[f"AA_{aa}"] = seq.count(aa) / len(seq)

    return features


def build_feature_table(seqs):
    """
    Build ML-ready dataframe from peptide list
    """
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)


def physicochemical_features(seq):
    """
    Compute physicochemical properties
    """
    ana = ProteinAnalysis(seq)
    h, t, s = ana.secondary_structure_fraction()

    return {
        "Length": len(seq),
        "Molecular weight (Da)": round(ana.molecular_weight(), 2),
        "Isoelectric point": round(ana.isoelectric_point(), 2),
        "Net charge (pH 7)": round(ana.charge_at_pH(7.0), 2),
        "Aromaticity": round(ana.aromaticity(), 3),
        "GRAVY": round(ana.gravy(), 3),
        "Instability index": round(ana.instability_index(), 2),
        "Helix fraction": round(h, 3),
        "Turn fraction": round(t, 3),
        "Sheet fraction": round(s, 3),
    }


def composition_features(seq):
    """
    Amino acid group composition
    """
    c = Counter(seq)
    L = len(seq)

    return {
        "Hydrophobic (%)": round(100 * sum(c[a] for a in "AILMFWV") / L, 1),
        "Polar (%)": round(100 * sum(c[a] for a in "STNQ") / L, 1),
        "Charged (%)": round(100 * sum(c[a] for a in "DEKRH") / L, 1),
        "Aromatic (%)": round(100 * sum(c[a] for a in "FWY") / L, 1),
    }

# ==========================================================
# SECTION 7 — STRUCTURE GENERATION & ANALYSIS
# ==========================================================

def build_peptide_pdb(seq):
    """
    Generate peptide PDB using PeptideBuilder
    """
    structure = PeptideBuilder.initialize_res(seq[0])
    for aa in seq[1:]:
        PeptideBuilder.add_residue(structure, Geometry.geometry(aa))

    io = PDBIO()
    io.set_structure(structure)
    io.save("predicted_peptide.pdb")

    return open("predicted_peptide.pdb").read()


def show_structure(pdb_text):
    """
    Render 3D structure using py3Dmol
    """
    view = py3Dmol.view(width=800, height=450)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    return view


def ramachandran(pdb_text):
    """
    Calculate phi-psi angles
    """
    open("_rama_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_rama_tmp.pdb")[0]

    pts = []
    for pp in PPBuilder().build_peptides(structure):
        for phi, psi in pp.get_phi_psi_list():
            if phi and psi:
                pts.append((np.degrees(phi), np.degrees(psi)))
    return pts


def ca_distance_map(pdb_text):
    """
    C-alpha distance matrix
    """
    open("_ca_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_ca_tmp.pdb")

    cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
    n = len(cas)
    mat = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            mat[i, j] = (cas[i] - cas[j]).norm()

    return mat


def ca_rmsd(pdb_text):
    """
    RMSD relative to first residue
    """
    open("_rmsd_tmp.pdb", "w").write(pdb_text)
    structure = PDBParser(QUIET=True).get_structure("x", "_rmsd_tmp.pdb")

    cas = [r["CA"].get_vector() for r in structure.get_residues() if "CA" in r]
    if len(cas) < 2:
        return None

    ref = cas[0]
    return np.sqrt(np.mean([(v - ref).norm() ** 2 for v in cas]))


# ==========================================================
# SECTION 8 — MODEL TRAINING
# ==========================================================

@st.cache_data
def train_models():
    """
    Train RF models for taste, solubility, docking
    """
    df = pd.read_excel(DATASET_PATH)
    df.columns = df.columns.str.lower().str.strip()

    df["peptide"] = df["peptide"].apply(clean_sequence)
    df = df[df["peptide"].str.len() >= 2]

    X = build_feature_table(df["peptide"])

    le_taste = LabelEncoder()
    le_sol = LabelEncoder()

    y_taste = le_taste.fit_transform(df["taste"])
    y_sol = le_sol.fit_transform(df["solubility"])
    y_dock = df["docking score (kcal/mol)"]

    Xtr, Xte, yt_tr, yt_te, ys_tr, ys_te, yd_tr, yd_te = train_test_split(
        X, y_taste, y_sol, y_dock,
        test_size=0.2,
        random_state=42
    )

    taste_model = RandomForestClassifier(n_estimators=300, random_state=42)
    sol_model = RandomForestClassifier(n_estimators=300, random_state=42)
    dock_model = RandomForestRegressor(n_estimators=400, random_state=42)

    taste_model.fit(Xtr, yt_tr)
    sol_model.fit(Xtr, ys_tr)
    dock_model.fit(Xtr, yd_tr)

    metrics = {
        "Taste accuracy": accuracy_score(yt_te, taste_model.predict(Xte)),
        "Taste F1": f1_score(yt_te, taste_model.predict(Xte), average="weighted"),
        "Solubility accuracy": accuracy_score(ys_te, sol_model.predict(Xte)),
        "Solubility F1": f1_score(ys_te, sol_model.predict(Xte), average="weighted"),
        "Docking RMSE": np.sqrt(mean_squared_error(yd_te, dock_model.predict(Xte))),
        "Docking R²": r2_score(yd_te, dock_model.predict(Xte)),
    }

    return df, X, taste_model, sol_model, dock_model, le_taste, le_sol, metrics


# ==========================================================
# SECTION 9 — LOAD MODELS
# ==========================================================

df_all, X_all, taste_model, sol_model, dock_model, le_taste, le_sol, metrics = train_models()


# ==========================================================
# SECTION 10 — PDF REPORT ENGINE
# ==========================================================

def generate_pdf(metrics, prediction, image_paths):
    """
    Build full analytics PDF
    """
    file_name = "PepTastePredictor_Full_Report.pdf"
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(file_name, pagesize=A4)

    story = []

    story.append(Paragraph("<b>PepTastePredictor</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "AI-driven peptide taste, solubility, docking, and structural analysis platform.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Model Performance</b>", styles["Heading2"]))
    for k, v in metrics.items():
        story.append(Paragraph(f"{k}: {round(v, 4)}", styles["Normal"]))
    story.append(Spacer(1, 12))

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
# SECTION 11 — HERO HEADER
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
# SECTION 12 — MODE SELECTION
# ==========================================================

st.markdown("## 🔧 Prediction & Analysis Mode Selection")

mode = st.radio(
    "Choose the analysis mode",
    [
        "Single Peptide Prediction",
        "Batch Peptide Prediction",
        "PDB Upload & Structural Analysis"
    ],
    horizontal=True
)
# ==========================================================
# SECTION 13 — SINGLE PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Single Peptide Prediction":

    st.markdown("## 🔬 Single Peptide Prediction")

    seq = st.text_input(
        "Enter peptide sequence (FASTA single-letter code)",
        help="Example: AGLWFK"
    )

    if st.button("Run Prediction"):
        st.session_state.pdf_figures = []

        # -------------------------------
        # Clean & validate sequence
        # -------------------------------
        seq = clean_sequence(seq)

        if len(seq) < 2:
            st.error("Peptide sequence must be at least 2 amino acids long.")
        else:
            Xp = pd.DataFrame([model_features(seq)])

            # -------------------------------
            # ML Predictions
            # -------------------------------
            taste = le_taste.inverse_transform(taste_model.predict(Xp))[0]
            sol = le_sol.inverse_transform(sol_model.predict(Xp))[0]
            dock = dock_model.predict(Xp)[0]

            st.session_state.last_prediction = {
                "Sequence": seq,
                "Predicted taste": taste,
                "Predicted solubility": sol,
                "Docking score (kcal/mol)": round(dock, 3)
            }

            st.session_state.show_analytics = True

            # -------------------------------
            # Summary card
            # -------------------------------
            st.markdown(f"""
            <div class="card">
                <div class="metric">Taste</div>
                <p>{taste}</p>
                <div class="metric">Solubility</div>
                <p>{sol}</p>
                <div class="metric">Docking Score</div>
                <p>{dock:.3f} kcal/mol</p>
            </div>
            """, unsafe_allow_html=True)

            # -------------------------------
            # Physicochemical properties
            # -------------------------------
            st.markdown("### 📌 Physicochemical Properties")
            props = physicochemical_features(seq)
            for k, v in props.items():
                st.write(f"**{k}**: {v}")

            # -------------------------------
            # Composition analysis
            # -------------------------------
            st.markdown("### 🧪 Amino Acid Composition")
            comp = composition_features(seq)
            for k, v in comp.items():
                st.write(f"**{k}**: {v}")

            # -------------------------------
            # Structure generation
            # -------------------------------
            st.markdown("## 🧬 Predicted 3D Peptide Structure")

            pdb_text = build_peptide_pdb(seq)
            st.session_state.pdb_text = pdb_text

            st.download_button(
                "⬇️ Download Predicted PDB",
                pdb_text,
                file_name="predicted_peptide.pdb"
            )

            st.components.v1.html(
                show_structure(pdb_text)._make_html(),
                height=520
            )

            # -------------------------------
            # RMSD
            # -------------------------------
            rmsd_val = ca_rmsd(pdb_text)
            if rmsd_val is not None:
                st.success(f"Cα RMSD: {rmsd_val:.3f} Å")

            # -------------------------------
            # Ramachandran plot
            # -------------------------------
            st.markdown("### 📐 Ramachandran Plot")

            phi_psi = ramachandran(pdb_text)
            if phi_psi:
                phi, psi = zip(*phi_psi)
                fig_rama, ax_rama = plt.subplots()
                ax_rama.scatter(phi, psi, s=25)
                ax_rama.set_xlabel("Phi (°)")
                ax_rama.set_ylabel("Psi (°)")
                ax_rama.set_title("Ramachandran Plot")
                save_fig(fig_rama, "ramachandran.png")
                st.pyplot(fig_rama)

            # -------------------------------
            # Cα distance map
            # -------------------------------
            st.markdown("### 🗺️ Cα Distance Map")

            dist_map = ca_distance_map(pdb_text)
            fig_dist, ax_dist = plt.subplots(figsize=(5, 5))
            sns.heatmap(dist_map, cmap="viridis", ax=ax_dist)
            ax_dist.set_title("Cα Distance Heatmap")
            save_fig(fig_dist, "ca_distance_map.png")
            st.pyplot(fig_dist)

           

# ==========================================================
# SECTION 14 — BATCH PEPTIDE PREDICTION MODE
# ==========================================================

if mode == "Batch Peptide Prediction":

    st.markdown("## 📦 Batch Peptide Prediction")

    batch_file = st.file_uploader(
        "Upload CSV file with a column named 'peptide'",
        type=["csv"]
    )

    if batch_file is not None:
        batch_df = pd.read_csv(batch_file)

        if "peptide" not in batch_df.columns:
            st.error("CSV must contain a column named 'peptide'")
        else:
            batch_df["peptide"] = batch_df["peptide"].apply(clean_sequence)
            batch_df = batch_df[batch_df["peptide"].str.len() >= 2]

            X_batch = build_feature_table(batch_df["peptide"])

            batch_df["Predicted Taste"] = le_taste.inverse_transform(
                taste_model.predict(X_batch)
            )
            batch_df["Predicted Solubility"] = le_sol.inverse_transform(
                sol_model.predict(X_batch)
            )
            batch_df["Predicted Docking Score"] = dock_model.predict(X_batch)

            st.markdown("### ✅ Batch Results")
            st.dataframe(batch_df)

            st.download_button(
                "⬇️ Download Batch Predictions",
                batch_df.to_csv(index=False),
                file_name="batch_predictions.csv"
            )

            st.session_state.show_analytics = True


# ==========================================================
# SECTION 15 — PDB UPLOAD & STRUCTURAL ANALYSIS MODE
# ==========================================================

if mode == "PDB Upload & Structural Analysis":

    st.markdown("## 🧩 Upload & Analyze PDB Structure")

    uploaded_pdb = st.file_uploader(
        "Upload a PDB file",
        type=["pdb"]
    )

    if uploaded_pdb is not None:
        pdb_text = uploaded_pdb.read().decode()
        st.session_state.pdb_text = pdb_text
        st.session_state.show_analytics = True

        # -------------------------------
        # 3D viewer
        # -------------------------------
        st.markdown("### 🧬 3D Structure Viewer")
        st.components.v1.html(
            show_structure(pdb_text)._make_html(),
            height=520
        )

        # -------------------------------
        # RMSD
        # -------------------------------
        rmsd_val = ca_rmsd(pdb_text)
        if rmsd_val is not None:
            st.success(f"Cα RMSD: {rmsd_val:.3f} Å")

        # -------------------------------
        # Ramachandran
        # -------------------------------
        st.markdown("### 📐 Ramachandran Plot")

        phi_psi = ramachandran(pdb_text)
        if phi_psi:
            phi, psi = zip(*phi_psi)
            fig_rama, ax = plt.subplots()
            ax.scatter(phi, psi)
            ax.set_xlabel("Phi (°)")
            ax.set_ylabel("Psi (°)")
            ax.set_title("Ramachandran Plot")
            save_fig(fig_rama, "pdb_ramachandran.png")
            st.pyplot(fig_rama)

        # -------------------------------
        # Cα distance map
        # -------------------------------
        st.markdown("### 🗺️ Cα Distance Map")

        fig_map, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(
            ca_distance_map(pdb_text),
            cmap="viridis",
            ax=ax
        )
        ax.set_title("Cα Distance Heatmap")
        save_fig(fig_map, "pdb_ca_distance.png")
        st.pyplot(fig_map)
# ==========================================================
# SECTION 16 — MODEL & DATASET ANALYTICS (POST-ACTION ONLY)
# ==========================================================

if st.session_state.show_analytics:

    st.markdown("---")

    with st.expander("📊 Model Performance & Dataset Analytics", expanded=False):

        # -------------------------------
        # Model performance metrics
        # -------------------------------
        st.markdown("### 📈 Model Performance Metrics")
        for k, v in metrics.items():
            st.write(f"**{k}**: {round(v, 4)}")

        # -------------------------------
        # PCA — Overall
        # -------------------------------
        st.markdown("### 🔹 PCA: Overall Feature Space")

        pca_all = PCA(n_components=2)
        coords_all = pca_all.fit_transform(X_all)

        fig, ax = plt.subplots()
        ax.scatter(coords_all[:, 0], coords_all[:, 1], alpha=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("PCA of Peptide Feature Space")
        save_fig(fig, "pca_overall.png")
        st.pyplot(fig)

        # -------------------------------
        # Confusion Matrix — Taste
        # -------------------------------
        st.markdown("### 🔹 Confusion Matrix — Taste")

        y_true_taste = le_taste.transform(df_all["taste"])
        y_pred_taste = taste_model.predict(X_all)

        cm_taste = confusion_matrix(y_true_taste, y_pred_taste)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm_taste,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=le_taste.classes_,
            yticklabels=le_taste.classes_,
            ax=ax
        )
        ax.set_title("Taste Confusion Matrix")
        save_fig(fig, "confusion_taste.png")
        st.pyplot(fig)

        # -------------------------------
        # Confusion Matrix — Solubility
        # -------------------------------
        st.markdown("### 🔹 Confusion Matrix — Solubility")

        y_true_sol = le_sol.transform(df_all["solubility"])
        y_pred_sol = sol_model.predict(X_all)

        cm_sol = confusion_matrix(y_true_sol, y_pred_sol)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm_sol,
            annot=True,
            fmt="d",
            cmap="Greens",
            xticklabels=le_sol.classes_,
            yticklabels=le_sol.classes_,
            ax=ax
        )
        ax.set_title("Solubility Confusion Matrix")
        save_fig(fig, "confusion_solubility.png")
        st.pyplot(fig)

        # -------------------------------
        # Feature Importance — Taste
        # -------------------------------
        st.markdown("### 🔹 Feature Importance — Taste Model")

        imp_df = (
            pd.DataFrame({
                "Feature": X_all.columns,
                "Importance": taste_model.feature_importances_
            })
            .sort_values("Importance", ascending=False)
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax)
        ax.set_title("Top 20 Important Features (Taste)")
        save_fig(fig, "feature_importance_taste.png")
        st.pyplot(fig)

        # -------------------------------
        # Docking Scatter
        # -------------------------------
        st.markdown("### 🔹 Docking Score: True vs Predicted")

        true_dock = df_all["docking score (kcal/mol)"]
        pred_dock = dock_model.predict(X_all)

        fig, ax = plt.subplots()
        ax.scatter(true_dock, pred_dock, alpha=0.6)
        ax.plot(
            [true_dock.min(), true_dock.max()],
            [true_dock.min(), true_dock.max()],
            linestyle="--"
        )
        ax.set_title("Docking Prediction Performance")
        save_fig(fig, "docking_scatter.png")
        st.pyplot(fig)
# ==========================================================
# SECTION 16B — FINAL PDF DOWNLOAD (ALL PLOTS)
# ==========================================================

if st.session_state.show_analytics and len(st.session_state.pdf_figures) > 0:

    st.markdown("## 📄 Download Complete PDF Report")

    pdf_path = generate_pdf(
        metrics,
        st.session_state.last_prediction,
        st.session_state.pdf_figures
    )

    with open(pdf_path, "rb") as f:
        st.download_button(
            "📥 Download Full Analytics PDF",
            f,
            file_name="PepTastePredictor_Full_Report.pdf",
            mime="application/pdf"
        )





# ==========================================================
# SECTION 17 — FOOTER
# ==========================================================

st.markdown(
    """
<div class="footer">
© 2025 <b>PepTastePredictor</b><br>
An AI + Structural Bioinformatics platform for peptide analysis<br>
For academic, educational, and research use
</div>
""",
    unsafe_allow_html=True
)
