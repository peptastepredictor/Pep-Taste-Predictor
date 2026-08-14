# 🧬 PepTastePredictor

**PepTastePredictor** is an end-to-end Streamlit platform for **multi-label peptide taste prediction**, **solubility and docking-score estimation**, and **structural bioinformatics** — with built-in **SHAP interpretability** and **automated PDF reporting**.

Built for academic, iGEM, and research use.

---

## 🚀 Features

### 🔬 Machine Learning Predictions
- **Multi-label taste prediction** — Bitter, Salty, Sour, Sweet, Umami predicted independently (`MultiOutputClassifier` wrapping one `ExtraTreesClassifier` per taste, 500 estimators each). A single peptide can carry multiple simultaneous tastes, matching real composite labels like "Sour Sweet Umami".
- **Solubility classification** (`ExtraTreesClassifier`, 300 estimators, class-balanced).
- **Docking score regression** in kcal/mol-scale units (`RandomForestRegressor`, 400 estimators).
- Trained on a curated, deduplicated peptide dataset (`AIML.xlsx`), with an 80/20 stratified train/test split (`random_state=42`).
- **5-fold stratified cross-validation** reported alongside the single train/test split, as a robustness check on model stability.

### 🧠 Explainability
- **SHAP (TreeExplainer)** interpretability per predicted taste — bar charts showing which sequence features pushed the model toward or away from each call, with auto-generated plain-language "what this shows / what it means" captions.
- **Feature importance** ranking averaged across all 5 taste classifiers.

### 🧪 Physicochemical & Sequence Analysis
- Molecular weight, isoelectric point, net charge (pH 7), aromaticity, GRAVY score, instability index
- Chou–Fasman secondary structure fractions (helix / sheet / coil) — computed for display, kept separate from the 432-dim model feature vector
- Amino acid composition (hydrophobic, polar, charged, aromatic, tiny groups)
- **432-dimensional feature vector**: 7 physicochemical + 20 amino-acid composition + 400 dipeptide composition (20×20) + 5 biochemical group ratios

### 🧬 Structural Bioinformatics — Hybrid Structural Engine v2
A four-tier fallback cascade generates a 3D structure for any peptide, prioritizing real/experimental data over generated conformations:
1. **RCSB PDB** — sequence search against experimentally deposited structures
2. **Remote ESMFold** (ESM Atlas API, Meta AI) — AI-predicted structure
3. **Chou-Fasman Peptide Folding Engine** — in-house backbone builder using predicted secondary structure and torsion angles
4. **PeptideBuilder** — geometric fallback (idealized backbone / linear chain)

Structural analysis includes:
- Interactive **3D visualization** (py3Dmol), with genuine pLDDT-based coloring when available
- **Cα RMSD** calculation
- **Ramachandran plot** (φ/ψ angles) with dynamic fold-composition captions
- **Cα distance heatmap** with compactness/fold-type interpretation
- **Per-residue pLDDT confidence profile** — clearly distinguishes genuine ESMFold/RCSB pLDDT from the in-house Chou-Fasman pseudo-confidence score
- **Secondary structure composition plot** (per-residue helix/sheet/coil call)
- **PDB upload mode** — analyze any external PDB (AlphaFold, ESMFold, experimental, or platform-generated)

### 📦 Batch Prediction
- Upload **CSV** (flexible `peptide`/`Peptide`/`PEPTIDES` column matching) or **FASTA**
- Predicts taste, solubility, docking score, fold type, and full physicochemical profile for every sequence
- Optional **3D structure generation for the whole batch**, downloadable as a **ZIP** of PDB files
- Download all batch predictions as CSV

### 📊 Model & Dataset Analytics
- Per-taste accuracy / precision / recall / F1 bar charts
- Per-taste confusion matrices (5-panel)
- Cross-validation mean ± std per taste, solubility, and docking R²
- PCA projection of the 432-dim feature space, colored by taste
- Feature importance (top contributors across all taste classifiers)
- Docking score true-vs-predicted scatter (R², RMSE)
- Every chart ships with an auto-generated, data-driven "Explanation & Inference" caption (not static boilerplate — values are computed from the actual result each time)

### 📄 Automated PDF Report
One-click PDF including:
- Model performance metrics + plain-language explanation of each metric
- 5-fold cross-validation table
- Single-prediction results (with per-field explanations)
- Every generated plot, each with its dynamic Explanation & Inference caption

### 🎨 Interface
- Light/dark mode-aware plotting (matplotlib theme adapts to Streamlit theme)
- Custom CSS styling with taste-specific badges and structure-engine badges

---

## 🖥️ Application Modes
1. **Single Peptide Prediction** — FASTA upload or paste, full ML + SHAP + structural pipeline
2. **Batch Peptide Prediction** — CSV/FASTA, optional structure generation
3. **PDB Upload & Structural Analysis** — bring your own PDB, run the structural pipeline only

---

## 🛠️ Tech Stack
- **Python**, **Streamlit**
- **Scikit-learn** (ExtraTreesClassifier, MultiOutputClassifier, RandomForestRegressor)
- **SHAP** (TreeExplainer)
- **Biopython**, **PeptideBuilder**
- **py3Dmol**
- **Matplotlib**, **Seaborn**
- **ReportLab**
- RCSB PDB Search API + ESM Atlas (ESMFold) remote API

---

## 📁 Project Structure
```
PepTastePredictor/
│
├── app.py               # Streamlit application (core file)
├── requirements.txt     # Python dependencies
├── AIML.xlsx             # Dataset used for model training
├── logo.png              # Application logo
├── README.md             # Project documentation
└── LICENSE               # License file (optional)
```

---

## ⚠️ Notes for Reproducibility
- The Chou-Fasman-derived pLDDT-style B-factors are **synthetic proxies**, not calibrated confidence scores — only ESMFold/RCSB/uploaded structures carry genuine pLDDT.
- Docking scores are model-predicted binding-energy estimates, not physically simulated docking results.
- For academic and research use only.
