# 🧬 PepTastePredictor

**PepTastePredictor** is a complete end-to-end, Streamlit-based bioinformatics platform for **peptide taste prediction and structural analysis**.  
It integrates **machine learning**, **physicochemical analysis**, **3D structure generation**, **structural bioinformatics**, **batch screening**, and **automated PDF reporting** into a single interactive web application.

This project is designed for **academic, educational, and research purposes**.

---

## 🚀 Features

### 🔬 Machine Learning Predictions
- Peptide **taste prediction**
- Peptide **solubility prediction**
- **Docking score estimation** (kcal/mol)
- Random Forest–based classification and regression models

### 🧪 Physicochemical & Sequence Analysis
- Molecular weight
- Isoelectric point (pI)
- Net charge (pH 7)
- GRAVY score
- Instability index
- Secondary structure fractions
- Amino acid composition (hydrophobic, polar, charged, aromatic)

### 🧬 Structural Bioinformatics
- **3D peptide structure generation** using PeptideBuilder
- Interactive **3D visualization** using py3Dmol
- **Cα RMSD calculation**
- **Ramachandran plot analysis**
- **Cα distance heatmap**

### 📦 Batch Prediction
- Upload CSV file containing peptide sequences
- Predict taste, solubility, and docking scores for multiple peptides
- Download batch prediction results

### 📊 Model & Dataset Analytics
- Model performance metrics (Accuracy, F1, RMSE, R²)
- PCA visualization of feature space
- Confusion matrices (Taste & Solubility)
- Feature importance analysis
- Docking score performance plots

### 📄 Automated PDF Report
- One-click generation of a **comprehensive PDF report**
- Includes:
  - Model performance
  - Prediction results
  - All generated plots and analytics

---

## 🖥️ Application Modes

The application supports three analysis modes:

1. **Single Peptide Prediction**
2. **Batch Peptide Prediction**
3. **PDB Upload & Structural Analysis**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Scikit-learn**
- **Pandas & NumPy**
- **Biopython**
- **PeptideBuilder**
- **py3Dmol**
- **Matplotlib & Seaborn**
- **ReportLab**

---

## 📁 Project Structure

PepTastePredictor/
│
├── main.py              # Streamlit application (core file)
├── requirements.txt     # Python dependencies
├── AIML (4).xlsx        # Dataset used for model training
├── logo.png             # Application logo
├── README.md            # Project documentation
└── LICENSE              # License file (optional)

