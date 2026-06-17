import os
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# === Copy these directly from your app.py ===
AA = "ACDEFGHIKLMNPQRSTVWY"
ALL_DIPEPTIDES = [a1 + a2 for a1 in AA for a2 in AA]
TASTE_CLASSES = ["Bitter", "Neutral", "Salty", "Sour", "Sweet", "Umami"]
KD_SCALE = {
    "A": 1.8,  "C": 2.5,  "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5,  "K": -3.9, "L": 3.8,
    "M": 1.9,  "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2,  "W": -0.9, "Y": -1.3,
}

def clean_sequence(seq):
    if not isinstance(seq, str):
        return ""
    lines = seq.splitlines()
    lines = [l for l in lines if not l.strip().startswith(">")]
    seq = "".join(lines)
    seq = seq.upper().replace(" ", "").replace("\n", "").replace("\t", "")
    return "".join(a for a in seq if a in AA)

def simplify_taste(taste_series):
    base_tastes = ["Bitter", "Sweet", "Salty", "Sour", "Umami", "Neutral"]
    def _map(t):
        words = [w.strip().lower() for w in str(t).split()]
        if len(words) == 1:
            for bt in base_tastes:
                if words[0] == bt.lower():
                    return bt
            return "Neutral"
        for bt in base_tastes[:-1]:
            if bt.lower() in words:
                return bt
        if "neutral" in words:
            return "Neutral"
        return "Neutral"
    return taste_series.apply(_map)

def model_features(seq):
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
    L = len(seq)
    features = {"length": L}
    if L >= 2:
        try:
            ana = ProteinAnalysis(seq)
            features.update({
                "mw": ana.molecular_weight(),
                "pI": ana.isoelectric_point(),
                "aromaticity": ana.aromaticity(),
                "instability": ana.instability_index(),
                "gravy": ana.gravy(),
                "charge": ana.charge_at_pH(7.0),
            })
        except:
            features.update({"mw":0,"pI":7.0,"aromaticity":0,
                             "instability":0,"gravy":0,"charge":0})
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

def build_feature_table(seqs):
    return pd.DataFrame([model_features(s) for s in seqs]).fillna(0)

# === Load and process dataset ===
df = pd.read_excel("AIML.xlsx")
df.columns = df.columns.str.lower().str.strip()
df["peptide"] = df["peptide"].apply(clean_sequence)
df = df[df["peptide"].str.len() >= 1].reset_index(drop=True)
df = df[
    df["taste"].notna()
    & df["solubility"].notna()
    & df["docking score (kcal/mol)"].notna()
].reset_index(drop=True)

df["solubility"] = df["solubility"].str.strip().str.rstrip(".")
df["taste"] = simplify_taste(df["taste"])
df = df[df["taste"].isin(TASTE_CLASSES)].reset_index(drop=True)

X = build_feature_table(df["peptide"])

le_taste = LabelEncoder()
le_taste.fit(TASTE_CLASSES)
y_taste = le_taste.transform(df["taste"])

idx = np.arange(len(X))
tr_idx, te_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y_taste)

Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
yt_tr, yt_te = y_taste[tr_idx], y_taste[te_idx]

# === Train model ===
taste_model = ExtraTreesClassifier(
    n_estimators=500, class_weight="balanced", random_state=42)
taste_model.fit(Xtr, yt_tr)

# === Get predictions ===
taste_preds = taste_model.predict(Xte)

# === Print everything you need ===
print("\n" + "="*50)
print("TABLE 1 — PER CLASS METRICS")
print("="*50)
report = classification_report(
    yt_te,
    taste_preds,
    target_names=le_taste.classes_,
    output_dict=True
)
df_report = pd.DataFrame(report).T
print(df_report.round(3).to_string())

print("\n" + "="*50)
print("TEST SET CLASS COUNTS (n column in Table 1)")
print("="*50)
unique, counts = np.unique(yt_te, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {le_taste.classes_[u]}: n = {c}")

print("\n" + "="*50)
print("OVERALL METRICS")
print("="*50)
from sklearn.metrics import accuracy_score, f1_score
print(f"  Accuracy: {accuracy_score(yt_te, taste_preds)*100:.1f}%")
print(f"  Weighted F1: {f1_score(yt_te, taste_preds, average='weighted'):.3f}")
