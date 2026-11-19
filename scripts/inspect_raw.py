"""Inspection brute des fichiers de la compétition learn-ai-bbc (lecture seule)."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
FICHIERS = {
    "TRAIN": "BBC News Train.csv",
    "TEST": "BBC News Test.csv",
    "SAMPLE": "BBC News Sample Solution.csv",
}
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

donnees = {}
for nom, fichier in FICHIERS.items():
    df = pd.read_csv(RAW / fichier)
    donnees[nom] = df
    print("=" * 70)
    print(f"{nom} : {fichier}")
    print("Forme (lignes, colonnes) :", df.shape)
    print("Colonnes et types :")
    print(df.dtypes.to_string())
    print("Valeurs manquantes par colonne :")
    print(df.isna().sum().to_string())
    # Aperçu : on tronque les textes longs pour garder une sortie lisible
    apercu = df.head(3).copy()
    for col in apercu.select_dtypes(include="object").columns:
        apercu[col] = apercu[col].astype(str).str.slice(0, 80)
    print("Aperçu des 3 premières lignes :")
    print(apercu.to_string())

# Analyses détaillées du fichier d'entraînement
train = donnees["TRAIN"]
print("=" * 70)
print("ANALYSE DÉTAILLÉE DE TRAIN")
if "Category" in train.columns:
    print("Répartition des catégories :")
    print(train["Category"].value_counts().to_string())
if "Text" in train.columns:
    mots = train["Text"].astype(str).str.split().str.len()
    print("Longueur des textes (en mots) :")
    print(mots.describe().round(1).to_string())
    print("Textes vides :", int((train["Text"].astype(str).str.strip() == "").sum()))
    print("Textes en double :", int(train["Text"].duplicated().sum()))
if "ArticleId" in train.columns:
    print("ArticleId en double :", int(train["ArticleId"].duplicated().sum()))

# Comparaison avec TEST
test = donnees["TEST"]
print("=" * 70)
print("COMPARAISON TRAIN / TEST")
print("Colonnes communes :", sorted(set(train.columns) & set(test.columns)))
print("Colonnes seulement dans TRAIN :", sorted(set(train.columns) - set(test.columns)))
if "ArticleId" in train.columns and "ArticleId" in test.columns:
    print("ArticleId communs à TRAIN et TEST :", len(set(train["ArticleId"]) & set(test["ArticleId"])))
if "Text" in train.columns and "Text" in test.columns:
    print("Textes de TEST aussi présents dans TRAIN :", int(test["Text"].isin(train["Text"]).sum()))