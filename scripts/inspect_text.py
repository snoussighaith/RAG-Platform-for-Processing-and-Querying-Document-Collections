"""Inspection ciblée : doublons et format du texte (lecture seule)."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
train = pd.read_csv(RAW / "BBC News Train.csv")
test = pd.read_csv(RAW / "BBC News Test.csv")
sample = pd.read_csv(RAW / "BBC News Sample Solution.csv")
texte = train["Text"]

print("=" * 70)
print("1. DOUBLONS DANS TRAIN")
masque = texte.duplicated(keep=False)          # toutes les copies, originaux inclus
groupes = train[masque].groupby("Text")
print("Lignes concernées (originaux + copies) :", int(masque.sum()))
print("Groupes de textes identiques :", groupes.ngroups)
print("Taille des groupes {taille: nombre} :", groupes.size().value_counts().to_dict())
print("Groupes avec catégories DIFFÉRENTES :", int((groupes["Category"].nunique() > 1).sum()))
print("Catégories des copies en trop :", train[texte.duplicated()]["Category"].value_counts().to_dict())

print("=" * 70)
print("2. TEST vs TRAIN")
print("Lignes de TEST dont le texte est dans TRAIN :", int(test["Text"].isin(texte).sum()))
print("Doublons à l'intérieur de TEST :", int(test["Text"].duplicated().sum()))

print("=" * 70)
print("3. FORMAT DU TEXTE (sur TRAIN, 1490 textes)")
print("Textes avec au moins une majuscule :", int(texte.str.contains(r"[A-Z]").sum()))
print("Textes avec au moins un point '.' :", int(texte.str.contains(".", regex=False).sum()))
print("Textes avec au moins une apostrophe :", int(texte.str.contains("'", regex=False).sum()))
print("Textes avec au moins un symbole £ :", int(texte.str.contains("£", regex=False).sum()))
print("Textes avec au moins un retour à la ligne :", int(texte.str.contains("\n", regex=False).sum()))

print("=" * 70)
print("4. DÉBUT DE DEUX ARTICLES (700 premiers caractères)")
for i in [0, 1]:
    print(f"--- ArticleId {train.loc[i, 'ArticleId']} ({train.loc[i, 'Category']}) ---")
    print(texte.iloc[i][:700])

print("=" * 70)
print("5. LONGUEURS EXTRÊMES (en mots)")
mots = texte.str.split().str.len()
print(mots.quantile([0.01, 0.05, 0.95, 0.99]).to_string())
print("Articles de plus de 1000 mots :", int((mots > 1000).sum()))
i_max = mots.idxmax()
print("Plus long :", train.loc[i_max, "ArticleId"], train.loc[i_max, "Category"], mots[i_max], "mots")
print("Début :", texte.loc[i_max][:200])

print("=" * 70)
print("6. SAMPLE SOLUTION : répartition des catégories")
print(sample["Category"].value_counts().to_dict())