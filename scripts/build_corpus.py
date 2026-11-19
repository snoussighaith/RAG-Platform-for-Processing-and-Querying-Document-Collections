"""Construit le corpus nettoyé : TRAIN sans doublons exacts, avec traçabilité."""
import hashlib
import json
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
FICHIER_TRAIN = RAW / "BBC News Train.csv"

# Empreinte du fichier source : prouve que deux machines partent des mêmes données
empreinte = hashlib.sha256(FICHIER_TRAIN.read_bytes()).hexdigest()

train = pd.read_csv(FICHIER_TRAIN)
n_initial = len(train)

# Tri déterministe : le résultat ne dépend pas de l'ordre du fichier
train = train.sort_values("ArticleId").reset_index(drop=True)

# Nettoyage minimal : espaces multiples réduits à un seul, espaces de bord retirés
train["text"] = train["Text"].str.replace(r"\s+", " ", regex=True).str.strip()

# Doublons exacts : on garde la première occurrence (plus petit ArticleId)
garde = train.drop_duplicates(subset="text", keep="first")
retire = train[train.duplicated(subset="text", keep="first")].copy()

# Journal des suppressions : quelle copie a été retirée, et quel original est conservé
original = garde.set_index("text")["ArticleId"]
retire["kept_article_id"] = retire["text"].map(original)
journal = retire[["ArticleId", "kept_article_id", "Category"]].rename(
    columns={"ArticleId": "removed_article_id", "Category": "category"}
)
journal.to_csv(OUT / "duplicates_removed.csv", index=False)

# Corpus final
corpus = garde.copy()
corpus["doc_id"] = "bbc_" + corpus["ArticleId"].astype(str)
corpus["n_words"] = corpus["text"].str.split().str.len()
corpus = corpus.rename(columns={"ArticleId": "article_id", "Category": "category"})
corpus = corpus[["doc_id", "article_id", "category", "n_words", "text"]]
corpus.to_csv(OUT / "corpus.csv", index=False, encoding="utf-8")

# Statistiques pour la datasheet
stats = {
    "source": "Kaggle competition learn-ai-bbc, fichier BBC News Train.csv",
    "sha256_source": empreinte,
    "n_initial": int(n_initial),
    "n_doublons_retires": int(len(retire)),
    "n_final": int(len(corpus)),
    "categories": {k: int(v) for k, v in corpus["category"].value_counts().items()},
    "mots_par_article": {k: float(v) for k, v in corpus["n_words"].describe().round(1).items()},
}
(OUT / "corpus_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

# Vérifications automatiques
relu = pd.read_csv(OUT / "corpus.csv")
assert relu["doc_id"].is_unique, "doc_id en double"
assert relu.isna().sum().sum() == 0, "valeurs manquantes dans le corpus"
assert len(relu) == n_initial - len(retire), "incohérence du nombre de lignes"

print("Articles initiaux :", n_initial)
print("Doublons retirés  :", len(retire), "(attendu : 50)")
print("Articles finaux   :", len(relu))
print("Catégories        :", stats["categories"])
print("SHA-256 de TRAIN  :", empreinte)
print("Fichiers écrits   :", sorted(p.name for p in OUT.iterdir()))
print("Vérifications     : OK")