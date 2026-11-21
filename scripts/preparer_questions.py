"""Prépare le gabarit de questions et la feuille de travail d'un auteur (a ou b)."""
import argparse
import random
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--auteur", required=True, choices=["a", "b"])
parser.add_argument("--force", action="store_true", help="écrase le gabarit existant")
args = parser.parse_args()
auteur = args.auteur
prefixe = auteur.upper()

PROC = Path("data/processed")
EVAL = Path("data/eval")
EVAL.mkdir(parents=True, exist_ok=True)
corpus = pd.read_csv(PROC / "corpus.csv").set_index("doc_id")

# Tirage reproductible : même graine chez les deux auteurs, tranches différentes
# (a prend les positions 0 à 4, b les positions 5 à 9) -> ensembles disjoints
rng = random.Random(42)
categories = sorted(corpus["category"].unique())
choisis = {}
for cat in categories:
    ids = sorted(corpus.index[corpus["category"] == cat])
    rng.shuffle(ids)
    choisis[cat] = ids[0:5] if auteur == "a" else ids[5:10]

lignes = []


def ajouter(type_question, docs, reponse=""):
    """Ajoute une ligne vide à remplir, avec les articles suggérés."""
    lignes.append({
        "question_id": f"{prefixe}{len(lignes) + 1:02d}",
        "auteur": auteur,
        "type": type_question,
        "question": "",
        "question_fr": "",
        "expected_doc_ids": "|".join(docs),
        "reference_answer": reponse,
        "notes": "",
    })


for cat in categories:                       # 15 factuelles : 3 par catégorie
    for d in choisis[cat][:3]:
        ajouter("factuelle", [d])
for cat in categories[:4]:                   # 4 multi-documents : 1 paire par catégorie
    ajouter("multi_doc", choisis[cat][3:5])
for d in choisis[categories[4]][3:5]:        # 2 ambiguës
    ajouter("ambigue", [d])
for _ in range(4):                           # 4 hors corpus (aucun article)
    ajouter("hors_corpus", [], "NOT_IN_CORPUS")

# Gabarit CSV : séparateur ';' (s'ouvre bien dans Excel FR), UTF-8 avec BOM
chemin = EVAL / f"questions_{auteur}.csv"
if chemin.exists() and not args.force:
    print(f"ATTENTION : {chemin} existe déjà : NON écrasé (ton travail est protégé).")
else:
    pd.DataFrame(lignes).to_csv(chemin, sep=";", index=False, encoding="utf-8-sig")
    print(f"Gabarit écrit : {chemin} ({len(lignes)} lignes)")

# Feuille de travail : textes des articles à lire (dans data/processed : non versionnée)
sortie = PROC / f"feuille_travail_{auteur}.txt"
distincts = set()
with open(sortie, "w", encoding="utf-8") as f:
    for l in lignes:
        f.write("=" * 80 + "\n")
        f.write(f"{l['question_id']} | type : {l['type']}\n")
        docs = [d for d in l["expected_doc_ids"].split("|") if d]
        for d in docs:
            distincts.add(d)
            r = corpus.loc[d]
            f.write(f"--- {d} | {r['category']} | {r['n_words']} mots ---\n{r['text']}\n\n")
        if not docs:
            f.write("(aucun article : invente une question hors corpus)\n\n")
print(f"Feuille de travail : {sortie}")
print(f"Articles distincts à lire : {len(distincts)}")