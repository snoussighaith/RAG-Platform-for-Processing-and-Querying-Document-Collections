"""Compte les articles contenant TOUS les mots donnés.
Usage : python scripts/chercher_mots.py mot1 mot2 ..."""
import re
import sys

import pandas as pd

mots = re.findall(r"[a-z0-9£$]+", " ".join(sys.argv[1:]).lower())
if not mots:
    sys.exit("Donne au moins un mot.")

corpus = pd.read_csv("data/processed/corpus.csv")


def contient(texte):
    jetons = set(re.findall(r"[a-z0-9£$]+", texte))
    return all(m in jetons for m in mots)


trouves = corpus[corpus["text"].apply(contient)]
print(f"Mots recherchés : {mots}")
print(f"Articles contenant tous ces mots : {len(trouves)}")
for r in trouves.head(15).itertuples():
    print(f"  {r.doc_id} | {r.category} | {r.n_words} mots")