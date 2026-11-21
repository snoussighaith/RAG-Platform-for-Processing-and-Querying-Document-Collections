"""Valide le fichier de questions d'un auteur (structure, identifiants, copie du texte)."""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--auteur", required=True, choices=["a", "b"])
args = parser.parse_args()

chemin = Path("data/eval") / f"questions_{args.auteur}.csv"
df = pd.read_csv(chemin, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
corpus = pd.read_csv("data/processed/corpus.csv").set_index("doc_id")
erreurs, avertissements = [], []

colonnes = ["question_id", "auteur", "type", "question", "question_fr",
            "expected_doc_ids", "reference_answer", "notes"]
manque = [c for c in colonnes if c not in df.columns]
if manque:
    sys.exit(f"ERREUR : colonnes manquantes {manque}")

if len(df) != 25:
    erreurs.append(f"{len(df)} lignes au lieu de 25")
if not df["question_id"].is_unique:
    erreurs.append("question_id en double")
attendu = {"factuelle": 15, "multi_doc": 4, "ambigue": 2, "hors_corpus": 4}
compte = df["type"].value_counts().to_dict()
for t, n in attendu.items():
    if compte.get(t, 0) != n:
        erreurs.append(f"type {t} : {compte.get(t, 0)} au lieu de {n}")


def jetons(texte):
    return re.findall(r"[a-z0-9£$]+", texte.lower())


def ngrammes(j, n):
    return {" ".join(j[i:i + n]) for i in range(len(j) - n + 1)}


for r in df.itertuples():
    qid = r.question_id
    docs = [d for d in r.expected_doc_ids.split("|") if d]
    if not r.question.strip():
        erreurs.append(f"{qid} : question vide")
    for d in docs:
        if d not in corpus.index:
            erreurs.append(f"{qid} : doc_id inconnu {d}")
    if r.type in ("factuelle", "ambigue") and len(docs) < 1:
        erreurs.append(f"{qid} : aucun document attendu")
    if r.type == "multi_doc" and len(docs) < 2:
        erreurs.append(f"{qid} : multi_doc exige au moins 2 documents")
    if r.type == "hors_corpus":
        if docs:
            erreurs.append(f"{qid} : hors_corpus ne doit avoir aucun document attendu")
        if r.reference_answer.strip() != "NOT_IN_CORPUS":
            erreurs.append(f"{qid} : reference_answer doit valoir NOT_IN_CORPUS")
    elif not r.reference_answer.strip():
        erreurs.append(f"{qid} : réponse de référence vide")
    # Copie trop fidèle du texte (6 mots consécutifs identiques)
    gq = ngrammes(jetons(r.question), 6)
    for d in docs:
        if d in corpus.index and gq & ngrammes(jetons(corpus.loc[d, "text"]), 6):
            avertissements.append(f"{qid} : 6 mots consécutifs identiques au texte de {d} (reformule)")

nb_fr = int(((df["type"] == "factuelle") & (df["question_fr"].str.strip() != "")).sum())
if nb_fr < 3:
    avertissements.append(f"question_fr renseignée pour {nb_fr} factuelles (minimum conseillé : 3)")

print(f"Fichier : {chemin}")
print(f"Types : {compte}")
print(f"Versions françaises : {nb_fr}")
for e in erreurs:
    print("ERREUR       :", e)
for a in avertissements:
    print("AVERTISSEMENT:", a)
print("RÉSULTAT :", "OK" if not erreurs else f"{len(erreurs)} erreur(s) à corriger")
sys.exit(1 if erreurs else 0)