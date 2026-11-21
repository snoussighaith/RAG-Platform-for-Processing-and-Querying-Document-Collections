"""Baseline BM25 : récupération lexicale seule, évaluée au niveau document."""
import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="pc1", help="nom de la machine (résultats séparés)")
args = parser.parse_args()

K = 5          # nombre d'articles considérés pour Recall/Precision/nDCG
SEED = 42
PROC, EVAL, RES = Path("data/processed"), Path("data/eval"), Path("results")
RES.mkdir(exist_ok=True)
STRATEGIES = ["fixed_100_o20", "fixed_200_o40", "sent_150"]


def jetons(texte):
    """Tokenisation : minuscules ; mots = lettres, chiffres, £, $ (même règle pour corpus et questions)."""
    return re.findall(r"[a-z0-9£$]+", texte.lower())


# --- Questions : on lit tous les fichiers questions_*.csv présents (a, b...) ---
morceaux = []
for f in sorted(EVAL.glob("questions_*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    vides = int((d["question"].str.strip() == "").sum())
    if vides:
        print(f"AVERTISSEMENT : {f.name} contient {vides} question(s) vide(s), ignorées")
    morceaux.append(d[d["question"].str.strip() != ""])
questions = pd.concat(morceaux, ignore_index=True)
evaluees = questions[questions["type"] != "hors_corpus"].copy()
evaluees["attendus"] = evaluees["expected_doc_ids"].apply(lambda s: [d for d in s.split("|") if d])
print(f"Fichiers de questions : {[f.name for f in sorted(EVAL.glob('questions_*.csv'))]}")
print(f"Questions totales : {len(questions)} | évaluées en récupération : {len(evaluees)} "
      f"(hors_corpus exclues : {len(questions) - len(evaluees)})")


def classement_documents(scores, doc_ids, n=10):
    """Ordre des articles : on parcourt les chunks par score décroissant, 1re apparition de chaque article."""
    ordre = np.argsort(-scores, kind="stable")
    vus, docs = set(), []
    for i in ordre:
        d = doc_ids[i]
        if d not in vus:
            vus.add(d)
            docs.append(d)
            if len(docs) >= n:
                break
    return docs


def metriques(docs, attendus):
    top = docs[:K]
    hits = [d in attendus for d in top]
    recall = sum(hits) / len(attendus)
    precision = sum(hits) / K
    rr = next((1 / i for i, d in enumerate(docs[:10], 1) if d in attendus), 0.0)
    dcg = sum(1 / np.log2(i + 1) for i, h in enumerate(hits, 1) if h)
    idcg = sum(1 / np.log2(i + 1) for i in range(1, min(len(attendus), K) + 1))
    return recall, precision, rr, dcg / idcg


def ic95(valeurs):
    """Intervalle de confiance bootstrap à 95 % de la moyenne (rééchantillonnage des questions)."""
    v = np.asarray(valeurs, dtype=float)
    rng = np.random.default_rng(SEED)
    moyennes = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
    return np.percentile(moyennes, [2.5, 97.5])


resume = []
for nom in STRATEGIES:
    chunks = pd.read_csv(PROC / f"chunks_{nom}.csv")
    doc_ids = chunks["doc_id"].to_numpy()
    t0 = time.perf_counter()
    bm25 = BM25Okapi([jetons(t) for t in chunks["text"]])
    t_index = time.perf_counter() - t0

    lignes = []
    for q in evaluees.itertuples():
        t0 = time.perf_counter()
        scores = bm25.get_scores(jetons(q.question))
        docs = classement_documents(scores, doc_ids)
        latence = (time.perf_counter() - t0) * 1000
        r, p, rr, nd = metriques(docs, q.attendus)
        lignes.append({
            "question_id": q.question_id, "type": q.type, "recall@5": r, "precision@5": p,
            "mrr@10": rr, "ndcg@5": nd, "latence_ms": latence,
            "attendus": "|".join(q.attendus), "top5": "|".join(docs[:K]),
        })
    df = pd.DataFrame(lignes)
    df.to_csv(RES / f"bm25_{nom}_{args.machine}.csv", index=False)

    lo_r, hi_r = ic95(df["recall@5"])
    lo_m, hi_m = ic95(df["mrr@10"])
    resume.append({
        "strategie": nom, "n_questions": len(df),
        "recall@5": round(df["recall@5"].mean(), 3), "recall_ic95": f"[{lo_r:.2f} ; {hi_r:.2f}]",
        "precision@5": round(df["precision@5"].mean(), 3),
        "mrr@10": round(df["mrr@10"].mean(), 3), "mrr_ic95": f"[{lo_m:.2f} ; {hi_m:.2f}]",
        "ndcg@5": round(df["ndcg@5"].mean(), 3),
        "latence_mediane_ms": round(df["latence_ms"].median(), 1),
        "construction_index_s": round(t_index, 2),
    })

    print(f"\n=== {nom} : Recall@5 par type de question ===")
    print(df.groupby("type")["recall@5"].agg(["mean", "count"]).round(3).to_string())
    echecs = df[df["recall@5"] == 0]
    print(f"Questions avec Recall@5 = 0 : {len(echecs)} sur {len(df)}")
    for e in echecs.head(5).itertuples():
        texte = evaluees.loc[evaluees["question_id"] == e.question_id, "question"].iloc[0]
        print(f"  {e.question_id} | attendus={e.attendus} | trouvés={e.top5} | {texte[:90]}")

tableau = pd.DataFrame(resume)
tableau.to_csv(RES / f"bm25_resume_{args.machine}.csv", index=False)
print("\n=== RÉSUMÉ BASELINE BM25 (niveau document) ===")
print(tableau.to_string(index=False))
print(f"\nFichiers écrits dans {RES}/ (machine = {args.machine})")