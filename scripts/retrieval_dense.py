"""Étape 11c : recherche dense (modèles x découpages), évaluée au niveau document."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="pc1")
parser.add_argument("--modeles", nargs="+", default=None)
parser.add_argument("--strategies", nargs="+",
                    default=["fixed_100_o20", "fixed_200_o40", "sent_150"])
args = parser.parse_args()

K, SEED = 5, 42
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
torch.manual_seed(SEED)
DEVICE = cfg["embeddings"].get("device", "cpu")
PROC, EVAL, RES = Path("data/processed"), Path("data/eval"), Path("results")
EMB = Path("indexes/embeddings")
RES.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)

tous = {k: v for k, v in cfg["embeddings"].items() if k.startswith("model_") and isinstance(v, dict)}
cles = args.modeles or list(tous)

# --- Questions (mêmes règles que la baseline BM25) ---
morceaux = []
for f in sorted(EVAL.glob("questions_*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    morceaux.append(d[d["question"].str.strip() != ""])
questions = pd.concat(morceaux, ignore_index=True)
evaluees = questions[questions["type"] != "hors_corpus"].copy()
evaluees["attendus"] = evaluees["expected_doc_ids"].apply(lambda s: [d for d in s.split("|") if d])
evaluees = evaluees.reset_index(drop=True)
print(f"Questions évaluées : {len(evaluees)} | avec version française : "
      f"{int((evaluees['question_fr'].str.strip() != '').sum())}")


def classement_documents(scores, doc_ids, n=10):
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
    """IC bootstrap 95 % de la moyenne (rééchantillonnage des questions, graine fixe)."""
    v = np.asarray(valeurs, dtype=float)
    rng = np.random.default_rng(SEED)
    moy = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
    return np.percentile(moy, [2.5, 97.5])


resume, francais = [], []
for cle in cles:
    m = tous[cle]
    pref_q = m.get("query_prefix", "")
    modele = SentenceTransformer(m["name"], device=DEVICE)
    print(f"\n##### {cle} : {m['name']} | préfixe de requête : {pref_q!r}")

    for strat in args.strategies:
        chunks = pd.read_csv(PROC / f"chunks_{strat}.csv")
        doc_ids = chunks["doc_id"].to_numpy()
        emb = np.load(EMB / f"{cle}__{strat}.npy")
        meta = json.loads((EMB / f"{cle}__{strat}.json").read_text(encoding="utf-8"))
        empreinte = hashlib.sha256("\n".join(chunks["chunk_id"]).encode()).hexdigest()
        assert empreinte == meta["sha256_ordre_chunk_ids"], f"vecteurs et chunks désalignés : {cle}/{strat}"
        assert emb.shape[0] == len(chunks)

        lignes, vecteurs_q = [], {}
        for q in evaluees.itertuples():
            t0 = time.perf_counter()
            qv = modele.encode([pref_q + q.question], normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)[0].astype("float32")
            t_enc = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            docs = classement_documents(emb @ qv, doc_ids)
            t_rech = (time.perf_counter() - t0) * 1000
            r, p, rr, nd = metriques(docs, q.attendus)
            lignes.append({"question_id": q.question_id, "type": q.type, "recall@5": r,
                           "precision@5": p, "mrr@10": rr, "ndcg@5": nd,
                           "lat_encodage_ms": t_enc, "lat_recherche_ms": t_rech,
                           "attendus": "|".join(q.attendus), "top5": "|".join(docs[:K])})
        df = pd.DataFrame(lignes)
        df.to_csv(RES / f"dense_{cle}_{strat}_{args.machine}.csv", index=False)

        # Comparaison appariée avec la baseline BM25 (même découpage)
        d_mrr = d_ndcg = None
        chemin_bm = RES / f"bm25_{strat}_{args.machine}.csv"
        if chemin_bm.exists():
            bm = pd.read_csv(chemin_bm).set_index("question_id")
            if set(df["question_id"]) <= set(bm.index):
                d_mrr = df["mrr@10"].to_numpy() - bm.loc[df["question_id"], "mrr@10"].to_numpy()
                d_ndcg = df["ndcg@5"].to_numpy() - bm.loc[df["question_id"], "ndcg@5"].to_numpy()

        lo_r, hi_r = ic95(df["recall@5"])
        lo_m, hi_m = ic95(df["mrr@10"])
        ligne = {
            "modele": cle, "strategie": strat, "n": len(df),
            "recall@5": round(df["recall@5"].mean(), 3), "recall_ic95": f"[{lo_r:.2f};{hi_r:.2f}]",
            "precision@5": round(df["precision@5"].mean(), 3),
            "mrr@10": round(df["mrr@10"].mean(), 3), "mrr_ic95": f"[{lo_m:.2f};{hi_m:.2f}]",
            "ndcg@5": round(df["ndcg@5"].mean(), 3),
            "lat_enc_ms": round(df["lat_encodage_ms"].median(), 1),
            "lat_rech_ms": round(df["lat_recherche_ms"].median(), 1),
            "n_echecs_recall0": int((df["recall@5"] == 0).sum()),
        }
        if d_mrr is not None:
            lo, hi = ic95(d_mrr)
            ligne["d_mrr_vs_bm25"] = f"{d_mrr.mean():+.3f} [{lo:+.2f};{hi:+.2f}]"
            lo, hi = ic95(d_ndcg)
            ligne["d_ndcg_vs_bm25"] = f"{d_ndcg.mean():+.3f} [{lo:+.2f};{hi:+.2f}]"
        resume.append(ligne)

        # Questions en français vs anglais (mêmes questions)
        sous = evaluees[evaluees["question_fr"].str.strip() != ""]
        if len(sous):
            rr_fr, rec_fr = [], []
            for q in sous.itertuples():
                qv = modele.encode([pref_q + q.question_fr], normalize_embeddings=True,
                                   show_progress_bar=False, convert_to_numpy=True)[0].astype("float32")
                docs = classement_documents(emb @ qv, doc_ids)
                r, _, rr, _ = metriques(docs, q.attendus)
                rr_fr.append(rr)
                rec_fr.append(r)
            en = df[df["question_id"].isin(sous["question_id"])]
            francais.append({"modele": cle, "strategie": strat, "n_questions_fr": len(sous),
                             "mrr_en": round(en["mrr@10"].mean(), 3), "mrr_fr": round(float(np.mean(rr_fr)), 3),
                             "recall_en": round(en["recall@5"].mean(), 3), "recall_fr": round(float(np.mean(rec_fr)), 3)})
        print(f"  {strat:14s} | Recall@5 {ligne['recall@5']} | MRR@10 {ligne['mrr@10']} | "
              f"nDCG@5 {ligne['ndcg@5']} | échecs {ligne['n_echecs_recall0']}")

tableau = pd.DataFrame(resume)
tableau.to_csv(RES / f"dense_resume_{args.machine}.csv", index=False)
print("\n=== RÉSUMÉ RECHERCHE DENSE (niveau document) ===")
print(tableau.to_string(index=False))
if francais:
    tf = pd.DataFrame(francais)
    tf.to_csv(RES / f"dense_francais_{args.machine}.csv", index=False)
    print("\n=== MÊMES QUESTIONS : ANGLAIS vs FRANÇAIS (échantillon très petit : indicatif) ===")
    print(tf.to_string(index=False))
print(f"\nFichiers écrits dans {RES}/ (machine = {args.machine})")