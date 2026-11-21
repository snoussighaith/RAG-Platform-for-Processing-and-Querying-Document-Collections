"""Étape 12 : recherche hybride (BM25 + dense, fusion RRF) et reranking cross-encoder.
Six variantes évaluées sur les mêmes questions, au niveau document."""
import argparse
import hashlib
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="pc1")
parser.add_argument("--modeles", nargs="+", default=["model_b", "model_d_optionnel"])
parser.add_argument("--strategies", nargs="+", default=["fixed_100_o20", "fixed_200_o40", "sent_150"])
args = parser.parse_args()

# --- Paramètres FIXÉS A PRIORI : ne pas les régler d'après les résultats ---
K = 5            # top-k pour Recall / Precision / nDCG
K_RRF = 60       # constante de la fusion RRF
N_FUSION = 50    # candidats retenus dans chaque classement avant fusion
N_RERANK = 20    # nombre de chunks soumis au cross-encoder
SEED = 42
VARIANTES = ["bm25", "dense", "rrf", "bm25_rerank", "dense_rerank", "rrf_rerank"]
PRIMAIRE = ("model_b", "sent_150")

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
torch.manual_seed(SEED)
DEVICE = cfg["embeddings"].get("device", "cpu")
PROC, EVAL, RES, EMB = Path("data/processed"), Path("data/eval"), Path("results"), Path("indexes/embeddings")
RES.mkdir(exist_ok=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
print(f"Paramètres : k_RRF={K_RRF}, candidats par classement={N_FUSION}, chunks reranké={N_RERANK}, seed={SEED}")

tous = {k: v for k, v in cfg["embeddings"].items() if k.startswith("model_") and isinstance(v, dict)}
inconnus = [c for c in args.modeles if c not in tous]
if inconnus:
    raise SystemExit(f"Clés inconnues : {inconnus}. Disponibles : {list(tous)}")

# --- Questions (mêmes règles que BM25 et dense) ---
morceaux = []
for f in sorted(EVAL.glob("questions_*.csv")):
    d = pd.read_csv(f, sep=";", encoding="utf-8-sig", dtype=str).fillna("")
    morceaux.append(d[d["question"].str.strip() != ""])
questions = pd.concat(morceaux, ignore_index=True)
evaluees = questions[questions["type"] != "hors_corpus"].copy()
evaluees["attendus"] = evaluees["expected_doc_ids"].apply(lambda s: [d for d in s.split("|") if d])
evaluees = evaluees.reset_index(drop=True)
sous_fr = evaluees[evaluees["question_fr"].str.strip() != ""]
print(f"Questions évaluées : {len(evaluees)} | avec version française : {len(sous_fr)}")

ce = CrossEncoder(cfg["reranker"]["model"], device=DEVICE)
print(f"Reranker : {cfg['reranker']['model']} | longueur max : {getattr(ce, 'max_length', 'inconnue')}")


def jetons(texte):
    return re.findall(r"[a-z0-9£$]+", texte.lower())


def aggreger(ordre, doc_ids, n=10):
    """Classement de chunks -> classement d'articles (1re apparition de chaque article)."""
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
    v = np.asarray(valeurs, dtype=float)
    rng = np.random.default_rng(SEED)
    moy = v[rng.integers(0, len(v), (10000, len(v)))].mean(axis=1)
    return np.percentile(moy, [2.5, 97.5])


def rrf(ordres):
    """Fusion RRF : score(chunk) = somme sur les classements de 1 / (K_RRF + rang)."""
    cumul = {}
    for ordre in ordres:
        for rang, i in enumerate(ordre[:N_FUSION], 1):
            cumul[int(i)] = cumul.get(int(i), 0.0) + 1.0 / (K_RRF + rang)
    return [i for i, _ in sorted(cumul.items(), key=lambda x: (-x[1], x[0]))]


def reordonner(ordre, textes, question):
    """Le cross-encoder reclasse les N_RERANK premiers chunks ; le reste garde son ordre."""
    tete = [int(i) for i in ordre[:N_RERANK]]
    scores = np.asarray(ce.predict([(question, textes[i]) for i in tete],
                                   batch_size=32, show_progress_bar=False), dtype=float)
    nouveau = [i for _, i in sorted(zip(-scores, tete))]
    return nouveau + [int(i) for i in ordre[N_RERANK:]]


def executer(question, ctx):
    """Calcule les six classements de chunks et leur latence (ms) pour une question."""
    t0 = time.perf_counter()
    o_bm = np.argsort(-ctx["bm25"].get_scores(jetons(question)), kind="stable")
    t_bm = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    qv = ctx["modele"].encode([ctx["pref_q"] + question], normalize_embeddings=True,
                              show_progress_bar=False, convert_to_numpy=True)[0].astype("float32")
    o_de = np.argsort(-(ctx["emb"] @ qv), kind="stable")
    t_de = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    o_rrf = rrf([o_bm, o_de])
    t_rrf = (time.perf_counter() - t0) * 1000

    sortie = {"bm25": (o_bm, t_bm), "dense": (o_de, t_de), "rrf": (o_rrf, t_bm + t_de + t_rrf)}
    for nom, ordre, base in [("bm25_rerank", o_bm, t_bm), ("dense_rerank", o_de, t_de),
                             ("rrf_rerank", o_rrf, t_bm + t_de + t_rrf)]:
        t0 = time.perf_counter()
        o = reordonner(ordre, ctx["textes"], question)
        sortie[nom] = (o, base + (time.perf_counter() - t0) * 1000)
    return sortie


lignes, lignes_fr = [], []
for cle in args.modeles:
    m = tous[cle]
    modele = SentenceTransformer(m["name"], device=DEVICE)
    print(f"\n##### {cle} : {m['name']} | préfixe de requête : {m.get('query_prefix', '')!r}")
    for strat in args.strategies:
        t0 = time.perf_counter()
        chunks = pd.read_csv(PROC / f"chunks_{strat}.csv")
        emb = np.load(EMB / f"{cle}__{strat}.npy")
        meta = json.loads((EMB / f"{cle}__{strat}.json").read_text(encoding="utf-8"))
        empreinte = hashlib.sha256("\n".join(chunks["chunk_id"]).encode()).hexdigest()
        assert empreinte == meta["sha256_ordre_chunk_ids"], f"vecteurs et chunks désalignés : {cle}/{strat}"
        ctx = {"bm25": BM25Okapi([jetons(t) for t in chunks["text"]]),
               "doc_ids": chunks["doc_id"].to_numpy(), "textes": chunks["text"].tolist(),
               "emb": emb, "modele": modele, "pref_q": m.get("query_prefix", "")}

        for q in evaluees.itertuples():
            for var, (ordre, lat) in executer(q.question, ctx).items():
                r, p, rr, nd = metriques(aggreger(ordre, ctx["doc_ids"]), q.attendus)
                lignes.append({"modele": cle, "strategie": strat, "variante": var,
                               "question_id": q.question_id, "type": q.type, "recall@5": r,
                               "precision@5": p, "mrr@10": rr, "ndcg@5": nd, "latence_ms": lat})
        for q in sous_fr.itertuples():
            res_fr = executer(q.question_fr, ctx)
            for var in ["dense", "rrf", "rrf_rerank"]:
                r, _, rr, _ = metriques(aggreger(res_fr[var][0], ctx["doc_ids"]), q.attendus)
                lignes_fr.append({"modele": cle, "strategie": strat, "variante": var,
                                  "question_id": q.question_id, "mrr_fr": rr, "recall_fr": r})

        # Vérification : BM25 recalculé = baseline de l'étape 10
        ref = RES / f"bm25_{strat}_{args.machine}.csv"
        if ref.exists():
            mon = pd.DataFrame(lignes)
            mon = mon[(mon["modele"] == cle) & (mon["strategie"] == strat) & (mon["variante"] == "bm25")]
            mon = mon.set_index("question_id")["mrr@10"]
            etape10 = pd.read_csv(ref).set_index("question_id")["mrr@10"]
            assert np.allclose(etape10.loc[mon.index], mon, atol=1e-9), "BM25 différent de l'étape 10 !"
            verdict = "BM25 identique à l'étape 10 : OK"
        else:
            verdict = "(pas de fichier BM25 de l'étape 10 pour comparer)"
        print(f"  {strat:14s} terminé en {time.perf_counter() - t0:.0f} s | {verdict}")

dfq = pd.DataFrame(lignes)
dfq.to_csv(RES / f"hybride_questions_{args.machine}.csv", index=False)

# --- Résumé par variante ---
resume = []
for (cle, strat, var), g in dfq.groupby(["modele", "strategie", "variante"], sort=False):
    lo, hi = ic95(g["mrr@10"])
    resume.append({"modele": cle, "strategie": strat, "variante": var, "n": len(g),
                   "recall@5": round(g["recall@5"].mean(), 3), "precision@5": round(g["precision@5"].mean(), 3),
                   "mrr@10": round(g["mrr@10"].mean(), 3), "mrr_ic95": f"[{lo:.2f};{hi:.2f}]",
                   "ndcg@5": round(g["ndcg@5"].mean(), 3), "lat_mediane_ms": round(g["latence_ms"].median(), 1)})
tableau = pd.DataFrame(resume)
tableau.to_csv(RES / f"hybride_resume_{args.machine}.csv", index=False)
print("\n=== RÉSUMÉ PAR VARIANTE (niveau document) ===")
print(tableau.to_string(index=False))

# --- Comparaisons appariées (même questions) ---
COMPARAISONS = [("rrf", "bm25"), ("rrf", "dense"), ("rrf_rerank", "rrf"),
                ("rrf_rerank", "bm25"), ("dense_rerank", "dense"), ("bm25_rerank", "bm25")]
comp = []
for (cle, strat), g in dfq.groupby(["modele", "strategie"], sort=False):
    mrr = g.pivot(index="question_id", columns="variante", values="mrr@10")
    ndcg = g.pivot(index="question_id", columns="variante", values="ndcg@5")
    for a, b in COMPARAISONS:
        dm, dn = (mrr[a] - mrr[b]).to_numpy(), (ndcg[a] - ndcg[b]).to_numpy()
        lm, hm = ic95(dm)
        ln, hn = ic95(dn)
        comp.append({"modele": cle, "strategie": strat, "comparaison": f"{a} - {b}",
                     "d_mrr": round(dm.mean(), 3), "d_mrr_ic95": f"[{lm:+.2f};{hm:+.2f}]",
                     "d_ndcg": round(dn.mean(), 3), "d_ndcg_ic95": f"[{ln:+.2f};{hn:+.2f}]",
                     "mrr_ic_contient_0": bool(lm <= 0 <= hm), "ndcg_ic_contient_0": bool(ln <= 0 <= hn)})
comp = pd.DataFrame(comp)
comp.to_csv(RES / f"hybride_comparaisons_{args.machine}.csv", index=False)
cles_aff = ["rrf - bm25", "rrf_rerank - rrf", "rrf_rerank - bm25"]
print("\n=== COMPARAISONS APPARIÉES CLÉS (différence moyenne [IC 95 %] ; si l'IC contient 0 : non démontré) ===")
print(comp[comp["comparaison"].isin(cles_aff)].drop(columns=["mrr_ic_contient_0", "ndcg_ic_contient_0"]).to_string(index=False))

prim = comp[(comp["modele"] == PRIMAIRE[0]) & (comp["strategie"] == PRIMAIRE[1])
            & (comp["comparaison"] == "rrf_rerank - bm25")]
if len(prim):
    p = prim.iloc[0]
    print(f"\n*** COMPARAISON PRIMAIRE ({PRIMAIRE[0]} / {PRIMAIRE[1]}) : rrf_rerank - bm25 ***")
    print(f"    delta MRR@10 = {p['d_mrr']:+.3f} {p['d_mrr_ic95']} | delta nDCG@5 = {p['d_ndcg']:+.3f} {p['d_ndcg_ic95']}")

# --- Questions françaises (indicatif) ---
if lignes_fr:
    fr = pd.DataFrame(lignes_fr)
    en = dfq[dfq["question_id"].isin(sous_fr["question_id"]) & dfq["variante"].isin(["dense", "rrf", "rrf_rerank"])]
    t_fr = (fr.groupby(["modele", "strategie", "variante"], sort=False)[["mrr_fr", "recall_fr"]].mean().round(3)
              .join(en.groupby(["modele", "strategie", "variante"], sort=False)[["mrr@10", "recall@5"]].mean().round(3)
                    .rename(columns={"mrr@10": "mrr_en", "recall@5": "recall_en"})).reset_index())
    t_fr.to_csv(RES / f"hybride_francais_{args.machine}.csv", index=False)
    print(f"\n=== MÊMES {len(sous_fr)} QUESTIONS : ANGLAIS vs FRANÇAIS (très petit échantillon : indicatif) ===")
    print(t_fr[["modele", "strategie", "variante", "mrr_en", "mrr_fr", "recall_en", "recall_fr"]].to_string(index=False))

print(f"\nFichiers écrits dans {RES}/ (machine = {args.machine})")