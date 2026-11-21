"""Affiche les comparaisons appariées calculées à l'étape 12 (lecture seule)."""
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="pc1")
args = parser.parse_args()

df = pd.read_csv(f"results/hybride_comparaisons_{args.machine}.csv")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 200)
colonnes = ["modele", "strategie", "comparaison", "d_mrr", "d_mrr_ic95", "d_ndcg", "d_ndcg_ic95"]

print("=== CONFIGURATION PRIMAIRE (model_b / sent_150) : les 6 comparaisons ===")
prim = df[(df["modele"] == "model_b") & (df["strategie"] == "sent_150")]
print(prim[colonnes].to_string(index=False))

print("\n=== COMPARAISONS NON AFFICHÉES À L'ÉTAPE 12, TOUTES CONFIGURATIONS ===")
autres = df[df["comparaison"].isin(["bm25_rerank - bm25", "rrf - dense", "dense_rerank - dense"])]
print(autres.sort_values(["comparaison", "modele", "strategie"])[colonnes].to_string(index=False))

print("\n=== SYNTHÈSE PAR COMPARAISON (6 configurations, NON indépendantes : mêmes 42 questions) ===")
synthese = df.groupby("comparaison").agg(
    n_configs=("d_mrr", "size"),
    d_mrr_positifs=("d_mrr", lambda s: int((s > 0).sum())),
    d_mrr_ic_exclut_0=("mrr_ic_contient_0", lambda s: int((~s).sum())),
    d_ndcg_positifs=("d_ndcg", lambda s: int((s > 0).sum())),
    d_ndcg_ic_exclut_0=("ndcg_ic_contient_0", lambda s: int((~s).sum())),
    d_mrr_moyen=("d_mrr", lambda s: round(float(s.mean()), 3)),
)
print(synthese.to_string())
print("\nRappel : intervalles non corrigés pour les comparaisons multiples ; "
      "'ic_exclut_0' = l'intervalle de confiance ne contient pas 0.")