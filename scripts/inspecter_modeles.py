"""Étape 11a : caractéristiques des modèles d'embeddings, troncature des chunks, temps d'encodage."""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")  # masque l'avertissement « séquence plus longue que le maximum »

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="pc1")
args = parser.parse_args()

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
SEED = cfg.get("seed", 42)
torch.manual_seed(SEED)
DEVICE = cfg["embeddings"].get("device", "cpu")
PROC, RES = Path("data/processed"), Path("results")
RES.mkdir(exist_ok=True)
STRATEGIES = ["fixed_100_o20", "fixed_200_o40", "sent_150"]

modeles = {k: v for k, v in cfg["embeddings"].items()
           if k.startswith("model_") and isinstance(v, dict)}
print(f"Modèles trouvés dans config.yaml : {[v['name'] for v in modeles.values()]}")
print(f"Device : {DEVICE} | threads torch : {torch.get_num_threads()} | machine : {args.machine}")

chunks = {s: pd.read_csv(PROC / f"chunks_{s}.csv") for s in STRATEGIES}
# Échantillon FIXE (graine 42) pour mesurer le temps : même 200 chunks pour tous les modèles
echantillon = chunks["fixed_200_o40"].sample(200, random_state=SEED)["text"].tolist()

lignes = []
for cle, m in modeles.items():
    nom = m["name"]
    pref = m.get("passage_prefix", "")
    t0 = time.perf_counter()
    modele = SentenceTransformer(nom, device=DEVICE)
    t_chargement = time.perf_counter() - t0
    max_len = modele.max_seq_length
    dim = modele.get_sentence_embedding_dimension()

    # Temps d'encodage : un petit échauffement, puis mesure sur 200 chunks
    textes = [pref + t for t in echantillon]
    modele.encode(textes[:16], batch_size=16, show_progress_bar=False)
    t0 = time.perf_counter()
    modele.encode(textes, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    t_enc = time.perf_counter() - t0

    print("\n" + "=" * 72)
    print(f"{cle} : {nom}")
    print(f"Dimension : {dim} | longueur max d'entrée : {max_len} tokens | préfixe passage : {pref!r}")
    print(f"Chargement : {t_chargement:.1f} s | encodage de 200 chunks : {t_enc:.1f} s "
          f"({200 / t_enc:.1f} chunks/s)")

    for s in STRATEGIES:
        textes_s = [pref + t for t in chunks[s]["text"]]
        longueurs = np.array([len(ids) for ids in
                              modele.tokenizer(textes_s, add_special_tokens=True,
                                               truncation=False)["input_ids"]])
        part_coupes = float((longueurs > max_len).mean())
        print(f"  {s:14s} | tokens médiane {np.median(longueurs):5.0f} | p95 {np.percentile(longueurs, 95):5.0f} "
              f"| max {longueurs.max():5d} | chunks coupés : {100 * part_coupes:5.1f} %")
        lignes.append({
            "modele": nom, "dim": dim, "max_seq_length": max_len, "strategie": s,
            "tokens_mediane": float(np.median(longueurs)),
            "tokens_p95": float(np.percentile(longueurs, 95)),
            "pct_chunks_coupes": round(100 * part_coupes, 1),
            "chunks_par_s_echantillon": round(200 / t_enc, 1),
            "machine": args.machine,
        })
    del modele

pd.DataFrame(lignes).to_csv(RES / f"embeddings_modeles_{args.machine}.csv", index=False)
print(f"\nRésultats écrits : results/embeddings_modeles_{args.machine}.csv")