"""Étape 11b : encode tous les chunks avec les modèles d'embeddings (reprise possible)."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sentence_transformers
import torch
import yaml
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--machine", default="pc1")
parser.add_argument("--modeles", nargs="+", default=None,
                    help="clés de config.yaml (ex : model_a model_b) ; par défaut : tous")
parser.add_argument("--strategies", nargs="+",
                    default=["fixed_100_o20", "fixed_200_o40", "sent_150"])
parser.add_argument("--force", action="store_true", help="recalcule même si le fichier existe")
args = parser.parse_args()

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
torch.manual_seed(cfg.get("seed", 42))
DEVICE = cfg["embeddings"].get("device", "cpu")
PROC, RES = Path("data/processed"), Path("results")
SORTIE = Path("indexes/embeddings")
SORTIE.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)

tous = {k: v for k, v in cfg["embeddings"].items() if k.startswith("model_") and isinstance(v, dict)}
cles = args.modeles or list(tous)
inconnus = [c for c in cles if c not in tous]
if inconnus:
    raise SystemExit(f"Clés inconnues : {inconnus}. Disponibles : {list(tous)}")

lignes = []
for cle in cles:
    m = tous[cle]
    pref = m.get("passage_prefix", "")
    modele = None  # chargé seulement si un calcul est nécessaire
    for strat in args.strategies:
        npy = SORTIE / f"{cle}__{strat}.npy"
        meta = SORTIE / f"{cle}__{strat}.json"
        if npy.exists() and meta.exists() and not args.force:
            print(f"[déjà fait] {cle} / {strat}")
            continue
        if modele is None:
            modele = SentenceTransformer(m["name"], device=DEVICE)
        chunks = pd.read_csv(PROC / f"chunks_{strat}.csv")
        textes = [pref + t for t in chunks["text"]]
        print(f"\n>>> {cle} ({m['name']}) | {strat} | {len(textes)} chunks")
        t0 = time.perf_counter()
        emb = modele.encode(textes, batch_size=32, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True).astype("float32")
        duree = time.perf_counter() - t0

        # Vérifications automatiques
        dim = modele.get_sentence_embedding_dimension()
        assert emb.shape == (len(chunks), dim), f"forme inattendue : {emb.shape}"
        assert np.isfinite(emb).all(), "valeurs NaN ou infinies"
        normes = np.linalg.norm(emb, axis=1)
        assert np.allclose(normes, 1.0, atol=1e-3), "vecteurs non normalisés"

        np.save(npy, emb)
        info = {
            "cle": cle, "modele": m["name"], "strategie": strat, "n_chunks": int(len(chunks)),
            "dim": int(dim), "max_seq_length": int(modele.max_seq_length),
            "passage_prefix": pref, "device": DEVICE, "machine": args.machine,
            "secondes": round(duree, 1), "chunks_par_s": round(len(chunks) / duree, 1),
            "sha256_ordre_chunk_ids": hashlib.sha256("\n".join(chunks["chunk_id"]).encode()).hexdigest(),
            "torch": torch.__version__, "sentence_transformers": sentence_transformers.__version__,
        }
        meta.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
        lignes.append(info)
        print(f"    OK : {emb.shape} en {duree:.0f} s ({info['chunks_par_s']} chunks/s)")

# Journal des temps (cumulatif : les lancements successifs s'ajoutent)
if lignes:
    chemin = RES / f"embeddings_temps_{args.machine}.csv"
    nouveau = pd.DataFrame(lignes).drop(columns=["sha256_ordre_chunk_ids"])
    if chemin.exists():
        nouveau = pd.concat([pd.read_csv(chemin), nouveau], ignore_index=True)
    nouveau.to_csv(chemin, index=False)
    print(f"\nJournal des temps : {chemin}")

# Inventaire final
print("\n=== INVENTAIRE DES FICHIERS DE VECTEURS ===")
for f in sorted(SORTIE.glob("*.npy")):
    print(f"{f.name:40s} {f.stat().st_size / 1e6:6.1f} Mo")