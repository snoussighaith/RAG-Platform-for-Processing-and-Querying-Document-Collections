
"""Télécharge les modèles Hugging Face dans le cache Docker."""

import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder

# Lire la configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Télécharger les modèles d'embeddings
for cle, val in cfg["embeddings"].items():
    if cle.startswith("model_") and isinstance(val, dict):
        print(f"\nTéléchargement du modèle : {val['name']}")
        SentenceTransformer(val["name"], device="cpu")

# Télécharger le reranker
print(f"\nTéléchargement du reranker : {cfg['reranker']['model']}")
CrossEncoder(cfg["reranker"]["model"], device="cpu")

print("\nTous les modèles sont téléchargés et mis en cache.")
