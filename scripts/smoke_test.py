
"""Vérification des composants du projet RAG."""

import yaml
import requests
import torch


# Charger la configuration
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

HOTE_OLLAMA = cfg["llm"]["host"]


def test_torch():
    return (
        f"PyTorch = {torch.__version__}, "
        f"CUDA disponible = {torch.cuda.is_available()}"
    )


def test_yaml():
    assert cfg["seed"] == 42
    assert "embeddings" in cfg
    assert "reranker" in cfg
    assert "llm" in cfg
    return "configuration chargée correctement"


def test_embeddings():
    from sentence_transformers import SentenceTransformer

    modele = SentenceTransformer(
        cfg["embeddings"]["model_a"]["name"],
        device=cfg["embeddings"]["device"]
    )

    vecteurs = modele.encode(
        [
            "The government announced a new budget.",
            "The football team won the match."
        ],
        normalize_embeddings=True
    )

    assert vecteurs.shape == (2, 384), (
        f"Dimension incorrecte : {vecteurs.shape}"
    )

    return f"forme des vecteurs = {vecteurs.shape}"


def test_bge():
    from sentence_transformers import SentenceTransformer

    modele = SentenceTransformer(
        cfg["embeddings"]["model_b"]["name"],
        device=cfg["embeddings"]["device"]
    )

    prefixe = cfg["embeddings"]["model_b"]["query_prefix"]

    vecteurs = modele.encode(
        [prefixe + "What happened to oil prices?"],
        normalize_embeddings=True
    )

    assert vecteurs.shape == (1, 384), (
        f"Dimension incorrecte : {vecteurs.shape}"
    )

    return f"forme des vecteurs = {vecteurs.shape}"


def test_bm25():
    from rank_bm25 import BM25Okapi

    corpus = [
        "oil prices increased significantly",
        "football team won the match",
        "government announced a new budget"
    ]

    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores("oil prices".split())

    assert scores[0] > scores[1], (
        "BM25 ne retrouve pas correctement le document pertinent"
    )

    return f"scores = {[round(float(s), 2) for s in scores]}"


def test_reranker():
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(
        cfg["reranker"]["model"],
        device=cfg["embeddings"]["device"]
    )

    q = "What happened to oil prices?"

    scores = ce.predict([
        (q, "Oil prices increased by 2%."),
        (q, "The football team won the match.")
    ])

    if scores[0] <= scores[1]:
        raise RuntimeError(
            "Le reranker classe mal un cas trivial"
        )

    return f"scores = {[round(float(s), 2) for s in scores]}"


def test_ollama():
    response = requests.get(
        f"{HOTE_OLLAMA}/api/tags",
        timeout=10
    )

    response.raise_for_status()

    modeles = [
        m["name"]
        for m in response.json().get("models", [])
    ]

    attendus = [
        cfg["llm"]["model"],
        cfg["llm"]["model_alt"]
    ]

    manquants = [
        modele for modele in attendus
        if modele not in modeles
    ]

    if manquants:
        raise RuntimeError(
            f"Modèles absents : {manquants}. Installés : {modeles}"
        )

    return f"serveur OK, modèles = {modeles}"


# Liste des tests
tests = [
    ("PyTorch", test_torch),
    ("Configuration YAML", test_yaml),
    ("Sentence Transformers", test_embeddings),
    ("BGE", test_bge),
    ("BM25", test_bm25),
    ("Reranker", test_reranker),
    ("Ollama", test_ollama),
]


# Exécution des tests
if __name__ == "__main__":

    echecs = []

    print("\n========== SMOKE TEST RAG ==========\n")

    for nom, fonction in tests:
        try:
            resultat = fonction()
            print(f"[OK] {nom} : {resultat}")

        except Exception as erreur:
            echecs.append(nom)
            print(f"[ERREUR] {nom} : {type(erreur).__name__}: {erreur}")

    print("\n====================================")

    if echecs:
        print(f"ÉCHEC : {len(echecs)} test(s) : {echecs}")
        raise SystemExit(1)

    print("TOUT EST PRÊT !")
