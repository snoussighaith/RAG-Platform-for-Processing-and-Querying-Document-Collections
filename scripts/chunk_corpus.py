"""Découpe le corpus en chunks avec 3 stratégies et vérifie la traçabilité."""
import re
from pathlib import Path

import pandas as pd

PROC = Path("data/processed")
corpus = pd.read_csv(PROC / "corpus.csv")
pd.set_option("display.width", 200)


def fenetres_fixes(n_mots, taille, chevauchement, decalage=0):
    """Fenêtres glissantes de `taille` mots. Renvoie [(début, fin)] en indices de mots."""
    pas = taille - chevauchement
    spans = []
    debut = 0
    while True:
        fin = min(debut + taille, n_mots)
        spans.append((decalage + debut, decalage + fin))
        if fin >= n_mots:  # la dernière fenêtre atteint la fin du texte
            break
        debut += pas
    return spans


def phrases_regroupees(texte, taille_max):
    """Regroupe des phrases entières jusqu'à `taille_max` mots, avec 1 phrase de chevauchement.
    Une phrase plus longue que `taille_max` est coupée en fenêtres de mots."""
    # Coupure sur . ! ? suivi d'une espace : les décimaux (95.5) ne sont pas coupés
    phrases = [p for p in re.split(r"(?<=[.!?])\s+", texte) if p]
    bornes = []  # (début, fin) de chaque phrase, en indices de mots
    pos = 0
    for p in phrases:
        n = len(p.split())
        bornes.append((pos, pos + n))
        pos += n

    spans = []
    courant = []     # indices des phrases du chunk en cours
    nouveau = False  # le chunk contient-il une phrase pas encore émise ?
    for i, (d, f) in enumerate(bornes):
        w = f - d
        if w > taille_max:  # phrase trop longue : repli sur des fenêtres de mots
            if nouveau:
                spans.append((bornes[courant[0]][0], bornes[courant[-1]][1]))
            courant, nouveau = [], False
            spans.extend(fenetres_fixes(w, taille_max, 0, decalage=d))
            continue
        total = sum(bornes[j][1] - bornes[j][0] for j in courant)
        if courant and total + w > taille_max:  # le chunk est plein : on l'émet
            if nouveau:
                spans.append((bornes[courant[0]][0], bornes[courant[-1]][1]))
            dernier = courant[-1]
            w_dernier = bornes[dernier][1] - bornes[dernier][0]
            # On garde la dernière phrase comme chevauchement si elle tient avec la nouvelle
            courant = [dernier] if w_dernier + w <= taille_max else []
        courant.append(i)
        nouveau = True
    if nouveau:
        spans.append((bornes[courant[0]][0], bornes[courant[-1]][1]))
    return spans


STRATEGIES = {
    "fixed_100_o20": lambda texte, n: fenetres_fixes(n, 100, 20),
    "fixed_200_o40": lambda texte, n: fenetres_fixes(n, 200, 40),
    "sent_150": lambda texte, n: phrases_regroupees(texte, 150),
}

resume = []
for nom, decouper in STRATEGIES.items():
    lignes = []
    for doc in corpus.itertuples(index=False):
        mots = doc.text.split()
        spans = decouper(doc.text, len(mots))
        # Vérifications de traçabilité : couverture complète, sans trou
        assert spans[0][0] == 0 and spans[-1][1] == len(mots), f"couverture incomplète : {doc.doc_id}"
        for k in range(len(spans) - 1):
            assert spans[k + 1][0] <= spans[k][1], f"trou entre deux chunks : {doc.doc_id}"
        for k, (a, b) in enumerate(spans):
            texte_chunk = " ".join(mots[a:b])
            assert b > a and texte_chunk in doc.text, f"chunk invalide : {doc.doc_id} #{k}"
            lignes.append({
                "chunk_id": f"{doc.doc_id}_c{k:02d}",
                "doc_id": doc.doc_id,
                "category": doc.category,
                "chunk_index": k,
                "start_word": a,
                "n_words": b - a,
                "text": texte_chunk,
            })
    df = pd.DataFrame(lignes)
    df["n_chunks_doc"] = df.groupby("doc_id")["chunk_id"].transform("count")
    assert df["chunk_id"].is_unique
    df.to_csv(PROC / f"chunks_{nom}.csv", index=False, encoding="utf-8")

    m = df["n_words"]
    resume.append({
        "strategie": nom,
        "n_chunks": len(df),
        "chunks_par_doc": round(len(df) / len(corpus), 2),
        "mots_moyen": round(m.mean(), 1),
        "mots_median": float(m.median()),
        "mots_min": int(m.min()),
        "mots_max": int(m.max()),
        "chunks_sous_30_mots": int((m < 30).sum()),
    })
    print(f"\n--- {nom} : 2 premiers chunks ---")
    for r in df.head(2).itertuples():
        print(f"{r.chunk_id} | début={r.start_word} | {r.n_words} mots | {r.text[:110]}...")

stats = pd.DataFrame(resume)
stats.to_csv(PROC / "chunking_stats.csv", index=False)
print("\n=== RÉSUMÉ ===")
print(stats.to_string(index=False))
print("Vérifications de traçabilité : OK")