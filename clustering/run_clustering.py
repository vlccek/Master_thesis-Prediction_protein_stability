import os
import argparse
import time
import polars as pl
import numpy as np

# --- OPTIMALIZACE PRO CPU ---
NUM_CORES = str(os.cpu_count())
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_CORES
os.environ["NUMEXPR_NUM_THREADS"] = NUM_CORES

from sklearn.preprocessing import normalize
from sklearn.cluster import HDBSCAN
import umap
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description="Finální Clustering (FIXED)")
    parser.add_argument("--input", type=str, required=True, help="Hlavní soubor s embeddingy")
    parser.add_argument("--valid_seqs", type=str, required=True, help="Soubor s validními sekvencemi")
    parser.add_argument("--output", type=str, required=True, help="Výstupní soubor")
    parser.add_argument("--embedding_col", type=str, default="embedding", help="Název sloupce s vektory")
    parser.add_argument("--seq_col", type=str, default="original_seq_full", help="Název sloupce se sekvencí")

    args = parser.parse_args()

    print(f"🔥 Využívám {NUM_CORES} CPU jader.")

    # 1. NAČTENÍ A FILTRACE
    print(f"📂 Načítám data...")
    start_time = time.time()

    df_full = pl.read_parquet(args.input)
    df_valid = pl.read_parquet(args.valid_seqs)

    # Inner Join a Deduplikace
    df_clean = df_full.join(df_valid, on=args.seq_col, how="inner")
    df_clean = df_clean.unique(subset=[args.seq_col], maintain_order=False)

    print(f"✅ Zpracovávám {len(df_clean)} unikátních proteinů.")

    # 2. PŘÍPRAVA MATICE
    print("🔄 Konvertuji embeddingy do NumPy matice...")
    X = np.stack(df_clean[args.embedding_col].to_numpy()).astype(np.float32)

    # 3. L2 NORMALIZACE
    print("✨ Provádím L2 Normalizaci...")
    X_norm = normalize(X, norm='l2')
    del X

    # 4. UMAP
    print("🧠 Spouštím UMAP (5120D -> 15D)...")
    umap_start = time.time()

    # OPRAVA: random_state=None umožní paralelizaci (n_jobs=-1)
    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=15,
        min_dist=0.0,
        metric='cosine',
        random_state=None,  # Změna pro rychlost!
        low_memory=True,
        n_jobs=-1,
        verbose=True
    )

    with threadpool_limits(limits=int(NUM_CORES), user_api='blas'):
        umap_embeddings = reducer.fit_transform(X_norm)

    print(f"✅ UMAP dokončen za {(time.time() - umap_start) / 60:.1f} min.")

    # 5. HDBSCAN CLUSTERING
    print("🔍 Spouštím HDBSCAN...")

    # OPRAVA: Odstraněn parametr prediction_data (není v sklearn verzi)
    clusterer = HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
        metric='euclidean',
        cluster_selection_method='eom',
        n_jobs=-1
    )

    labels = clusterer.fit_predict(umap_embeddings)

    # V sklearn verzi jsou pravděpodobnosti dostupné takto (pokud je model vypočítal)
    # Někdy vrací 0 pro šum, což je ok.
    if hasattr(clusterer, "probabilities_"):
        probabilities = clusterer.probabilities_
    else:
        # Fallback pro starší verze sklearn, kde to nemusí být dostupné
        probabilities = np.zeros_like(labels, dtype=float)

    # Statistiky
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_percent = (np.sum(labels == -1) / len(labels)) * 100
    print(f"🎉 Nalezeno {n_clusters} clusterů.")
    print(f"🗑️  Míra šumu: {noise_percent:.2f}%")

    # 6. ULOŽENÍ
    print(f"💾 Ukládám výsledky do {args.output}...")

    df_result = df_clean.with_columns([
        pl.Series("cluster_id", labels),
        pl.Series("membership_prob", probabilities)
    ])

    df_result.write_parquet(args.output)
    print("✅ HOTOVO!")


if __name__ == "__main__":
    main()