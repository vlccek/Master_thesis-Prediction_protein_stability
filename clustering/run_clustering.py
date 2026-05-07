# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2026-03-03

import os
import gc
import joblib
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
import hdbscan
import umap
from joblib import Parallel, delayed

# ==========================================
# --- NASTAVENÍ A CESTY ---
# ==========================================
META_PARQUET = "../protbert/datasets/dataset_merged_with_families.parquet"
EMBEDDINGS_PARQUET = "tmp_chunks/chunk_merged.parquet"

OUT_DIR = "experiment_direct_max"
os.makedirs(OUT_DIR, exist_ok=True)

WT_PARQUET = os.path.join(OUT_DIR, "noreverse.parquet")
MUT_PARQUET = os.path.join(OUT_DIR, "mutants_only.parquet")

MODEL_UMAP = os.path.join(OUT_DIR, "model_umap.pkl")
MODEL_HDBSCAN = os.path.join(OUT_DIR, "model_hdbscan.pkl")
RESULTS_WT = os.path.join(OUT_DIR, "results_wt.parquet")
RESULTS_MUT = os.path.join(OUT_DIR, "results_mut.parquet")

PLOT_FILE_PNG = os.path.join(OUT_DIR, "publication_plot.png")
PLOT_FILE_PDF = os.path.join(OUT_DIR, "publication_plot.pdf")

EMBEDDING_COL = "embedding"
SEQ_COL = "original_seq_full"


# ==========================================
# --- POMOCNÉ FUNKCE ---
# ==========================================

def convert_to_numpy(df, col):
    """Efektivnější převod Polars List sloupce na NumPy matici."""
    print(f"⏳ Převádím sloupec {col} na NumPy (v RAM to vyvolá špičku)...")
    # Stackování listů je RAM náročné, ale v 750GB RAM to projde
    return np.vstack(df[col].to_numpy()).astype(np.float32)


def predict_chunk(chunk, clusterer):
    """Funkce pro paralelní predikci jednoho balíku dat."""
    labels, probs = hdbscan.approximate_predict(clusterer, chunk)
    return labels, probs


# ==========================================
# --- PIPELINE KROKY ---
# ==========================================

def step_1_split_data():
    print("📂 KROK 1: Spojuji a rozděluji data...")
    if os.path.exists(WT_PARQUET) and os.path.exists(MUT_PARQUET):
        print("✅ Hotovo, přeskakuji.")
        return

    df_meta = pl.read_parquet(META_PARQUET).select([SEQ_COL, "reverse"]).unique(subset=[SEQ_COL])
    df_emb = pl.read_parquet(EMBEDDINGS_PARQUET).select([SEQ_COL, EMBEDDING_COL]).unique(subset=[SEQ_COL])

    df_full = df_emb.join(df_meta, on=SEQ_COL, how="inner")
    del df_meta, df_emb;
    gc.collect()

    df_full.filter(pl.col("reverse") == False).write_parquet(WT_PARQUET)
    df_full.filter(pl.col("reverse") == True).write_parquet(MUT_PARQUET)
    print("✅ Data rozdělena.")


def step_2_train_wt_space():
    print("\n🧬 KROK 2: Trénování WT prostoru (DIRECT 5120D)")
    df_wt = pl.read_parquet(WT_PARQUET)
    X_wt = convert_to_numpy(df_wt, EMBEDDING_COL)

    print("✨ L2 Normalizace...")
    X_wt = normalize(X_wt, norm='l2')

    print("🧠 Učím DIRECT UMAP (5120D -> 15D)...")
    # Vypnuto random_state pro maximální paralelizaci (n_jobs=-1)
    reducer = umap.UMAP(
        n_components=15,
        n_neighbors=30,
        min_dist=0.0,
        metric='cosine',
        low_memory=False,  # Máme 750GB, chceme rychlost
        n_jobs=-1,
        verbose=True
    )
    wt_umap_15d = reducer.fit_transform(X_wt)
    joblib.dump(reducer, MODEL_UMAP)

    print("🧠 Učím vizualizační UMAP (5120D -> 2D)...")
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', n_jobs=-1)
    wt_umap_2d = reducer_2d.fit_transform(X_wt)
    del X_wt;
    gc.collect()

    print("🔍 Shlukuji HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=5, metric='euclidean',
        prediction_data=True, core_dist_n_jobs=-1
    )
    wt_labels = clusterer.fit_predict(wt_umap_15d)
    joblib.dump(clusterer, MODEL_HDBSCAN)

    df_wt = df_wt.with_columns([
        pl.Series("cluster_id", wt_labels, dtype=pl.Int32),
        pl.Series("umap_2d_x", wt_umap_2d[:, 0], dtype=pl.Float32),
        pl.Series("umap_2d_y", wt_umap_2d[:, 1], dtype=pl.Float32)
    ])
    df_wt.drop(EMBEDDING_COL).write_parquet(RESULTS_WT)
    print("✅ KROK 2 dokončen.")


def step_3_project_mutants():
    """
    KROK 3: Masivně paralelizovaná projekce mutantů.
    Využívá horizontální škálování pro UMAP transform i HDBSCAN predikci.
    """
    print("\n🦠 KROK 3: Paralelní projekce mutantů (Full Power Mode)")

    # Nastavení počtu jader pro paralelizaci (využijeme polovinu tvých 256 pro stabilitu)
    N_CORES = 128

    # 1. Načtení modelů
    print("📂 Načítám uložené modely z disku...")
    reducer = joblib.load(MODEL_UMAP)
    clusterer = joblib.load(MODEL_HDBSCAN)

    # 2. Načtení a příprava dat
    print(f"⏳ Načítám mutanty z {MUT_PARQUET}...")
    df_mut = pl.read_parquet(MUT_PARQUET)

    # Efektivní převod na NumPy (vyvolá krátkodobou špičku v RAM)
    X_mut = np.vstack(df_mut[EMBEDDING_COL].to_numpy()).astype(np.float32)

    print("✨ L2 Normalizace mutantů...")
    X_mut = normalize(X_mut, norm='l2')

    # --- ČÁST A: PARALELNÍ UMAP TRANSFORM (5120D -> 15D) ---
    print(f"🧠 Spouštím UMAP transformaci na {N_CORES} jádrech...")

    # Rozdělíme matici na chunks (horizontální škálování)
    chunks_in = np.array_split(X_mut, N_CORES)

    # UMAP transformace v paralelních procesech (backend loky využívá memmapping)
    transformed_chunks = Parallel(n_jobs=N_CORES, backend="loky")(
        delayed(reducer.transform)(c) for c in chunks_in
    )

    # Slepíme výsledky (15D embeddingy)
    mut_umap_15d = np.vstack(transformed_chunks)

    # Uvolnění původních 5120D dat (ušetříme ~60GB+ v RAM)
    del X_mut, chunks_in, transformed_chunks;
    gc.collect()

    # --- ČÁST B: PARALELNÍ HDBSCAN PREDIKCE ---
    print(f"🎯 Spouštím HDBSCAN predikci na {N_CORES} jádrech...")

    # Rozdělíme 15D matici na chunks
    chunks_umap = np.array_split(mut_umap_15d, N_CORES)

    # Paralelní predikce clusterů
    prediction_results = Parallel(n_jobs=N_CORES, backend="loky")(
        delayed(hdbscan.approximate_predict)(clusterer, c) for c in chunks_umap
    )

    # Rozbalení výsledků (každý chunk vrací tuple: labels, probabilities)
    final_labels = np.concatenate([res[0] for res in prediction_results])
    final_probs = np.concatenate([res[1] for res in prediction_results])

    # Finální úklid pomocných polí
    del mut_umap_15d, chunks_umap, prediction_results;
    gc.collect()

    # 3. Uložení výsledků
    print("💾 Zapisuji výsledky do Parquetu...")
    df_mut = df_mut.with_columns([
        pl.Series("cluster_id", final_labels, dtype=pl.Int32),
        pl.Series("membership_prob", final_probs, dtype=pl.Float32)
    ])

    # Uložíme bez embeddingů (ty už nepotřebujeme, ušetříme místo na disku)
    df_mut.drop(EMBEDDING_COL).write_parquet(RESULTS_MUT)

    print(f"✅ KROK 3 DOKONČEN. Výsledky uloženy v {RESULTS_MUT}")



def step_4_publication_plot():
    print("\n🎨 KROK 4: Generování grafu")
    df_wt = pl.read_parquet(RESULTS_WT).to_pandas()

    plt.figure(figsize=(10, 8))
    # Šum
    sns.scatterplot(data=df_wt[df_wt['cluster_id'] == -1], x='umap_2d_x', y='umap_2d_y', color='lightgrey', s=2,
                    alpha=0.3, label='Noise')
    # Clustery
    sns.scatterplot(data=df_wt[df_wt['cluster_id'] != -1], x='umap_2d_x', y='umap_2d_y', hue='cluster_id',
                    palette='tab20', s=5, alpha=0.7, edgecolor=None, legend=False)

    plt.title("Direct 5120D Protein Landscape", pad=20)
    sns.despine()
    plt.savefig(PLOT_FILE_PNG, dpi=300);
    plt.savefig(PLOT_FILE_PDF)
    print("✅ Grafy uloženy.")


if __name__ == "__main__":
    step_1_split_data()
    step_2_train_wt_space()
    step_3_project_mutants()
    step_4_publication_plot()
    print("🎉 PIPELINE KOMPLETNÍ!")