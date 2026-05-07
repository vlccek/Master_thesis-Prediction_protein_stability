# Brno universioty of technology: VUT FIT / BUT FIT
# Master thesis
# Predikce vlivu mutací na stabilitu proteinů / Prediction of the Effect of Mutations on Protein Stability
# 
# author: Jakub Vlk
# date: 2026-03-03

import polars as pl
import glob
import os
import argparse
import gc
import uuid
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Hierarchické sloučení s průběžným mazáním a deduplikací.")
    parser.add_argument("--dir", type=str, default="./tmp_chunks", help="Složka s kousky")
    parser.add_argument("--out_name", type=str, default="chunk_merged_collection.parquet",
                        help="Název výsledného souboru")
    parser.add_argument("--batch_size", type=int, default=1000, help="Kolik souborů se má spojit najednou")
    args = parser.parse_args()

    # Název sloupce pro kontrolu duplicit
    DEDUP_COLUMN = "original_seq_full"

    # 1. Najít všechny .parquet soubory
    pattern = os.path.join(args.dir, "*.parquet")
    all_files = glob.glob(pattern)

    final_path = os.path.abspath(os.path.join(args.dir, args.out_name))

    files = [f for f in all_files if os.path.abspath(f) != final_path and "_temp_batch_" not in f]
    existing_temp_batches = [f for f in all_files if "_temp_batch_" in f]

    if not files and not existing_temp_batches:
        print(f"ℹ️ Ve složce {args.dir} nejsou žádné soubory ke sloučení.")
        return

    print(f"🔍 Nalezeno {len(files)} malých souborů a {len(existing_temp_batches)} již předsloučených bloků.")

    # 2. Rozdělení malých souborů do dávek
    batches = [files[i:i + args.batch_size] for i in range(0, len(files), args.batch_size)]
    if batches:
        print(f"📦 Rozděleno do {len(batches)} dávek (po max {args.batch_size} souborech).")

    temp_batch_files = existing_temp_batches.copy()

    # 3. Zpracování jednotlivých dávek (Fáze 1)
    for batch in tqdm(batches, desc="🚀 Slučování, deduplikace a mazání dávek"):
        dfs = []
        valid_files = []

        for f in batch:
            try:
                dfs.append(pl.read_parquet(f))
                valid_files.append(f)
            except Exception as e:
                print(f"\n❌ Chyba při načítání {f}: {e}")

        if not dfs:
            continue

        # Spojení dávky
        batch_df = pl.concat(dfs)

        # ✂️ DEDUPLIKACE DÁVKY ✂️
        if DEDUP_COLUMN in batch_df.columns:
            batch_df = batch_df.unique(subset=[DEDUP_COLUMN], maintain_order=False)

        # Uložení středního souboru
        temp_file = os.path.join(args.dir, f"_temp_batch_{uuid.uuid4().hex[:8]}.parquet")
        batch_df.write_parquet(temp_file)
        temp_batch_files.append(temp_file)

        # Agresivní úklid RAM
        del dfs
        del batch_df
        gc.collect()

        # 🔥 PRŮBĚŽNÉ MAZÁNÍ 🔥
        for f in valid_files:
            try:
                os.remove(f)
            except Exception:
                pass

                # 4. Finální sloučení středních souborů (Fáze 2)
    if temp_batch_files:
        print("\n⚡ Načítám střední soubory pro finální sloučení...")
        final_dfs = []
        for f in tqdm(temp_batch_files, desc="🎯 Finální slučování"):
            final_dfs.append(pl.read_parquet(f))

        print("💾 Spojuji a provádím finální deduplikaci...")
        final_df = pl.concat(final_dfs)

        # ✂️ FINÁLNÍ DEDUPLIKACE NAPŘÍČ VŠEMI DÁVKAMI ✂️
        start_rows = len(final_df)
        if DEDUP_COLUMN in final_df.columns:
            final_df = final_df.unique(subset=[DEDUP_COLUMN], maintain_order=False)
            end_rows = len(final_df)
            print(f"✂️ Odstraněno {start_rows - end_rows} duplicitních řádků celkově.")

        temp_final_path = final_path + ".tmp"
        final_df.write_parquet(temp_final_path)

        # Úklid RAM
        del final_dfs
        del final_df
        gc.collect()

        # MAZÁNÍ DOČASNÝCH STŘEDNÍCH SOUBORŮ
        print("🧹 Mažu dočasné střední soubory...")
        for f in temp_batch_files:
            try:
                os.remove(f)
            except Exception:
                pass

        # Přejmenování na finální název
        os.replace(temp_final_path, final_path)

        print(f"\n✨ Hotovo! Finální soubor (bez duplicit) uložen jako '{args.out_name}'.")


if __name__ == "__main__":
    main()