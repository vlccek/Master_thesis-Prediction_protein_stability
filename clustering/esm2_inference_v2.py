import os
import argparse
import glob
import torch
from transformers import AutoTokenizer, AutoModel
import polars as pl
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="ESM2 Inference Wrapper for AMD GPUs")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to output Parquet file")
    parser.add_argument("--model", type=str, default="facebook/esm2_t48_15B_UR50D", help="ESM2 model name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--column", type=str, default="original_seq_full", help="Column name containing sequences")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length")
    # Přidán parametr pro dočasné soubory
    parser.add_argument("--tmp_dir", type=str, default="./tmp_chunks", help="Directory for temporary chunk files")

    args = parser.parse_args()

    # Na ROCm PyTorch se AMD karty hlásí jako 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Načtení v bfloat16 (nativní pro MI200+ architekturu)
    # Pro 15B model je to cca 30 GB VRAM.
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )

    # OPRAVA 1: Přesun modelu na grafickou kartu!
    model = model.to(device)
    model.eval()

    print(f"Reading input data: {args.input}")
    df_to_process = pl.read_parquet(args.input)
    print(f"Total sequences in input: {len(df_to_process)}")

    # Vytvoření dočasné složky pro ukládání po kouscích
    os.makedirs(args.tmp_dir, exist_ok=True)

    # --- RESUME LOGIC (Upraveno pro chunky) ---
    processed_seqs = set()

    # 1. Zkontrolujeme hlavní výstupní soubor
    if os.path.exists(args.output):
        try:
            df_existing = pl.read_parquet(args.output)
            if args.column in df_existing.columns:
                processed_seqs.update(df_existing[args.column].to_list())
        except Exception as e:
            print(f"Error reading existing output: {e}. Starting fresh.")

    # 2. Zkontrolujeme nedokončené chunky v dočasné složce
    chunk_files = glob.glob(os.path.join(args.tmp_dir, "*.parquet"))
    for cf in chunk_files:
        try:
            df_chunk = pl.read_parquet(cf)
            processed_seqs.update(df_chunk[args.column].to_list())
        except Exception:
            pass

    if len(processed_seqs) > 0:
        print(f"Found {len(processed_seqs)} already processed sequences. Resuming...")

    # Vyfiltrování již zpracovaných
    df_to_process = df_to_process.filter(~pl.col(args.column).is_in(list(processed_seqs)))
    print(f"Sequences left to process: {len(df_to_process)}")

    if len(df_to_process) == 0:
        print("✅ All sequences already processed.")
        return

    # OPRAVA 2: Ukládání po chunkech s unikátními názvy
    import uuid

    for df_batch in tqdm(df_to_process.iter_slices(n_rows=args.batch_size),
                         total=(len(df_to_process) // args.batch_size + 1)):
        seqs = df_batch[args.column].to_list()

        inputs = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # torch.inference_mode() je nepatrně rychlejší a paměťově úspornější než no_grad()
        with torch.inference_mode():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state

            # Mean Pooling přes délku sekvence (dim 1)
            # Přetypování masky na bfloat16, aby odpovídala výstupu z modelu
            mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).to(torch.bfloat16)
            sum_embeddings = torch.sum(last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)

            # Převedení zpět na float32 pro uložení do Parquetu
            mean_pooled = (sum_embeddings / sum_mask).cpu().to(torch.float32).numpy()

        # Uložení aktuálního batche do malého parquet souboru s unikátním názvem
        df_batch_result = df_batch.with_columns(
            pl.Series("embedding", mean_pooled.tolist())
        )

        chunk_name = f"chunk_{uuid.uuid4().hex[:12]}.parquet"
        chunk_path = os.path.join(args.tmp_dir, chunk_name)
        df_batch_result.write_parquet(chunk_path)

    print("✅ Inference hotova. Slučuji dočasné soubory do finálního výstupu...")

    # OPRAVA 3: Robustnější spojení
    chunk_files = glob.glob(os.path.join(args.tmp_dir, "*.parquet"))
    dfs = []

    # Nejdříve načteme hlavní soubor, pokud existuje
    if os.path.exists(args.output):
        try:
            dfs.append(pl.read_parquet(args.output))
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")

    # Pak načteme všechny chunky
    for cf in tqdm(chunk_files, desc="Načítání chunků"):
        try:
            dfs.append(pl.read_parquet(cf))
        except Exception as e:
            print(f"Warning: Skipping corrupted chunk {cf}: {e}")

    if dfs:
        # Spojení a atomický zápis (do .tmp a pak přejmenovat)
        merged_df = pl.concat(dfs)
        temp_output = args.output + ".tmp"
        merged_df.write_parquet(temp_output)
        os.replace(temp_output, args.output)

        # Úklid dočasné složky jen po úspěšném zápisu
        for cf in chunk_files:
            try:
                os.remove(cf)
            except Exception:
                pass

    print(f"✅ Vše hotovo. Výsledky uloženy v {args.output}")


if __name__ == "__main__":
    main()
