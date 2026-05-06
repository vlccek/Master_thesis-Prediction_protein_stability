import polars as pl
import os
from tqdm.notebook import tqdm

# --- NASTAVENÍ ---
CHUNK_SIZE = 5000         # Počet sekvencí na jeden soubor (uprav podle potřeby)
OUTPUT_DIR = "./input_chunks"
MIN_SEQ_LEN = 5           # Minimální délka sekvence pro validaci

df_merged = pl.load_csv("../protbert/merged_classification_dataset.csv")


# Vytvoření výstupní složky
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Původní počet řádků: {df_merged.height}")


print("Zpracovávám data (deduplikace a formátování)...")

processed_df = (
    df_merged.lazy()
    .select(pl.col("original_seq_full"))
    .unique()
    .filter(
        pl.col("original_seq_full").is_not_null()
    )
    .with_row_index(name="id")
    .select(
        pl.format(">seq_{}\n{}", pl.col("id"), pl.col("original_seq_full")).alias("fasta_entry")
    )
    .collect()  # Spustíme výpočet a získáme výsledný DataFrame
)

total_seqs = processed_df.height
print(f"Počet unikátních validních sekvencí: {total_seqs}")

print(f"Ukládám do složky '{OUTPUT_DIR}' po {CHUNK_SIZE} sekvencích...")

# Vypočítáme počet chunků
num_chunks = (total_seqs + CHUNK_SIZE - 1) // CHUNK_SIZE

for i in tqdm(range(num_chunks), desc="Generování souborů", unit="chunk"):
    # Rychlý slice bez kopírování dat v paměti
    offset = i * CHUNK_SIZE
    chunk_slice = processed_df["fasta_entry"].slice(offset, CHUNK_SIZE)

    # Převod na list stringů a spojení do jednoho velkého textu (nejrychlejší IO v Pythonu)
    # join("\n") spojí řádky, přidáme "\n" na konec souboru
    file_content = "\n".join(chunk_slice.to_list()) + "\n"

    filename = os.path.join(OUTPUT_DIR, f"chunk_{i:04d}.fasta")

    with open(filename, "w") as f:
        f.write(file_content)

print(f"Hotovo! Vytvořeno {num_chunks} souborů.")