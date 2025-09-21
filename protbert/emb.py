#
# SCRIPT: generate_diff_embeddings_from_uniprot.py
#
# Tento skript provádí kompletní proces přípravy dat:
# 1. Načte data.
# 2. Stáhne WT sekvence z UniProt databáze.
# 3. Vygeneruje vnoření pro všechny sekvence.
# 4. Vypočítá a uloží diferenční vnoření (Mutant - WT).
#
import pandas as pd
import numpy as np
import torch
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import time

# --- KROK 1: Načtení a základní čištění dat ---

print("Krok 1: Načítání a čištění datasetu...")
try:
    df = pd.read_pickle('lehner_dataset.pkl')
except FileNotFoundError:
    df = pd.read_csv('lehner_dataset.txt')

# Přejmenování sloupců pro konzistenci
if 'aa_seq' in df.columns:
    df = df.rename(columns={'aa_seq': 'sequence'})
id_col = 'uniprot_ID'

print(df)

# Základní čištění
df.dropna(subset=['normalized_fitness', 'sequence', 'uniprot_ID'], inplace=True)
df = df[~df['sequence'].str.contains("\*", na=False)]
df.reset_index(drop=True, inplace=True)

print(f"Dataset načten. Nalezeno {df['uniprot_ID'].nunique()} unikátních proteinů.")


# --- KROK 2: Stažení WT sekvencí z UniProt ---

print("\nKrok 2: Stahování kanonických WT sekvencí z databáze UniProt...")

def get_sequence_from_uniprot(uniprot_ID, session):
    """Stáhne FASTA sekvenci pro dané UniProt ID."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_ID}.fasta"
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Vyvolá chybu pro status kódy 4xx/5xx
        # Odstranění FASTA hlavičky a znaků nového řádku
        return "".join(response.text.split('\n')[1:])
    except requests.exceptions.RequestException as e:
        print(f"  Chyba při stahování {uniprot_ID}: {e}")
        return None

# Nastavení session s automatickým opakováním pro robustnost
retry_strategy = Retry(total=3, status_forcelist=[429, 500, 502, 503, 504], backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

unique_uniprot_IDs = df['uniprot_ID'].unique()
wt_sequences_map = {}
for uid in tqdm(unique_uniprot_IDs, desc="Stahování z UniProt"):
    wt_sequences_map[uid] = get_sequence_from_uniprot(uid, session)
    time.sleep(0.1) # Malá pauza, abychom nezahltili API

# Přidání sloupce 'wt_sequence' do DataFrame
df['wt_sequence'] = df['uniprot_ID'].map(wt_sequences_map)

# Odstranění řádků, pro které se nepodařilo stáhnout WT sekvenci
initial_rows = len(df)
df.dropna(subset=['wt_sequence'], inplace=True)
if len(df) < initial_rows:
    print(f"Varování: {initial_rows - len(df)} řádků bylo odstraněno, protože se pro ně nepodařilo stáhnout WT sekvenci.")

print("WT sekvence úspěšně staženy a přidány do datasetu.")


# --- KROK 3: Generování vnoření pro všechny unikátní sekvence ---

print("\nKrok 3: Příprava a generování vnoření...")

# Inicializace ProtBERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "Rostlab/prot_bert_bfd"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
model = BertModel.from_pretrained(model_name).to(device)
model.eval()

def get_embeddings_batched(sequences, batch_size=32):
    """Funkce pro dávkové generování vnoření."""
    all_embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

# Sestavení seznamu všech unikátních sekvencí (WT i mutanti) pro efektivitu
all_unique_sequences = pd.concat([df['sequence'], df['wt_sequence']]).unique()
all_unique_sequences_spaced = [' '.join(list(s)) for s in all_unique_sequences]

print(f"Nalezeno {len(all_unique_sequences)} unikátních sekvencí k převedení na vnoření.")
embeddings_array = get_embeddings_batched(all_unique_sequences_spaced)

# Vytvoření slovníku pro rychlé vyhledávání: {sekvence: vnoření}
all_embeddings_dict = {seq: emb for seq, emb in zip(all_unique_sequences, embeddings_array)}
print("Vnoření pro všechny unikátní sekvence byla vytvořena.")


# --- KROK 4: Výpočet a uložení diferenčních vnoření ---

print("\nKrok 4: Výpočet diferenčních vnoření (Mutant - WT)...")

diff_embeddings = []
# Použití .values pro rychlejší iteraci
sequences = df['sequence'].values
wt_sequences = df['wt_sequence'].values

for i in tqdm(range(len(df)), desc="Počítání rozdílů"):
    mutant_emb = all_embeddings_dict[sequences[i]]
    wt_emb = all_embeddings_dict[wt_sequences[i]]
    diff_embeddings.append(mutant_emb - wt_emb)

X_diff = np.array(diff_embeddings)
y = df['normalized_fitness'].values

print(f"\nDiferenční vnoření úspěšně vytvořena. Finální tvar matice příznaků: {X_diff.shape}")

# Uložení výsledků
np.save('embeddings_diff.npy', X_diff)
np.save('labels.npy', y)
print("\n hotovo!")
print("Soubory 'embeddings_diff.npy' a 'labels.npy' byly úspěšně uloženy.")
print("Nyní můžete na těchto souborech spustit skript 'train_model.py'.")