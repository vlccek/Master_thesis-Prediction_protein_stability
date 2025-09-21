import pandas as pd
import requests
from io import StringIO
from urllib.parse import quote


def fetch_uniprot_sequence(uniprot_id):
    """Fetch the original protein sequence from UniProt"""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse the FASTA file to extract the sequence
        fasta_data = StringIO(response.text)
        header = fasta_data.readline()
        sequence = "".join(line.strip() for line in fasta_data)
        return sequence
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sequence for {uniprot_id}: {e}")
        return None


def process_csv(input_file, output_file):
    """Process the CSV file and add original sequences"""
    df = pd.read_csv(input_file)

    # Get unique UniProt IDs
    uniprot_ids = df['uniprot_ID'].unique()

    # Create a dictionary to store sequences
    sequences = {}

    # Fetch sequences for each UniProt ID
    for uniprot_id in uniprot_ids:
        if pd.isna(uniprot_id):
            continue
        sequence = fetch_uniprot_sequence(uniprot_id)
        if sequence:
            sequences[uniprot_id] = sequence

    # Add original sequence column to dataframe
    df['original_sequence'] = df['uniprot_ID'].map(sequences)

    # Save to new CSV file
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


# Example usage
input_csv = "lehner_dataset.csv"
output_csv = "example_with_sequences.csv"
process_csv(input_csv, output_csv)