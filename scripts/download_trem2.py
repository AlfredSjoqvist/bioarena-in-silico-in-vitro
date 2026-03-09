#!/usr/bin/env python3
"""Download TREM2 PDB structure and prepare target files."""

import urllib.request
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "trem2"

# TREM2 ectodomain sequence (residues 19-131, UniProt Q9NZC2)
TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)

PDB_FILES = {
    "5UD7": "https://files.rcsb.org/download/5UD7.pdb",  # WT apo (2.2 Å)
    "6YYE": "https://files.rcsb.org/download/6YYE.pdb",  # TREM2 + scFv complex
}


def download_pdb(pdb_id: str, url: str, output_dir: Path):
    output_path = output_dir / f"{pdb_id}.pdb"
    if output_path.exists():
        print(f"  {pdb_id}.pdb already exists, skipping")
        return output_path

    print(f"  Downloading {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, output_path)
    print(f"  Saved to {output_path}")
    return output_path


def extract_chain_a(pdb_path: Path, output_path: Path):
    """Extract chain A from PDB file (TREM2 monomer)."""
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain = line[21]
                if chain == "A":
                    lines.append(line)
            elif line.startswith("END"):
                lines.append(line)
                break

    with open(output_path, "w") as f:
        f.writelines(lines)
    print(f"  Extracted chain A -> {output_path}")


def write_fasta(sequence: str, name: str, output_path: Path):
    with open(output_path, "w") as f:
        f.write(f">{name}\n{sequence}\n")
    print(f"  Wrote FASTA -> {output_path}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Downloading TREM2 structures ===\n")

    for pdb_id, url in PDB_FILES.items():
        pdb_path = download_pdb(pdb_id, url, DATA_DIR)

    # Extract chain A from 5UD7 (the monomer we'll use as target)
    print("\n=== Extracting TREM2 monomer (chain A from 5UD7) ===")
    extract_chain_a(
        DATA_DIR / "5UD7.pdb",
        DATA_DIR / "trem2_chainA.pdb",
    )

    # Write FASTA sequence
    print("\n=== Writing TREM2 sequence ===")
    write_fasta(
        TREM2_SEQUENCE,
        "TREM2_ectodomain_19-131",
        DATA_DIR / "trem2.fasta",
    )

    # Write a simple text file with the sequence for quick reference
    seq_path = DATA_DIR / "trem2_sequence.txt"
    with open(seq_path, "w") as f:
        f.write(TREM2_SEQUENCE)
    print(f"  Wrote sequence -> {seq_path}")

    print(f"\n=== Done! Files in {DATA_DIR} ===")
    for p in sorted(DATA_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
