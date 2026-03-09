#!/usr/bin/env python3
"""
Score binder designs using ipSAE (DunbrackLab/IPSAE).
Requires the IPSAE tool to be cloned into tools/IPSAE/.

Usage:
    python scripts/score_ipsae.py --input results/designs/ --output results/scores/
    python scripts/score_ipsae.py --pae-file pae.npz --structure-file complex.cif
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

IPSAE_DIR = Path(__file__).parent.parent / "tools" / "IPSAE"
IPSAE_SCRIPT = IPSAE_DIR / "ipsae.py"


def check_ipsae_installed():
    if not IPSAE_SCRIPT.exists():
        print("ERROR: IPSAE not found. Clone it first:")
        print(f"  git clone https://github.com/DunbrackLab/IPSAE.git {IPSAE_DIR}")
        sys.exit(1)


def score_single(pae_file: Path, structure_file: Path, pae_cutoff: int = 10, dist_cutoff: int = 10) -> dict:
    """Run ipSAE scoring on a single prediction."""
    cmd = [
        sys.executable, str(IPSAE_SCRIPT),
        str(pae_file), str(structure_file),
        str(pae_cutoff), str(dist_cutoff),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(IPSAE_DIR))

    if result.returncode != 0:
        print(f"  IPSAE error: {result.stderr}")
        return {}

    # Parse output
    scores = {}
    for line in result.stdout.strip().split("\n"):
        if ":" in line:
            key, val = line.split(":", 1)
            try:
                scores[key.strip()] = float(val.strip())
            except ValueError:
                scores[key.strip()] = val.strip()

    return scores


def score_batch(input_dir: Path, output_dir: Path, pae_cutoff: int = 10, dist_cutoff: int = 10):
    """Score all designs in a directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all structure files with corresponding PAE files
    results = []
    structure_files = list(input_dir.glob("*.pdb")) + list(input_dir.glob("*.cif"))

    for struct_file in sorted(structure_files):
        stem = struct_file.stem
        # Look for corresponding PAE file
        pae_candidates = [
            input_dir / f"{stem}_pae.json",
            input_dir / f"pae_{stem}.npz",
            input_dir / f"{stem}.npz",
            input_dir / f"scores_{stem}.json",
        ]
        pae_file = None
        for candidate in pae_candidates:
            if candidate.exists():
                pae_file = candidate
                break

        if pae_file is None:
            print(f"  Skipping {stem}: no PAE file found")
            continue

        print(f"  Scoring {stem}...")
        scores = score_single(pae_file, struct_file, pae_cutoff, dist_cutoff)

        if scores:
            scores["design_id"] = stem
            scores["structure_file"] = str(struct_file)
            scores["pae_file"] = str(pae_file)
            results.append(scores)

            # Save individual score
            score_path = output_dir / f"{stem}_scores.json"
            with open(score_path, "w") as f:
                json.dump(scores, f, indent=2)

    if not results:
        print("\nNo designs could be scored.")
        print("Note: ipSAE requires PAE files (JSON for AF2/AF3, NPZ for Boltz).")
        print("If using Mosaic/Boltz-2, the PAE is embedded in the prediction output.")
        return results

    # Sort by ipSAE (descending)
    ipsae_key = "ipSAE" if "ipSAE" in results[0] else list(results[0].keys())[0]
    results.sort(key=lambda r: r.get(ipsae_key, 0), reverse=True)

    # Save summary CSV
    csv_path = output_dir / "scores_summary.csv"
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved summary: {csv_path}")

    # Save summary JSON
    json_path = output_dir / "scores_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON: {json_path}")

    # Print leaderboard
    print(f"\n{'='*70}")
    print(f"{'Rank':<6}{'Design':<40}{'ipSAE':<10}{'ipTM':<10}")
    print(f"{'='*70}")
    for i, r in enumerate(results[:20]):
        print(f"  {i+1:<4} {r.get('design_id', 'N/A'):<38} "
              f"{r.get('ipSAE', 'N/A'):<10} {r.get('ipTM_af', 'N/A'):<10}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Score designs using ipSAE")
    parser.add_argument("--input", type=str, default="results/designs", help="Input directory with PDB/CIF + PAE files")
    parser.add_argument("--output", type=str, default="results/scores", help="Output directory for scores")
    parser.add_argument("--pae-cutoff", type=int, default=10, help="PAE cutoff for ipSAE")
    parser.add_argument("--dist-cutoff", type=int, default=10, help="Distance cutoff for ipSAE")
    # Single file mode
    parser.add_argument("--pae-file", type=str, help="Single PAE file to score")
    parser.add_argument("--structure-file", type=str, help="Single structure file to score")
    args = parser.parse_args()

    check_ipsae_installed()

    if args.pae_file and args.structure_file:
        scores = score_single(
            Path(args.pae_file), Path(args.structure_file),
            args.pae_cutoff, args.dist_cutoff,
        )
        print(json.dumps(scores, indent=2))
    else:
        score_batch(
            Path(args.input), Path(args.output),
            args.pae_cutoff, args.dist_cutoff,
        )


if __name__ == "__main__":
    main()
