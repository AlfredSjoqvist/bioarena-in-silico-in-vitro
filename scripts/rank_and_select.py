#!/usr/bin/env python3
"""
Rank all designs and select the top N for submission.
Combines ipSAE scores, ipTM, and pLDDT for final ranking.

Usage:
    python scripts/rank_and_select.py --scores results/scores/ --top 10
    python scripts/rank_and_select.py --designs results/designs/ --top 10
"""

import argparse
import json
import shutil
from pathlib import Path


def load_design_metadata(designs_dir: Path) -> list[dict]:
    """Load all design metadata JSON files."""
    results = []
    for json_file in sorted(designs_dir.glob("*.json")):
        if "batch_summary" in json_file.name or "scores" in json_file.name:
            continue
        with open(json_file) as f:
            data = json.load(f)
            results.append(data)
    return results


def load_ipsae_scores(scores_dir: Path) -> dict:
    """Load ipSAE scores and index by design_id."""
    scores = {}
    for json_file in sorted(scores_dir.glob("*_scores.json")):
        with open(json_file) as f:
            data = json.load(f)
            design_id = data.get("design_id", json_file.stem.replace("_scores", ""))
            scores[design_id] = data
    return scores


def rank_designs(designs: list[dict], ipsae_scores: dict = None) -> list[dict]:
    """Rank designs by combined metric."""
    for d in designs:
        design_id = d.get("design_id", "")

        # Get ipSAE score if available
        ipsae = 0.0
        if ipsae_scores and design_id in ipsae_scores:
            ipsae = float(ipsae_scores[design_id].get("ipSAE", 0))
            d["ipsae"] = ipsae

        # Combined ranking score:
        # Primary: ipSAE (if available), otherwise ipTM
        # Secondary: pLDDT as tiebreaker
        iptm = d.get("iptm", 0)
        plddt = d.get("plddt_mean", 0)

        if ipsae > 0:
            # Weighted combination: 60% ipSAE + 30% ipTM + 10% pLDDT
            d["rank_score"] = 0.6 * ipsae + 0.3 * iptm + 0.1 * (plddt / 100.0)
        else:
            # Without ipSAE, use ipTM + pLDDT
            d["rank_score"] = 0.7 * iptm + 0.3 * (plddt / 100.0)

    designs.sort(key=lambda d: d["rank_score"], reverse=True)
    return designs


def select_diverse_top(designs: list[dict], top_n: int = 10, min_edit_dist: int = 10) -> list[dict]:
    """
    Select top N designs ensuring sequence diversity.
    Avoids submitting near-identical sequences.
    """
    selected = []

    for d in designs:
        if len(selected) >= top_n:
            break

        seq = d.get("binder_sequence", "")
        is_diverse = True

        for s in selected:
            existing_seq = s.get("binder_sequence", "")
            if len(seq) == len(existing_seq):
                edit_dist = sum(a != b for a, b in zip(seq, existing_seq))
                if edit_dist < min_edit_dist:
                    is_diverse = False
                    break

        if is_diverse:
            selected.append(d)

    return selected


def write_submission(selected: list[dict], output_dir: Path, designs_dir: Path = None):
    """Write final submission files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined FASTA
    fasta_path = output_dir / "submission.fasta"
    with open(fasta_path, "w") as f:
        for i, d in enumerate(selected):
            design_id = d.get("design_id", f"design_{i}")
            seq = d.get("binder_sequence", "")
            iptm = d.get("iptm", 0)
            ipsae = d.get("ipsae", 0)
            f.write(f">{design_id} rank={i+1} ipTM={iptm:.4f} ipSAE={ipsae:.4f}\n")
            f.write(f"{seq}\n")
    print(f"Wrote submission FASTA: {fasta_path}")

    # Summary JSON
    summary_path = output_dir / "submission_summary.json"
    with open(summary_path, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"Wrote submission summary: {summary_path}")

    # Copy PDB files if available
    if designs_dir:
        pdb_dir = output_dir / "structures"
        pdb_dir.mkdir(exist_ok=True)
        for d in selected:
            design_id = d.get("design_id", "")
            pdb_src = designs_dir / f"{design_id}.pdb"
            if pdb_src.exists():
                shutil.copy2(pdb_src, pdb_dir / f"{design_id}.pdb")

    # Print final selection
    print(f"\n{'='*80}")
    print(f"FINAL SUBMISSION: {len(selected)} binders")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Design ID':<45}{'ipSAE':<10}{'ipTM':<10}{'pLDDT':<10}{'Score':<10}")
    print(f"{'-'*80}")
    for i, d in enumerate(selected):
        print(
            f"  {i+1:<4} {d.get('design_id', 'N/A'):<43} "
            f"{d.get('ipsae', 0):<10.4f}"
            f"{d.get('iptm', 0):<10.4f}"
            f"{d.get('plddt_mean', 0):<10.1f}"
            f"{d.get('rank_score', 0):<10.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Rank designs and select top N for submission")
    parser.add_argument("--designs", type=str, default="results/designs", help="Designs directory")
    parser.add_argument("--scores", type=str, default="results/scores", help="Scores directory")
    parser.add_argument("--top", type=int, default=10, help="Number of designs to submit")
    parser.add_argument("--output", type=str, default="results/submissions", help="Output directory")
    parser.add_argument("--min-edit-dist", type=int, default=10, help="Minimum edit distance for diversity")
    args = parser.parse_args()

    designs_dir = Path(args.designs)
    scores_dir = Path(args.scores)
    output_dir = Path(args.output)

    # Load data
    print("Loading designs...")
    designs = load_design_metadata(designs_dir)
    print(f"  Found {len(designs)} designs")

    ipsae_scores = {}
    if scores_dir.exists():
        print("Loading ipSAE scores...")
        ipsae_scores = load_ipsae_scores(scores_dir)
        print(f"  Found {len(ipsae_scores)} scored designs")

    # Rank
    print("\nRanking designs...")
    ranked = rank_designs(designs, ipsae_scores)

    # Select diverse top N
    print(f"Selecting top {args.top} diverse designs...")
    selected = select_diverse_top(ranked, top_n=args.top, min_edit_dist=args.min_edit_dist)

    # Write submission
    write_submission(selected, output_dir, designs_dir)


if __name__ == "__main__":
    main()
