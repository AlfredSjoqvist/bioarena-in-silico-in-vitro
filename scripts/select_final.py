#!/usr/bin/env python3
"""
Phase 3: Final Selection for Submission.
Integrates design metrics + monomer validation to select 10 diverse, high-quality binders.

Pipeline:
  1. Load all design metadata (Phase 1 output)
  2. Merge monomer validation results (Phase 2 output)
  3. Hard filter: zero cysteines, monomer pLDDT > 80, RMSD < 2.0
  4. Rank by composite score: 0.6*ipSAE + 0.3*ipTM + 0.1*(pLDDT/100)
  5. Greedy diverse selection with edit distance threshold
  6. Enforce length diversity (at least 2 length buckets)
  7. Output final 10 as FASTA

Usage:
    # Single batch
    python scripts/select_final.py \\
        --batch results/designs/batch_v3_XXXXXXXX.json \\
        --validation results/designs/monomer_validation.json \\
        --output results/submission

    # Multiple batches (gradient + BoltzGen)
    python scripts/select_final.py \\
        --batch results/designs/batch_v3_*.json results/designs/batch_boltzgen_*.json \\
        --output results/submission --skip-validation

    # Without monomer validation
    python scripts/select_final.py \\
        --batch results/designs/batch_v3_XXXXXXXX.json \\
        --output results/submission --skip-validation
"""

import argparse
import glob
import json
import os
from pathlib import Path


def edit_distance(s1: str, s2: str) -> int:
    """Levenshtein edit distance (handles different-length sequences)."""
    if len(s1) == len(s2):
        # Fast path for same-length: Hamming distance
        return sum(a != b for a, b in zip(s1, s2))

    # Full DP edit distance for different lengths
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


IG_MOTIFS = ["GKRFAW", "GRYRCLAL", "GRYRCRAL", "CPFD", "PWTL", "SHPD"]
IG_MOTIFS_REGEX = [r"YRCL?AL", r"GK?RFAW"]


def detect_ig_motifs(sequence: str) -> list[str]:
    """Check if a designed sequence contains Ig-fold framework motifs."""
    import re
    found = [m for m in IG_MOTIFS if m in sequence]
    for pattern in IG_MOTIFS_REGEX:
        match = re.search(pattern, sequence)
        if match and match.group() not in found:
            found.append(match.group())
    return found


def get_ipsae_min(d: dict) -> float:
    """Extract ipSAE_min from design result, computing min(bt, tb) if needed."""
    # First check if ipsae_min is directly available (from updated pipeline)
    if "ipsae_min" in d and d["ipsae_min"] > 0:
        return float(d["ipsae_min"])

    # Try ranking_aux keys
    ipsae_bt = 0.0
    ipsae_tb = 0.0
    ranking_aux = d.get("ranking_aux", {})
    if isinstance(ranking_aux, dict):
        for key in ["BinderTargetIPSAE", "binder_target_ipsae", "IPSAE_bt"]:
            if key in ranking_aux and isinstance(ranking_aux[key], (int, float)):
                ipsae_bt = float(ranking_aux[key])
                break
        for key in ["TargetBinderIPSAE", "target_binder_ipsae", "IPSAE_tb"]:
            if key in ranking_aux and isinstance(ranking_aux[key], (int, float)):
                ipsae_tb = float(ranking_aux[key])
                break

    # Compute min (use min if both are available, otherwise take whatever we have)
    if ipsae_bt > 0 and ipsae_tb > 0:
        return min(ipsae_bt, ipsae_tb)
    elif ipsae_bt > 0:
        return ipsae_bt
    elif ipsae_tb > 0:
        return ipsae_tb

    # Legacy fallback
    for key in ["IPSAE_min", "ipsae_min", "ipsae", "ipSAE"]:
        if key in ranking_aux and isinstance(ranking_aux[key], (int, float)):
            return float(ranking_aux[key])

    if "ipsae" in d:
        return float(d["ipsae"])

    return 0.0


def get_ipsae_asymmetry(d: dict) -> float:
    """Get ipSAE asymmetry (|bt - tb|). Returns 0 if data unavailable."""
    if "ipsae_asymmetry" in d:
        return float(d["ipsae_asymmetry"])

    ipsae_bt = d.get("ipsae_bt", 0)
    ipsae_tb = d.get("ipsae_tb", 0)
    if ipsae_bt > 0 and ipsae_tb > 0:
        return abs(ipsae_bt - ipsae_tb)
    return 0.0


def net_charge(seq: str) -> float:
    """Approximate net charge at pH 7.4."""
    charge = 0.0
    for aa in seq:
        if aa in "RK":
            charge += 1.0
        elif aa in "DE":
            charge -= 1.0
        elif aa == "H":
            charge += 0.1  # ~10% protonated at pH 7.4
    charge += 0.9 - 1.0  # N-terminal, C-terminal contributions
    return charge


def max_homopolymer_run(seq: str) -> int:
    """Length of longest consecutive identical amino acid run."""
    if not seq:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def compute_rank_score(d: dict) -> float:
    """Composite ranking score. Higher is better."""
    ipsae = get_ipsae_min(d)
    iptm = d.get("iptm", 0)
    plddt = d.get("binder_plddt", d.get("plddt_mean", 0))
    asymmetry = get_ipsae_asymmetry(d)

    if ipsae > 0:
        score = 0.6 * ipsae + 0.3 * iptm + 0.1 * (plddt / 100.0)
    elif iptm > 0:
        score = 0.7 * iptm + 0.3 * (plddt / 100.0)
    else:
        # Fallback for BoltzGen designs: use negative ranking_loss (lower loss = better)
        ranking_loss = d.get("ranking_loss", 0)
        if ranking_loss != 0:
            score = -ranking_loss * 0.1
        else:
            score = 0.0

    # Penalize high ipSAE asymmetry (unreliable predictions)
    if asymmetry > 0.25:
        score *= 0.9

    return score


def select_diverse(
    designs: list[dict],
    top_n: int = 10,
    min_edit_dist: int = 10,
    min_length_buckets: int = 2,
) -> list[dict]:
    """
    Greedy diverse selection:
    1. Pick the top design
    2. For each subsequent candidate, check edit distance > threshold from all selected
    3. After initial selection, verify length diversity
    """
    selected = []
    length_buckets = set()

    for d in designs:
        if len(selected) >= top_n:
            break

        seq = d["binder_sequence"]
        is_diverse = True

        for s in selected:
            dist = edit_distance(seq, s["binder_sequence"])
            if dist < min_edit_dist:
                is_diverse = False
                break

        if is_diverse:
            selected.append(d)
            length_buckets.add(d["binder_length"])

    # Check length diversity - if not met, try to swap in candidates from underrepresented lengths
    if len(length_buckets) < min_length_buckets and len(selected) >= top_n:
        all_lengths = set(d["binder_length"] for d in designs)
        missing_lengths = all_lengths - length_buckets

        for missing_len in missing_lengths:
            if len(length_buckets) >= min_length_buckets:
                break
            # Find best candidate of this length
            for d in designs:
                if d["binder_length"] != missing_len or d in selected:
                    continue
                # Check diversity against current selection (minus last)
                is_diverse = True
                for s in selected[:-1]:
                    dist = edit_distance(d["binder_sequence"], s["binder_sequence"])
                    if dist < min_edit_dist:
                        is_diverse = False
                        break
                if is_diverse:
                    # Replace the worst selected design
                    selected[-1] = d
                    length_buckets.add(missing_len)
                    break

    return selected


def main():
    parser = argparse.ArgumentParser(description="Select final 10 binders for submission")
    parser.add_argument("--batch", required=True, nargs="+", help="Batch summary JSON(s) — supports globs")
    parser.add_argument("--validation", default="", help="Monomer validation JSON from Phase 2")
    parser.add_argument("--output", default="results/submission", help="Output directory")
    parser.add_argument("--top", type=int, default=10, help="Number of binders to submit")
    parser.add_argument("--min-edit-dist", type=int, default=15, help="Min edit distance for diversity")
    parser.add_argument("--min-ipsae", type=float, default=0.0, help="Minimum ipSAE threshold (0 to disable)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip monomer validation filter")
    args = parser.parse_args()

    # ── Load design results (supports multiple files and globs) ──
    batch_files = []
    for pattern in args.batch:
        expanded = glob.glob(pattern)
        if expanded:
            batch_files.extend(expanded)
        elif os.path.exists(pattern):
            batch_files.append(pattern)
        else:
            print(f"  WARNING: No files matched: {pattern}")
    batch_files = sorted(set(batch_files))

    if not batch_files:
        print("ERROR: No batch files found!")
        return

    designs = []
    for bf in batch_files:
        print(f"Loading: {bf}")
        with open(bf) as f:
            batch_data = json.load(f)
        if isinstance(batch_data, list):
            designs.extend(batch_data)
        else:
            designs.append(batch_data)
    print(f"  Total designs from {len(batch_files)} file(s): {len(designs)}")

    # ── Load monomer validation ──
    validation = {}
    if args.validation and Path(args.validation).exists():
        print(f"Loading monomer validation: {args.validation}")
        with open(args.validation) as f:
            val_results = json.load(f)
        validation = {r["design_id"]: r for r in val_results}
        print(f"  Validated designs: {len(validation)}")
    elif not args.skip_validation:
        print("  WARNING: No monomer validation results. Use --skip-validation to proceed anyway.")

    # ── Merge validation into designs ──
    for d in designs:
        did = d["design_id"]
        if did in validation:
            d["monomer_plddt"] = validation[did].get("monomer_plddt", 0)
            d["monomer_rmsd"] = validation[did].get("monomer_complex_rmsd")
            d["monomer_passed"] = validation[did].get("passed", False)

    # ── Hard filters ──
    print("\nApplying hard filters...")
    candidates = designs.copy()

    # Filter: zero cysteines
    before = len(candidates)
    candidates = [d for d in candidates if d.get("n_cysteines", 0) == 0]
    print(f"  Cysteine-free: {len(candidates)}/{before}")

    # Filter: no Ig-fold framework motifs
    before = len(candidates)
    for d in candidates:
        d["_ig_motifs"] = detect_ig_motifs(d["binder_sequence"])
    rejected_ig = [d for d in candidates if d["_ig_motifs"]]
    if rejected_ig:
        print(f"  Ig-fold motif rejections:")
        for d in rejected_ig:
            print(f"    {d['design_id']}: {d['_ig_motifs']}")
    candidates = [d for d in candidates if not d["_ig_motifs"]]
    print(f"  Ig-motif-free: {len(candidates)}/{before}")

    # Filter: length range
    before = len(candidates)
    candidates = [d for d in candidates if 60 <= d["binder_length"] <= 150]
    if len(candidates) < before:
        print(f"  Length range (60-150): {len(candidates)}/{before}")

    # Filter: net charge (-5 to +5 for solubility)
    before = len(candidates)
    for d in candidates:
        d["_net_charge"] = net_charge(d["binder_sequence"])
    extreme_charge = [d for d in candidates if abs(d["_net_charge"]) > 5]
    if extreme_charge:
        print(f"  Extreme charge rejections:")
        for d in extreme_charge:
            print(f"    {d['design_id']}: charge={d['_net_charge']:.1f}")
    candidates = [d for d in candidates if abs(d["_net_charge"]) <= 5]
    print(f"  Reasonable charge (-5 to +5): {len(candidates)}/{before}")

    # Filter: homopolymer runs (max 4 consecutive identical AA)
    before = len(candidates)
    for d in candidates:
        d["_max_homopolymer"] = max_homopolymer_run(d["binder_sequence"])
    bad_hp = [d for d in candidates if d["_max_homopolymer"] > 4]
    if bad_hp:
        print(f"  Homopolymer run rejections:")
        for d in bad_hp:
            print(f"    {d['design_id']}: max_run={d['_max_homopolymer']}")
    candidates = [d for d in candidates if d["_max_homopolymer"] <= 4]
    if len(candidates) < before:
        print(f"  Homopolymer run <= 4: {len(candidates)}/{before}")

    # Filter: ipSAE asymmetry (warn but don't reject outright unless very high)
    for d in candidates:
        d["_ipsae_asymmetry"] = get_ipsae_asymmetry(d)
    high_asym = [d for d in candidates if d["_ipsae_asymmetry"] > 0.25]
    if high_asym:
        print(f"  WARNING: {len(high_asym)} designs with ipSAE asymmetry > 0.25:")
        for d in high_asym:
            print(f"    {d['design_id']}: asymmetry={d['_ipsae_asymmetry']:.3f}")

    # Filter: monomer validation (if available)
    if validation and not args.skip_validation:
        before = len(candidates)
        candidates = [d for d in candidates if d.get("monomer_passed", False)]
        print(f"  Monomer validated: {len(candidates)}/{before}")

    # Filter: minimum ipSAE (if set)
    if args.min_ipsae > 0:
        before = len(candidates)
        candidates = [d for d in candidates if get_ipsae_min(d) >= args.min_ipsae]
        print(f"  Above ipSAE threshold ({args.min_ipsae}): {len(candidates)}/{before}")

    if not candidates:
        print("\nERROR: No candidates passed all filters!")
        print("Try: --skip-validation or lower --min-ipsae")
        return

    # ── Rank ──
    for d in candidates:
        d["rank_score"] = compute_rank_score(d)
    candidates.sort(key=lambda d: d["rank_score"], reverse=True)

    print(f"\nTop 20 candidates after filtering:")
    print(f"  {'Rank':<6}{'Design ID':<40}{'Score':<8}{'ipSAE':<8}{'ipTM':<8}{'pLDDT':<8}{'Len':<6}{'Asym':<7}")
    print(f"  {'-'*88}")
    for i, d in enumerate(candidates[:20]):
        print(
            f"  {i+1:<4}  {d['design_id']:<38} "
            f"{d['rank_score']:<8.4f}"
            f"{get_ipsae_min(d):<8.4f}"
            f"{d.get('iptm', 0):<8.4f}"
            f"{d.get('binder_plddt', d.get('plddt_mean', 0)):<8.4f}"
            f"{d['binder_length']:<6}"
            f"{d.get('_ipsae_asymmetry', 0):<7.3f}"
        )

    # ── Diverse selection ──
    print(f"\nSelecting top {args.top} diverse binders (min edit dist={args.min_edit_dist})...")
    selected = select_diverse(
        candidates,
        top_n=args.top,
        min_edit_dist=args.min_edit_dist,
        min_length_buckets=2,
    )

    # ── Final cysteine verification (zero tolerance) ──
    for d in selected:
        if "C" in d["binder_sequence"]:
            print(f"  FATAL: Cysteine in {d['design_id']}! Removing from submission.")
    selected = [d for d in selected if "C" not in d["binder_sequence"]]

    if not selected:
        print("\nERROR: No designs left after cysteine verification!")
        return

    # ── Write output ──
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FASTA submission file
    fasta_path = output_dir / "submission.fasta"
    with open(fasta_path, "w") as f:
        for i, d in enumerate(selected):
            did = d["design_id"]
            iptm = d.get("iptm", 0)
            score = d.get("rank_score", 0)
            ipsae = get_ipsae_min(d)
            length = d["binder_length"]
            f.write(f">{did} rank={i+1} ipTM={iptm:.4f} ipSAE={ipsae:.4f} score={score:.4f} L={length}\n")
            f.write(f"{d['binder_sequence']}\n")
    print(f"\nWrote submission FASTA: {fasta_path}")

    # Summary JSON
    summary_path = output_dir / "submission_summary.json"
    summary = []
    for i, d in enumerate(selected):
        summary.append({
            "rank": i + 1,
            "design_id": d["design_id"],
            "binder_sequence": d["binder_sequence"],
            "binder_length": d["binder_length"],
            "rank_score": d.get("rank_score", 0),
            "ipsae_min": get_ipsae_min(d),
            "ipsae_asymmetry": get_ipsae_asymmetry(d),
            "iptm": d.get("iptm", 0),
            "binder_plddt": d.get("binder_plddt", d.get("plddt_mean", 0)),
            "ranking_loss": d.get("ranking_loss", 0),
            "n_cysteines": d.get("n_cysteines", 0),
            "ig_motifs": d.get("ig_motifs_found", []),
            "monomer_plddt": d.get("monomer_plddt"),
            "monomer_rmsd": d.get("monomer_rmsd"),
        })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote submission summary: {summary_path}")

    # ── Final report ──
    length_buckets = set(d["binder_length"] for d in selected)
    print(f"\n{'='*90}")
    print(f"FINAL SUBMISSION: {len(selected)} binders")
    print(f"Length diversity: {sorted(length_buckets)} ({len(length_buckets)} buckets)")
    print(f"{'='*90}")
    print(f"  {'Rank':<6}{'Design ID':<40}{'Score':<8}{'ipSAE':<8}{'ipTM':<8}{'pLDDT':<8}{'Len':<6}{'Seq'}")
    print(f"  {'-'*90}")
    for i, d in enumerate(selected):
        seq_preview = d["binder_sequence"][:30] + "..." if len(d["binder_sequence"]) > 30 else d["binder_sequence"]
        print(
            f"  {i+1:<4}  {d['design_id']:<38} "
            f"{d.get('rank_score', 0):<8.4f}"
            f"{get_ipsae_min(d):<8.4f}"
            f"{d.get('iptm', 0):<8.4f}"
            f"{d.get('binder_plddt', d.get('plddt_mean', 0)):<8.4f}"
            f"{d['binder_length']:<6}"
            f"{seq_preview}"
        )
    print()


if __name__ == "__main__":
    main()
