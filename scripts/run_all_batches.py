#!/usr/bin/env python3
"""
Multi-batch orchestrator for TREM2 binder design.
Launches multiple parallel batches with different loss variants to maximize diversity.

Variant strategies:
  default:        HelixLoss=0.3, balanced baseline
  contact_heavy:  BinderTargetContact=2.0, larger interfaces
  helix_biased:   HelixLoss=2.0, aggressive anti-Ig-fold
  compact:        +DistogramRadiusOfGyration, tighter cores
  no_helix:       HelixLoss=0.0, maximum topology freedom

Usage:
    # Full run (~35 designs, ~$70, ~35 min)
    modal run scripts/run_all_batches.py

    # Quick test (12 designs, ~$24)
    modal run scripts/run_all_batches.py --quick

    # Budget cap
    modal run scripts/run_all_batches.py --budget 50

    # Wave 2 only (fill gaps after reviewing Wave 1)
    modal run scripts/run_all_batches.py --wave2-only
"""

import json
import os
import sys
import time

# Add scripts directory to path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modal

# Import the existing design function and Modal resources from design_and_rank.py
from design_and_rank import app, design_single_binder, results_volume, VARIANT_CONFIGS

# ── Batch Definitions ──

BATCHES_FULL = [
    {
        "name": "A_default",
        "description": "Balanced baseline: HelixLoss=0.3",
        "variant": "default",
        "lengths": [70, 80, 85],
        "n_per_length": 5,
        "seed_offset": 0,
    },
    {
        "name": "B_contact_heavy",
        "description": "Doubled contact weight for larger interfaces",
        "variant": "contact_heavy",
        "lengths": [70, 80],
        "n_per_length": 4,
        "seed_offset": 1000,
    },
    {
        "name": "C_helix_biased",
        "description": "Aggressive anti-Ig: HelixLoss=2.0",
        "variant": "helix_biased",
        "lengths": [80, 85],
        "n_per_length": 3,
        "seed_offset": 2000,
    },
    {
        "name": "D_compact",
        "description": "Compact binders with radius of gyration loss",
        "variant": "compact",
        "lengths": [70, 75],
        "n_per_length": 3,
        "seed_offset": 3000,
    },
]

BATCHES_QUICK = [
    {
        "name": "Q_quick_test",
        "description": "Quick test: 3 lengths, default variant",
        "variant": "default",
        "lengths": [70, 80, 85],
        "n_per_length": 4,
        "seed_offset": 0,
    },
]

BATCHES_WAVE2 = [
    {
        "name": "W2_no_helix",
        "description": "Wave 2: maximum topology freedom",
        "variant": "no_helix",
        "lengths": [70, 80, 85],
        "n_per_length": 4,
        "seed_offset": 4000,
    },
    {
        "name": "W2_contact_short",
        "description": "Wave 2: contact-heavy at short lengths",
        "variant": "contact_heavy",
        "lengths": [65, 75],
        "n_per_length": 4,
        "seed_offset": 5000,
    },
]

# Seed sub-offsets by length (avoid collisions within a batch)
LENGTH_SEED_OFFSETS = {
    65: 0, 70: 50, 75: 100, 80: 150, 85: 200, 90: 250, 100: 300, 120: 400,
}

COST_PER_DESIGN = 2.0  # ~$2 on A100 (30 min at $4/hr)


def flatten_batches(batches: list[dict]) -> list[dict]:
    """Flatten batch definitions into individual design configs."""
    designs = []
    for batch in batches:
        for length in batch["lengths"]:
            base_seed = batch["seed_offset"] + LENGTH_SEED_OFFSETS.get(length, length * 5)
            for i in range(batch["n_per_length"]):
                designs.append({
                    "binder_length": length,
                    "seed": base_seed + i,
                    "helix_loss_weight": batch["helix_loss_weight"],
                    "batch_name": batch["name"],
                })
    return designs


def trim_to_budget(designs: list[dict], budget: float) -> list[dict]:
    """Trim designs to fit within budget, keeping proportional representation."""
    max_designs = int(budget / COST_PER_DESIGN)
    if len(designs) <= max_designs:
        return designs
    # Keep proportional representation from each batch
    batch_names = list(dict.fromkeys(d["batch_name"] for d in designs))
    per_batch = max(1, max_designs // len(batch_names))
    trimmed = []
    for batch_name in batch_names:
        batch_designs = [d for d in designs if d["batch_name"] == batch_name]
        trimmed.extend(batch_designs[:per_batch])
    return trimmed[:max_designs]


@app.local_entrypoint()
def main(
    quick: bool = False,
    wave2_only: bool = False,
    budget: float = 150.0,
):
    """Launch multi-strategy binder design batches."""

    if quick:
        batches = BATCHES_QUICK
        mode = "QUICK TEST"
    elif wave2_only:
        batches = BATCHES_WAVE2
        mode = "WAVE 2 (GAP FILL)"
    else:
        batches = BATCHES_FULL
        mode = "FULL RUN"

    # Flatten to individual designs
    all_designs = flatten_batches(batches)

    # Budget check
    estimated_cost = len(all_designs) * COST_PER_DESIGN
    if estimated_cost > budget:
        print(f"WARNING: {len(all_designs)} designs (${estimated_cost:.0f}) exceeds budget (${budget:.0f})")
        all_designs = trim_to_budget(all_designs, budget)
        print(f"  Trimmed to {len(all_designs)} designs (${len(all_designs) * COST_PER_DESIGN:.0f})")

    total = len(all_designs)

    # Print launch summary
    print(f"\n{'='*80}")
    print(f"TREM2 MULTI-BATCH DESIGN: {mode}")
    print(f"{'='*80}")
    for batch in batches:
        n = sum(1 for d in all_designs if d["batch_name"] == batch["name"])
        print(f"  {batch['name']}: {n} designs, lengths={batch['lengths']}, "
              f"HelixLoss={batch['helix_loss_weight']}")
    print(f"\n  Total designs:    {total}")
    print(f"  Estimated cost:   ~${total * COST_PER_DESIGN:.0f}")
    print(f"  Estimated time:   ~35 min (parallel on Modal A100s)")
    print(f"{'='*80}\n")

    # Launch ALL designs in parallel via Modal .map()
    # We pass all parameters as positional iterables to support per-design helix weights
    t_start = time.time()
    results = []
    for result in design_single_binder.map(
        [d["binder_length"] for d in all_designs],           # binder_length
        [d["seed"] for d in all_designs],                    # seed
        [100] * total,                                       # stage1_steps
        [50] * total,                                        # stage2_steps
        [15] * total,                                        # stage3_steps
        [4] * total,                                         # design_samples
        [1] * total,                                         # design_recycling
        [6] * total,                                         # ranking_samples
        [3] * total,                                         # ranking_recycling
        [d["helix_loss_weight"] for d in all_designs],       # helix_loss_weight
    ):
        idx = len(results)
        batch_name = all_designs[idx]["batch_name"]
        results.append(result)
        result["batch_name"] = batch_name

        cys_flag = " CYS!" if result["n_cysteines"] > 0 else ""
        ig_flag = " IG!" if result.get("ig_motifs_found") else ""
        print(
            f"  [{len(results)}/{total}] {result['design_id']} "
            f"({batch_name}) "
            f"| ipTM={result['iptm']:.3f} "
            f"| ipSAE={result.get('ipsae_min', 0):.3f} "
            f"| pLDDT={result['binder_plddt']:.1f}"
            f"{cys_flag}{ig_flag}"
        )

    elapsed = time.time() - t_start

    # ── Per-batch statistics ──
    print(f"\n{'='*80}")
    print(f"ALL {len(results)} DESIGNS COMPLETE in {elapsed/60:.1f} min")
    print(f"{'='*80}")

    for batch in batches:
        batch_results = [r for r in results if r.get("batch_name") == batch["name"]]
        if not batch_results:
            continue
        n_cys_free = sum(1 for r in batch_results if r["n_cysteines"] == 0)
        n_ig_free = sum(1 for r in batch_results if not r.get("ig_motifs_found"))
        avg_ipsae = sum(r.get("ipsae_min", 0) for r in batch_results) / len(batch_results)
        avg_iptm = sum(r.get("iptm", 0) for r in batch_results) / len(batch_results)
        print(f"\n  {batch['name']} ({len(batch_results)} designs, HelixLoss={batch['helix_loss_weight']}):")
        print(f"    Cys-free: {n_cys_free}/{len(batch_results)}, Ig-free: {n_ig_free}/{len(batch_results)}")
        print(f"    Avg ipSAE: {avg_ipsae:.3f}, Avg ipTM: {avg_iptm:.3f}")

    # ── Unified leaderboard ──
    results.sort(key=lambda r: r.get("ipsae_min", 0), reverse=True)

    print(f"\n{'='*100}")
    print(f"UNIFIED LEADERBOARD (sorted by ipSAE_min)")
    print(f"{'='*100}")
    print(f"{'#':<4}{'Design ID':<40}{'Batch':<22}{'ipSAE':<8}{'ipTM':<8}{'pLDDT':<7}{'Len':<5}{'Cys':<5}{'Ig':<6}")
    print(f"{'-'*95}")
    for i, r in enumerate(results[:30]):
        ig_flag = "YES" if r.get("ig_motifs_found") else "-"
        print(
            f"{i+1:<4}{r['design_id']:<40}"
            f"{r.get('batch_name', '?'):<22}"
            f"{r.get('ipsae_min', 0):<8.3f}"
            f"{r['iptm']:<8.3f}"
            f"{r['binder_plddt']:<7.1f}"
            f"{r['binder_length']:<5}"
            f"{r['n_cysteines']:<5}"
            f"{ig_flag}"
        )

    # ── Save merged results ──
    os.makedirs("results/designs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    merged_path = f"results/designs/all_batches_merged_{ts}.json"
    with open(merged_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved merged results: {merged_path}")

    # Save FASTA
    fasta_path = f"results/designs/all_batches_{ts}.fasta"
    with open(fasta_path, "w") as f:
        for i, r in enumerate(results):
            f.write(
                f">{r['design_id']} rank={i+1} "
                f"ipSAE={r.get('ipsae_min', 0):.4f} "
                f"ipTM={r['iptm']:.4f} "
                f"batch={r.get('batch_name', '?')} "
                f"cys={r['n_cysteines']}\n"
            )
            f.write(f"{r['binder_sequence']}\n")
    print(f"Saved FASTA: {fasta_path}")

    # ── Quality summary ──
    n_cys_free = sum(1 for r in results if r["n_cysteines"] == 0)
    n_ig_free = sum(1 for r in results if not r.get("ig_motifs_found"))
    n_good_ipsae = sum(1 for r in results if r.get("ipsae_min", 0) > 0.5)
    n_great_ipsae = sum(1 for r in results if r.get("ipsae_min", 0) > 0.7)
    lengths_seen = sorted(set(r["binder_length"] for r in results))

    print(f"\n{'='*60}")
    print(f"QUALITY SUMMARY:")
    print(f"  Total designs:       {len(results)}")
    print(f"  Cysteine-free:       {n_cys_free}/{len(results)}")
    print(f"  Ig-motif-free:       {n_ig_free}/{len(results)}")
    print(f"  ipSAE > 0.5:         {n_good_ipsae}/{len(results)}")
    print(f"  ipSAE > 0.7:         {n_great_ipsae}/{len(results)}")
    print(f"  Lengths represented: {lengths_seen}")
    print(f"  Elapsed time:        {elapsed/60:.1f} min")
    print(f"  Estimated cost:      ~${len(results) * COST_PER_DESIGN:.0f}")
    print(f"{'='*60}")

    # ── Next steps ──
    print(f"\nNEXT STEPS:")
    print(f"  1. Validate monomers:")
    print(f"     modal run scripts/validate_monomers.py --batch {merged_path} --top 20")
    print(f"  2. Select final 10:")
    print(f"     python scripts/select_final.py --batch {merged_path} "
          f"--validation results/designs/monomer_validation.json --output results/submission")
    print(f"  3. Submit: upload results/submission/submission.fasta to bioArena.xyz")
