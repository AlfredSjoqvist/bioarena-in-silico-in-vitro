#!/usr/bin/env python3
"""
TURBO Design Launcher: Maximum diversity TREM2 binder generation.

Launches ~40 designs across multiple length/seed/loss-weight variants in a single
Modal batch. All designs run in parallel (~30 min wall clock, ~$80-120).

Variants:
  - Standard (16 designs): Baseline with HelixLoss=2.5, lengths 70/75/80/85
  - Contact-heavy (8 designs): BinderTargetContact=2.0, lengths 75/80
  - Helix-strong (8 designs): HelixLoss=3.5, lengths 70/80
  - Explorer (8 designs): HelixLoss=2.0, lengths 75/85 (more freedom)

Usage:
    # Full turbo batch (~40 designs, ~30 min, ~$80-120)
    modal run scripts/design_turbo.py

    # Smoke test (1 design per variant = 4 total)
    modal run scripts/design_turbo.py --smoke-test

    # Custom design count
    modal run scripts/design_turbo.py --n-standard 20 --n-contact 10
"""

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("trem2-turbo-design")

# Container image with Mosaic + all dependencies
mosaic_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/escalante-bio/mosaic.git /opt/mosaic",
        "cd /opt/mosaic && uv sync --group jax-cuda",
    )
    .env({
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    })
)

results_volume = modal.Volume.from_name("trem2-results-v2", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


@app.function(
    gpu="A100",
    image=mosaic_image,
    timeout=3600,
    volumes={"/results": results_volume},
    retries=1,
)
def design_binder_variant(
    binder_length: int,
    seed: int,
    variant: str = "standard",
    helix_weight: float = 2.5,
    contact_weight: float = 1.0,
    stage1_steps: int = 100,
    stage2_steps: int = 50,
    stage3_steps: int = 15,
    design_samples: int = 4,
    design_recycling: int = 1,
    ranking_samples: int = 6,
    ranking_recycling: int = 3,
) -> dict:
    """Design a single TREM2 binder with variant-specific loss weights."""
    import sys
    sys.path.insert(0, "/opt/mosaic/src")
    sys.path.insert(0, "/opt/mosaic")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain
    from mosaic.optimizers import simplex_APGM
    from mosaic.common import TOKENS
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    import mosaic.losses.structure_prediction as sp

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    design_id = f"trem2_{variant}_L{binder_length}_s{seed}_{timestamp}"
    print(f"\n{'='*60}")
    print(f"DESIGN: {design_id}")
    print(f"Variant={variant}, Length={binder_length}, Seed={seed}")
    print(f"HelixLoss={helix_weight}, ContactWeight={contact_weight}")
    print(f"{'='*60}")

    t_start = time.time()

    # ── Load models ──
    print("Loading Boltz-2...")
    model = Boltz2()
    print("Loading ProteinMPNN...")
    mpnn = ProteinMPNN.from_pretrained()
    print(f"Models loaded in {time.time() - t_start:.1f}s")

    # ── Generate features ──
    print("Generating features...")
    features, structure_writer = model.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(TREM2_SEQUENCE, use_msa=False)],
    )

    # ── Cysteine bias ──
    cys_idx = TOKENS.index("C")
    cys_bias = np.zeros((binder_length, 20))
    cys_bias[:, cys_idx] = -1e6

    # ── Build variant-specific design loss ──
    print(f"Building design loss (variant={variant})...")
    design_loss = model.build_multisample_loss(
        loss=(
            contact_weight * sp.BinderTargetContact()
            + 1.0 * sp.WithinBinderContact()
            + 10.0 * InverseFoldingSequenceRecovery(
                mpnn,
                temp=jax.numpy.array(0.001),
                bias=jnp.array(cys_bias),
            )
            + helix_weight * sp.HelixLoss()
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 0.025 * sp.pTMEnergy()
            + 0.1 * sp.PLDDTLoss()
        ),
        features=features,
        recycling_steps=design_recycling,
        num_samples=design_samples,
    )

    # ── Stage 1: Soft PSSM ──
    print(f"\n── Stage 1: Soft PSSM ({stage1_steps} steps) ──")
    t0 = time.time()
    _, PSSM = simplex_APGM(
        loss_function=design_loss,
        n_steps=stage1_steps,
        x=jax.nn.softmax(
            0.5 * jax.random.gumbel(
                key=jax.random.key(seed),
                shape=(binder_length, 20),
            )
        ),
        stepsize=0.2 * np.sqrt(binder_length),
        momentum=0.3,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Stage 2: Sharpening ──
    print(f"\n── Stage 2: Sharpening ({stage2_steps} steps) ──")
    t0 = time.time()
    _, PSSM_sharp = simplex_APGM(
        loss_function=design_loss,
        n_steps=stage2_steps,
        x=PSSM,
        stepsize=0.5 * np.sqrt(binder_length),
        scale=1.25,
        logspace=True,
        momentum=0.0,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Stage 3: Final sharpening ──
    print(f"\n── Stage 3: Final sharpening ({stage3_steps} steps) ──")
    t0 = time.time()
    _, PSSM_final = simplex_APGM(
        loss_function=design_loss,
        n_steps=stage3_steps,
        x=PSSM_sharp,
        stepsize=0.5 * np.sqrt(binder_length),
        scale=1.4,
        logspace=True,
        momentum=0.0,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Stage 4: Extract sequence (with hard cysteine clamp) ──
    PSSM_final = PSSM_final.at[:, cys_idx].set(-float("inf"))
    binder_sequence = "".join(TOKENS[i] for i in PSSM_final.argmax(-1))
    n_cys = binder_sequence.count("C")
    print(f"\nDesigned sequence ({len(binder_sequence)} AA, {n_cys} Cys): {binder_sequence}")

    # ── Ranking refold ──
    print(f"\n── Stage 4: Ranking refold ({ranking_samples} samples, {ranking_recycling} recycling) ──")
    t0 = time.time()

    ranking_loss = model.build_multisample_loss(
        loss=(
            1.0 * sp.IPTMLoss()
            + 0.5 * sp.TargetBinderIPSAE()
            + 0.5 * sp.BinderTargetIPSAE()
        ),
        features=features,
        recycling_steps=ranking_recycling,
        num_samples=ranking_samples,
    )

    one_hot_seq = jax.nn.one_hot(
        jnp.array([TOKENS.index(c) for c in binder_sequence]), 20
    )

    ranking_score, ranking_aux = ranking_loss(one_hot_seq, key=jax.random.key(seed + 1000))
    ranking_score = float(ranking_score)

    prediction = model.predict(
        PSSM=one_hot_seq,
        features=features,
        writer=structure_writer,
        recycling_steps=ranking_recycling,
        key=jax.random.key(seed + 2000),
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Extract metrics ──
    iptm = float(prediction.iptm)
    plddt_mean = float(prediction.plddt.mean())
    binder_plddt = float(prediction.plddt[:binder_length].mean())
    pae_matrix = np.array(prediction.pae)

    ranking_aux_flat = {}
    for k, v in ranking_aux.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                try:
                    ranking_aux_flat[k2] = float(v2) if not hasattr(v2, '__len__') else [float(x) for x in v2]
                except (TypeError, ValueError):
                    pass
        else:
            try:
                ranking_aux_flat[k] = float(v) if not hasattr(v, '__len__') else [float(x) for x in v]
            except (TypeError, ValueError):
                pass

    # ipSAE extraction
    ipsae_bt = 0.0
    ipsae_tb = 0.0
    for key in ["BinderTargetIPSAE", "binder_target_ipsae", "IPSAE_bt"]:
        if key in ranking_aux_flat and isinstance(ranking_aux_flat[key], (int, float)):
            ipsae_bt = float(ranking_aux_flat[key])
            break
    for key in ["TargetBinderIPSAE", "target_binder_ipsae", "IPSAE_tb"]:
        if key in ranking_aux_flat and isinstance(ranking_aux_flat[key], (int, float)):
            ipsae_tb = float(ranking_aux_flat[key])
            break
    ipsae_min = min(ipsae_bt, ipsae_tb) if (ipsae_bt > 0 and ipsae_tb > 0) else max(ipsae_bt, ipsae_tb)
    ipsae_asymmetry = abs(ipsae_bt - ipsae_tb)

    # Ig-motif check
    ig_motifs_found = [m for m in ["GKRFAW", "GRYRCLAL", "CPFD", "PWTL", "SHPD"] if m in binder_sequence]

    total_time = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"RESULTS: {design_id}")
    print(f"  ipTM={iptm:.4f}  pLDDT={binder_plddt:.4f}  ipSAE_min={ipsae_min:.4f}")
    print(f"  Cys={n_cys}  IgMotifs={ig_motifs_found or 'None'}  Time={total_time:.0f}s")
    print(f"{'='*60}")

    # ── Save outputs ──
    out_dir = f"/results/{design_id}"
    os.makedirs(out_dir, exist_ok=True)

    pdb_str = prediction.st.make_pdb_string()
    with open(f"{out_dir}/{design_id}.pdb", "w") as f:
        f.write(pdb_str)

    prediction.st.write_minimal_cif(f"{out_dir}/{design_id}.cif")
    np.savez(f"{out_dir}/{design_id}_pae.npz", pae=pae_matrix)

    with open(f"{out_dir}/{design_id}.fasta", "w") as f:
        f.write(f">{design_id} ipTM={iptm:.4f} ipSAE_min={ipsae_min:.4f} variant={variant}\n")
        f.write(f"{binder_sequence}\n")

    result = {
        "design_id": design_id,
        "variant": variant,
        "target": "TREM2",
        "target_sequence": TREM2_SEQUENCE,
        "binder_sequence": binder_sequence,
        "binder_length": binder_length,
        "seed": seed,
        "iptm": iptm,
        "plddt_mean": plddt_mean,
        "binder_plddt": binder_plddt,
        "ranking_loss": ranking_score,
        "ranking_aux": ranking_aux_flat,
        "ipsae_bt": ipsae_bt,
        "ipsae_tb": ipsae_tb,
        "ipsae_min": ipsae_min,
        "ipsae_asymmetry": ipsae_asymmetry,
        "n_cysteines": n_cys,
        "ig_motifs_found": ig_motifs_found,
        "total_time_seconds": total_time,
        "params": {
            "variant": variant,
            "helix_weight": helix_weight,
            "contact_weight": contact_weight,
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "stage3_steps": stage3_steps,
            "design_samples": design_samples,
            "ranking_samples": ranking_samples,
            "ranking_recycling": ranking_recycling,
        },
    }
    with open(f"{out_dir}/{design_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    results_volume.commit()
    return result


# ═══════════════════════════════════════════════════════════════
# Design Variants
# ═══════════════════════════════════════════════════════════════

VARIANTS = {
    "standard": {
        "helix_weight": 2.5,
        "contact_weight": 1.0,
        "lengths": [70, 75, 80, 85],
        "seeds_per_length": 4,
        "seed_base": 0,
    },
    "contact": {
        "helix_weight": 2.5,
        "contact_weight": 2.0,
        "lengths": [75, 80],
        "seeds_per_length": 4,
        "seed_base": 1000,
    },
    "helix_strong": {
        "helix_weight": 3.5,
        "contact_weight": 1.0,
        "lengths": [70, 80],
        "seeds_per_length": 4,
        "seed_base": 2000,
    },
    "explorer": {
        "helix_weight": 2.0,
        "contact_weight": 1.0,
        "lengths": [75, 85],
        "seeds_per_length": 4,
        "seed_base": 3000,
    },
}


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    n_standard: int = 16,
    n_contact: int = 8,
    n_helix_strong: int = 8,
    n_explorer: int = 8,
):
    """Launch full turbo batch with multiple loss variants."""

    if smoke_test:
        print("TURBO SMOKE TEST: 1 design per variant (4 total)")
        designs = []
        for variant_name, cfg in VARIANTS.items():
            designs.append({
                "binder_length": cfg["lengths"][0],
                "seed": cfg["seed_base"],
                "variant": variant_name,
                "helix_weight": cfg["helix_weight"],
                "contact_weight": cfg["contact_weight"],
                "stage1_steps": 20,
                "stage2_steps": 10,
                "stage3_steps": 5,
                "design_samples": 2,
                "ranking_samples": 4,
                "ranking_recycling": 2,
            })
    else:
        # Build full design matrix
        designs = []
        variant_counts = {
            "standard": n_standard,
            "contact": n_contact,
            "helix_strong": n_helix_strong,
            "explorer": n_explorer,
        }

        for variant_name, cfg in VARIANTS.items():
            target_count = variant_counts[variant_name]
            lengths = cfg["lengths"]
            per_length = max(1, target_count // len(lengths))

            for length in lengths:
                for i in range(per_length):
                    designs.append({
                        "binder_length": length,
                        "seed": cfg["seed_base"] + length + i,
                        "variant": variant_name,
                        "helix_weight": cfg["helix_weight"],
                        "contact_weight": cfg["contact_weight"],
                    })

    total = len(designs)
    print(f"\n{'='*80}")
    print(f"TURBO DESIGN LAUNCHER: {total} designs")
    print(f"{'='*80}")

    # Print variant breakdown
    from collections import Counter
    variant_counts = Counter(d["variant"] for d in designs)
    length_counts = Counter(d["binder_length"] for d in designs)
    for v, c in sorted(variant_counts.items()):
        cfg = VARIANTS[v]
        print(f"  {v:<15} {c:>3} designs  helix={cfg['helix_weight']}  contact={cfg['contact_weight']}")
    print(f"  {'---':>15}")
    for l, c in sorted(length_counts.items()):
        print(f"  Length {l:<8} {c:>3} designs")
    print(f"\n  Estimated cost: ~${total * 2}")
    print(f"  Estimated time: ~30 min (all parallel on Modal)")
    print()

    # Launch ALL designs in parallel via .starmap()
    call_args = []
    for d in designs:
        call_args.append((
            d["binder_length"],
            d["seed"],
            d.get("variant", "standard"),
            d.get("helix_weight", 2.5),
            d.get("contact_weight", 1.0),
            d.get("stage1_steps", 100),
            d.get("stage2_steps", 50),
            d.get("stage3_steps", 15),
            d.get("design_samples", 4),
            d.get("design_recycling", 1),
            d.get("ranking_samples", 6),
            d.get("ranking_recycling", 3),
        ))

    results = []
    for result in design_binder_variant.starmap(call_args):
        results.append(result)
        cys_flag = " CYS!" if result["n_cysteines"] > 0 else ""
        ig_flag = " IG!" if result.get("ig_motifs_found") else ""
        print(
            f"  [{len(results):>2}/{total}] {result['design_id']:<55} "
            f"ipTM={result['iptm']:.3f} "
            f"ipSAE={result.get('ipsae_min', 0):.3f} "
            f"pLDDT={result['binder_plddt']:.3f}"
            f"{cys_flag}{ig_flag}"
        )

    # ── Sort and report ──
    results.sort(key=lambda r: r.get("ipsae_min", 0), reverse=True)

    print(f"\n{'='*110}")
    print(f"ALL {len(results)} TURBO DESIGNS COMPLETE")
    print(f"{'='*110}")
    print(f"{'#':<4}{'Variant':<15}{'Design ID':<40}{'ipTM':<8}{'pLDDT':<8}{'ipSAE':<8}{'Asym':<7}{'Cys':<5}{'Ig':<8}")
    print(f"{'-'*100}")
    for i, r in enumerate(results[:30]):
        ig_flag = ",".join(r.get("ig_motifs_found", [])) or "-"
        print(
            f"{i+1:<4}{r.get('variant', '?'):<15}{r['design_id']:<40}"
            f"{r['iptm']:<8.4f}"
            f"{r['binder_plddt']:<8.4f}"
            f"{r.get('ipsae_min', 0):<8.4f}"
            f"{r.get('ipsae_asymmetry', 0):<7.3f}"
            f"{r['n_cysteines']:<5}"
            f"{ig_flag}"
        )

    # ── Quality summary ──
    n_cys_free = sum(1 for r in results if r["n_cysteines"] == 0)
    n_ig_free = sum(1 for r in results if not r.get("ig_motifs_found"))
    n_clean = sum(1 for r in results if r["n_cysteines"] == 0 and not r.get("ig_motifs_found"))
    n_good_ipsae = sum(1 for r in results if r.get("ipsae_min", 0) > 0.5)
    n_great_ipsae = sum(1 for r in results if r.get("ipsae_min", 0) > 0.7)

    print(f"\n{'='*60}")
    print(f"QUALITY SUMMARY:")
    print(f"  Cysteine-free:      {n_cys_free}/{len(results)}")
    print(f"  Ig-motif-free:      {n_ig_free}/{len(results)}")
    print(f"  Clean (both):       {n_clean}/{len(results)}")
    print(f"  ipSAE > 0.5:        {n_good_ipsae}/{len(results)}")
    print(f"  ipSAE > 0.7:        {n_great_ipsae}/{len(results)}")

    # Variant breakdown
    print(f"\n  Per-variant quality:")
    for v in VARIANTS:
        v_results = [r for r in results if r.get("variant") == v]
        if v_results:
            v_clean = sum(1 for r in v_results if r["n_cysteines"] == 0 and not r.get("ig_motifs_found"))
            v_good = sum(1 for r in v_results if r.get("ipsae_min", 0) > 0.5)
            avg_ipsae = sum(r.get("ipsae_min", 0) for r in v_results) / len(v_results)
            print(f"    {v:<15} {len(v_results)} designs, {v_clean} clean, {v_good} ipSAE>0.5, avg={avg_ipsae:.3f}")
    print(f"{'='*60}")

    # ── Save batch summary ──
    os.makedirs("results/designs", exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    summary_path = f"results/designs/batch_turbo_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved turbo batch: {summary_path}")

    fasta_path = f"results/designs/turbo_sequences_{ts}.fasta"
    with open(fasta_path, "w") as f:
        for i, r in enumerate(results):
            f.write(
                f">{r['design_id']} rank={i+1} variant={r.get('variant', '?')} "
                f"ipTM={r['iptm']:.4f} ipSAE={r.get('ipsae_min', 0):.4f} cys={r['n_cysteines']}\n"
            )
            f.write(f"{r['binder_sequence']}\n")
    print(f"Saved sequences: {fasta_path}")

    print(f"\n  NEXT STEPS:")
    print(f"  1. modal run scripts/validate_monomers.py --batch {summary_path} --top 15")
    print(f"  2. python scripts/select_final.py --batch {summary_path} --validation results/designs/monomer_validation.json --output results/submission")
    print(f"  3. Submit results/submission/submission.fasta to bioArena")
