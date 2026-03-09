#!/usr/bin/env python3
"""
Optimal TREM2 Binder Design Pipeline on Modal.
Based on the Mosaic/Boltz-2 approach that won the Adaptyv Nipah competition (9/10 binders bound).

Three-phase pipeline:
  1. Gradient-based design (4-stage PSSM optimization)
  2. Ranking refold (4 samples, 2 recycling, ipSAE scoring)
  3. Results export (sequence, PDB, PAE, metrics)

Usage:
    # Full run: 20 designs across 3 length buckets (~$40-50)
    modal run scripts/design_and_rank.py

    # Quick smoke test: 1 design with reduced steps
    modal run scripts/design_and_rank.py --smoke-test

    # Custom run
    modal run scripts/design_and_rank.py --lengths 80 100 --n-per-length 5
"""

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("trem2-binder-design-v2")

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

# Persistent volume for results
results_volume = modal.Volume.from_name("trem2-results-v2", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)

# Loss weight variants for structural diversity
VARIANT_CONFIGS = {
    "default": {
        "helix_loss_weight": 0.3,
        "binder_target_contact_weight": 1.0,
        "use_rog": False,
    },
    "contact_heavy": {
        "helix_loss_weight": 0.3,
        "binder_target_contact_weight": 2.0,
        "use_rog": False,
    },
    "helix_biased": {
        "helix_loss_weight": 2.0,
        "binder_target_contact_weight": 1.0,
        "use_rog": False,
    },
    "compact": {
        "helix_loss_weight": 0.3,
        "binder_target_contact_weight": 1.0,
        "use_rog": True,
    },
    "no_helix": {
        "helix_loss_weight": 0.0,
        "binder_target_contact_weight": 1.0,
        "use_rog": False,
    },
}


@app.function(
    gpu="A100",
    image=mosaic_image,
    timeout=3600,
    volumes={"/results": results_volume},
    retries=1,
)
def design_single_binder(
    binder_length: int,
    seed: int,
    # Stage step counts (reduce for smoke test)
    stage1_steps: int = 100,
    stage2_steps: int = 50,
    stage3_steps: int = 15,
    # Design compute (4 samples for quality)
    design_samples: int = 4,
    design_recycling: int = 1,
    # Ranking compute (6 samples, 3 recycling for accurate scoring)
    ranking_samples: int = 6,
    ranking_recycling: int = 3,
    # TREM2-specific: HelixLoss weight (prevents Ig-fold convergence)
    # 0.3 = gentle nudge (research consensus); 2.0 = aggressive anti-Ig
    helix_loss_weight: float = 0.3,
    # Variant system for structural diversity
    binder_target_contact_weight: float = 1.0,
    use_rog: bool = False,
    variant_name: str = "default",
) -> dict:
    """Design a single TREM2 binder using the full Mosaic/Boltz-2 pipeline."""
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
    design_id = f"trem2_L{binder_length}_s{seed}_{timestamp}"
    print(f"\n{'='*60}")
    print(f"DESIGN: {design_id}")
    print(f"Length={binder_length}, Seed={seed}")
    print(f"{'='*60}")

    # ── Load models ──
    t_start = time.time()
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

    # ── Cysteine bias (CRITICAL: prevent disulfides) ──
    cys_idx = TOKENS.index("C")
    cys_bias = np.zeros((binder_length, 20))
    cys_bias[:, cys_idx] = -1e6

    # ── Build design loss (TREM2-adapted from Nipah winner + cross-bot findings) ──
    print(f"Building design loss (variant={variant_name})...")
    base_loss = (
        binder_target_contact_weight * sp.BinderTargetContact()
        + 1.0 * sp.WithinBinderContact()
        + 10.0 * InverseFoldingSequenceRecovery(
            mpnn,
            temp=jax.numpy.array(0.001),
            bias=jnp.array(cys_bias),
        )
        + helix_loss_weight * sp.HelixLoss()
        + 0.05 * sp.TargetBinderPAE()
        + 0.05 * sp.BinderTargetPAE()
        + 0.025 * sp.IPTMLoss()
        + 0.4 * sp.WithinBinderPAE()
        + 0.025 * sp.pTMEnergy()
        + 0.1 * sp.PLDDTLoss()
    )
    if use_rog:
        base_loss = base_loss + 1.0 * sp.DistogramRadiusOfGyration()

    design_loss = model.build_multisample_loss(
        loss=base_loss,
        features=features,
        recycling_steps=design_recycling,
        num_samples=design_samples,
    )

    # ══════════════════════════════════════════════════════════════
    # STAGE 1: Soft PSSM generation
    # ══════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════
    # STAGE 2: Sharpening
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Stage 2: Sharpening ({stage2_steps} steps, scale=1.25) ──")
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

    # ══════════════════════════════════════════════════════════════
    # STAGE 3: Final sharpening
    # ══════════════════════════════════════════════════════════════
    print(f"\n── Stage 3: Final sharpening ({stage3_steps} steps, scale=1.4) ──")
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

    # ══════════════════════════════════════════════════════════════
    # STAGE 4: Extract hard sequence + Ranking refold
    # ══════════════════════════════════════════════════════════════
    # CRITICAL: Hard clamp cysteines out of PSSM before extracting sequence
    # MPNN bias alone is insufficient — proven by existing 20 binders all having cysteines
    PSSM_final = PSSM_final.at[:, cys_idx].set(-float("inf"))
    binder_sequence = "".join(TOKENS[i] for i in PSSM_final.argmax(-1))
    n_cys = binder_sequence.count("C")
    print(f"\nDesigned sequence ({len(binder_sequence)} AA, {n_cys} cysteines):")
    print(f"  {binder_sequence}")
    if n_cys > 0:
        print(f"  FATAL: {n_cys} cysteines detected despite hard clamp! This should never happen.")

    print(f"\n── Stage 4: Ranking refold ({ranking_samples} samples, {ranking_recycling} recycling) ──")
    t0 = time.time()

    # Build ranking loss with ipSAE (not just ipTM)
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

    # Evaluate ranking loss
    ranking_score, ranking_aux = ranking_loss(one_hot_seq, key=jax.random.key(seed + 1000))
    ranking_score = float(ranking_score)

    # Get full prediction for structure export
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

    # Extract ipSAE-related values from ranking aux
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

    total_time = time.time() - t_start

    # ── Ig-fold motif detection ──
    ig_motifs_found = []
    for motif in ["GKRFAW", "GRYRCLAL", "CPFD", "PWTL", "SHPD"]:
        if motif in binder_sequence:
            ig_motifs_found.append(motif)

    # ── Extract ipSAE values for asymmetry tracking ──
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

    print(f"\n{'='*60}")
    print(f"RESULTS: {design_id}")
    print(f"  Sequence:      {binder_sequence}")
    print(f"  ipTM:          {iptm:.4f}")
    print(f"  Mean pLDDT:    {plddt_mean:.4f}")
    print(f"  Binder pLDDT:  {binder_plddt:.4f}")
    print(f"  Ranking loss:  {ranking_score:.4f}")
    print(f"  ipSAE B→T:     {ipsae_bt:.4f}")
    print(f"  ipSAE T→B:     {ipsae_tb:.4f}")
    print(f"  ipSAE min:     {ipsae_min:.4f}")
    print(f"  ipSAE asym:    {ipsae_asymmetry:.4f}" + (" WARNING" if ipsae_asymmetry > 0.25 else ""))
    print(f"  Cysteines:     {n_cys}")
    print(f"  Ig motifs:     {ig_motifs_found if ig_motifs_found else 'None'}")
    print(f"  Variant:       {variant_name}")
    print(f"  HelixLoss wt:  {helix_loss_weight}")
    print(f"  Contact wt:    {binder_target_contact_weight}")
    print(f"  Use RoG:       {use_rog}")
    print(f"  Total time:    {total_time:.1f}s")
    print(f"{'='*60}")

    # ── Save outputs ──
    out_dir = f"/results/{design_id}"
    os.makedirs(out_dir, exist_ok=True)

    # PDB structure
    pdb_str = prediction.st.make_pdb_string()
    with open(f"{out_dir}/{design_id}.pdb", "w") as f:
        f.write(pdb_str)

    # CIF structure (for IPSAE tool)
    prediction.st.write_minimal_cif(f"{out_dir}/{design_id}.cif")

    # PAE matrix as NPZ (for IPSAE tool)
    np.savez(f"{out_dir}/{design_id}_pae.npz", pae=pae_matrix)

    # FASTA
    with open(f"{out_dir}/{design_id}.fasta", "w") as f:
        f.write(f">{design_id} ipTM={iptm:.4f} pLDDT={plddt_mean:.4f} ipSAE_min={ipsae_min:.4f}\n")
        f.write(f"{binder_sequence}\n")

    # Full metadata JSON
    result = {
        "design_id": design_id,
        "variant": variant_name,
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
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "stage3_steps": stage3_steps,
            "design_samples": design_samples,
            "design_recycling": design_recycling,
            "ranking_samples": ranking_samples,
            "ranking_recycling": ranking_recycling,
            "helix_loss_weight": helix_loss_weight,
            "binder_target_contact_weight": binder_target_contact_weight,
            "use_rog": use_rog,
            "variant": variant_name,
        },
    }
    with open(f"{out_dir}/{design_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Commit volume
    results_volume.commit()

    return result


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    lengths: list[int] = [70, 80, 85],
    n_per_length: int = 7,
    seed_offset: int = 0,
    helix_loss_weight: float = 0.3,
    wave2: bool = False,
    variant: str = "default",
):
    """Launch parallel binder designs on Modal.

    Default (Wave 1): 21 designs at lengths 70, 80, 85 with HelixLoss=0.3
    --wave2: 14 designs at lengths 65, 75 to fill gaps
    --variant: Loss weight variant (default, contact_heavy, helix_biased, compact, no_helix)
    """
    # Resolve variant config
    if variant in VARIANT_CONFIGS:
        vcfg = VARIANT_CONFIGS[variant]
        helix_loss_weight = vcfg["helix_loss_weight"]
        binder_target_contact_weight = vcfg["binder_target_contact_weight"]
        use_rog = vcfg["use_rog"]
        print(f"Using variant '{variant}': HelixLoss={helix_loss_weight}, Contact={binder_target_contact_weight}, RoG={use_rog}")
    else:
        print(f"WARNING: Unknown variant '{variant}', using default weights")
        binder_target_contact_weight = 1.0
        use_rog = False

    if smoke_test:
        print("SMOKE TEST: 1 design with reduced steps")
        result = design_single_binder.remote(
            binder_length=80,
            seed=42,
            stage1_steps=20,
            stage2_steps=10,
            stage3_steps=5,
            design_samples=2,
            ranking_samples=4,
            ranking_recycling=2,
            helix_loss_weight=helix_loss_weight,
            binder_target_contact_weight=binder_target_contact_weight,
            use_rog=use_rog,
            variant_name=variant,
        )
        print(f"\nSmoke test result:")
        print(f"  Variant:  {variant}")
        print(f"  Sequence: {result['binder_sequence']}")
        print(f"  ipTM: {result['iptm']:.4f}")
        print(f"  pLDDT: {result['plddt_mean']:.4f}")
        print(f"  ipSAE min: {result.get('ipsae_min', 0):.4f}")
        print(f"  Cysteines: {result['n_cysteines']}")
        ig_motifs = ["GKRFAW", "GRYRCLAL", "CPFD", "PWTL", "SHPD"]
        found = [m for m in ig_motifs if m in result["binder_sequence"]]
        if found:
            print(f"  WARNING: Ig-fold motifs detected: {found}")
        else:
            print(f"  OK: No Ig-fold motifs detected")
        return

    # Wave 2: fill gap lengths
    if wave2:
        lengths = [65, 75]
        print(f"WAVE 2: Filling gap lengths {lengths}")

    # Seed offset mapping per length bucket
    SEED_OFFSETS = {65: 300, 70: 0, 75: 400, 80: 100, 85: 200, 90: 500, 95: 600}

    # Build design matrix
    designs = []
    for length in lengths:
        base_offset = SEED_OFFSETS.get(length, length * 10)
        for i in range(n_per_length):
            designs.append({
                "binder_length": length,
                "seed": seed_offset + base_offset + i,
                "helix_loss_weight": helix_loss_weight,
                "binder_target_contact_weight": binder_target_contact_weight,
                "use_rog": use_rog,
                "variant_name": variant,
            })

    total = len(designs)
    print(f"Launching {total} TREM2 binder designs")
    print(f"  Variant: {variant}")
    print(f"  Lengths: {lengths}")
    print(f"  Per length: {n_per_length}")
    print(f"  HelixLoss weight: {helix_loss_weight}")
    print(f"  Contact weight: {binder_target_contact_weight}")
    print(f"  Use RoG: {use_rog}")
    print(f"  Estimated cost: ~${total * 2}")
    print(f"  Estimated time: ~30 min (parallel on Modal)")

    # Launch all designs in parallel
    results = []
    for result in design_single_binder.map(
        [d["binder_length"] for d in designs],
        [d["seed"] for d in designs],
        kwargs={
            "helix_loss_weight": helix_loss_weight,
            "binder_target_contact_weight": binder_target_contact_weight,
            "use_rog": use_rog,
            "variant_name": variant,
        },
    ):
        results.append(result)
        cys_flag = " CYSTEINE!" if result["n_cysteines"] > 0 else ""
        print(
            f"  [{len(results)}/{total}] {result['design_id']} "
            f"| ipTM={result['iptm']:.4f} "
            f"| pLDDT={result['binder_plddt']:.4f} "
            f"| rank={result['ranking_loss']:.4f}"
            f"{cys_flag}"
        )

    # Sort by ranking loss (lower is better since it's a loss)
    results.sort(key=lambda r: r["ranking_loss"])

    print(f"\n{'='*100}")
    print(f"ALL {len(results)} DESIGNS COMPLETE (variant={variant}, HelixLoss={helix_loss_weight})")
    print(f"{'='*100}")
    print(f"{'Rank':<6}{'Design ID':<40}{'ipTM':<8}{'pLDDT':<8}{'ipSAE':<8}{'Asym':<7}{'Cys':<5}{'IgMotifs':<10}")
    print(f"{'-'*90}")
    for i, r in enumerate(results[:20]):
        ig_flag = ",".join(r.get("ig_motifs_found", [])) or "-"
        print(
            f"  {i+1:<4} {r['design_id']:<38} "
            f"{r['iptm']:<8.4f}"
            f"{r['binder_plddt']:<8.4f}"
            f"{r.get('ipsae_min', 0):<8.4f}"
            f"{r.get('ipsae_asymmetry', 0):<7.3f}"
            f"{r['n_cysteines']:<5}"
            f"{ig_flag}"
        )

    # Save batch summary locally
    os.makedirs("results/designs", exist_ok=True)
    summary_path = f"results/designs/batch_v2_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")

    # Also save just the sequences as FASTA
    fasta_path = f"results/designs/all_sequences_{time.strftime('%Y%m%d_%H%M%S')}.fasta"
    with open(fasta_path, "w") as f:
        for i, r in enumerate(results):
            f.write(f">{r['design_id']} rank={i+1} ipTM={r['iptm']:.4f} ipSAE={r.get('ipsae_min',0):.4f} cys={r['n_cysteines']}\n")
            f.write(f"{r['binder_sequence']}\n")
    print(f"Saved sequences: {fasta_path}")

    # Quality summary
    n_cys_free = sum(1 for r in results if r["n_cysteines"] == 0)
    n_ig_free = sum(1 for r in results if not r.get("ig_motifs_found"))
    n_good_asym = sum(1 for r in results if r.get("ipsae_asymmetry", 1) < 0.25)
    n_good_ipsae = sum(1 for r in results if r.get("ipsae_min", 0) > 0.5)
    print(f"\n{'='*60}")
    print(f"QUALITY SUMMARY:")
    print(f"  Cysteine-free:      {n_cys_free}/{len(results)}")
    print(f"  Ig-motif-free:      {n_ig_free}/{len(results)}")
    print(f"  Good asymmetry:     {n_good_asym}/{len(results)} (< 0.25)")
    print(f"  Good ipSAE:         {n_good_ipsae}/{len(results)} (> 0.5)")
    print(f"{'='*60}")
