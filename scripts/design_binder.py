#!/usr/bin/env python3
"""
TREM2 Binder Design using Mosaic (Boltz-2 gradient-based optimization).
Adapted from the Adaptyv Nipah competition winning approach by Escalante Bio.

Usage:
    python design_binder.py --binder-length 100 --seed 42
    python design_binder.py --binder-length 100 --seed 42 --output results/designs/
"""

import argparse
import json
import time
from pathlib import Path

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

# TREM2 ectodomain sequence (residues ~19-131 from PDB 5UD7)
TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


def design_binder(
    binder_length: int = 100,
    seed: int = 42,
    output_dir: str = "results/designs",
    # Stage 1 params
    stage1_steps: int = 100,
    stage1_stepsize_factor: float = 0.2,
    stage1_momentum: float = 0.3,
    # Stage 2 params
    stage2_steps: int = 50,
    stage2_stepsize_factor: float = 0.5,
    stage2_scale: float = 1.25,
    # Stage 3 params
    stage3_steps: int = 15,
    stage3_stepsize_factor: float = 0.5,
    stage3_scale: float = 1.4,
    # Design params
    design_samples: int = 4,
    design_recycling: int = 1,
    ranking_samples: int = 6,
    ranking_recycling: int = 3,
    # Loss weights
    contact_weight: float = 1.0,
    within_contact_weight: float = 1.0,
    mpnn_weight: float = 10.0,
    pae_weight: float = 0.05,
    within_pae_weight: float = 0.4,
    iptm_weight: float = 0.025,
    ptm_weight: float = 0.025,
    plddt_weight: float = 0.1,
):
    """Run full 4-stage Boltz-2 binder design pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    design_id = f"trem2_L{binder_length}_s{seed}_{timestamp}"

    print(f"{'='*60}")
    print(f"TREM2 Binder Design: {design_id}")
    print(f"Binder length: {binder_length}, Seed: {seed}")
    print(f"{'='*60}\n")

    # Initialize models
    print("Loading Boltz-2...")
    model = Boltz2()

    print("Loading ProteinMPNN...")
    mpnn = ProteinMPNN.from_pretrained()

    # Create features
    print("Generating features...")
    features, structure_writer = model.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(TREM2_SEQUENCE, use_msa=False)],
    )

    # Cysteine bias (prevent disulfide bonds)
    cys_idx = TOKENS.index("C")
    cys_bias = np.zeros((binder_length, 20))
    cys_bias[:, cys_idx] = -1e6

    # Build design loss
    print("Building design loss...")
    design_loss = model.build_multisample_loss(
        loss=(
            contact_weight * sp.BinderTargetContact()
            + within_contact_weight * sp.WithinBinderContact()
            + mpnn_weight * InverseFoldingSequenceRecovery(
                mpnn,
                temp=jax.numpy.array(0.001),
                bias=jnp.array(cys_bias),
            )
            + pae_weight * sp.TargetBinderPAE()
            + pae_weight * sp.BinderTargetPAE()
            + iptm_weight * sp.IPTMLoss()
            + within_pae_weight * sp.WithinBinderPAE()
            + ptm_weight * sp.pTMEnergy()
            + plddt_weight * sp.PLDDTLoss()
        ),
        features=features,
        recycling_steps=design_recycling,
        num_samples=design_samples,
    )

    # Stage 1: Soft PSSM generation
    print(f"\n--- Stage 1: Soft PSSM ({stage1_steps} steps) ---")
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
        stepsize=stage1_stepsize_factor * np.sqrt(binder_length),
        momentum=stage1_momentum,
    )
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Stage 2: First sharpening
    print(f"\n--- Stage 2: Sharpening ({stage2_steps} steps, scale={stage2_scale}) ---")
    t0 = time.time()
    PSSM_sharp, _ = simplex_APGM(
        loss_function=design_loss,
        n_steps=stage2_steps,
        x=PSSM,
        stepsize=stage2_stepsize_factor * np.sqrt(binder_length),
        scale=stage2_scale,
        logspace=True,
        momentum=0.0,
    )
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Stage 3: Final sharpening
    print(f"\n--- Stage 3: Final sharpening ({stage3_steps} steps, scale={stage3_scale}) ---")
    t0 = time.time()
    PSSM_final, _ = simplex_APGM(
        loss_function=design_loss,
        n_steps=stage3_steps,
        x=PSSM_sharp,
        stepsize=stage3_stepsize_factor * np.sqrt(binder_length),
        scale=stage3_scale,
        logspace=True,
        momentum=0.0,
    )
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Extract hard sequence
    binder_sequence = "".join(TOKENS[i] for i in PSSM_final.argmax(-1))
    print(f"\nDesigned binder sequence ({len(binder_sequence)} aa):")
    print(f"  {binder_sequence}")

    # Stage 4: Refold and rank
    print(f"\n--- Stage 4: Refolding ({ranking_samples} samples, {ranking_recycling} recycling) ---")
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
    prediction = model.predict(
        PSSM=one_hot_seq,
        features=features,
        writer=structure_writer,
        recycling_steps=ranking_recycling,
        key=jax.random.key(seed),
    )
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Extract scores
    iptm = float(prediction.iptm)
    plddt_mean = float(prediction.plddt.mean())
    print(f"\n{'='*60}")
    print(f"RESULTS: {design_id}")
    print(f"  ipTM:       {iptm:.4f}")
    print(f"  Mean pLDDT: {plddt_mean:.4f}")
    print(f"  Sequence:   {binder_sequence}")
    print(f"{'='*60}")

    # Save outputs
    # PDB structure
    pdb_path = output_path / f"{design_id}.pdb"
    with open(pdb_path, "w") as f:
        f.write(prediction.st.make_pdb_string())
    print(f"\nSaved PDB: {pdb_path}")

    # FASTA sequence
    fasta_path = output_path / f"{design_id}.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">{design_id} ipTM={iptm:.4f} pLDDT={plddt_mean:.4f}\n")
        f.write(f"{binder_sequence}\n")
    print(f"Saved FASTA: {fasta_path}")

    # Metadata JSON
    metadata = {
        "design_id": design_id,
        "target": "TREM2",
        "target_sequence": TREM2_SEQUENCE,
        "binder_sequence": binder_sequence,
        "binder_length": binder_length,
        "seed": seed,
        "iptm": iptm,
        "plddt_mean": plddt_mean,
        "params": {
            "stage1_steps": stage1_steps,
            "stage2_steps": stage2_steps,
            "stage3_steps": stage3_steps,
            "design_samples": design_samples,
            "ranking_samples": ranking_samples,
            "contact_weight": contact_weight,
            "mpnn_weight": mpnn_weight,
        },
    }
    meta_path = output_path / f"{design_id}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Design TREM2 binders using Mosaic/Boltz-2")
    parser.add_argument("--binder-length", type=int, default=100, help="Binder length in residues")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/designs", help="Output directory")
    parser.add_argument("--stage1-steps", type=int, default=100)
    parser.add_argument("--stage2-steps", type=int, default=50)
    parser.add_argument("--stage3-steps", type=int, default=15)
    parser.add_argument("--design-samples", type=int, default=4, help="Diffusion samples during design")
    parser.add_argument("--ranking-samples", type=int, default=6, help="Diffusion samples during ranking")
    args = parser.parse_args()

    design_binder(
        binder_length=args.binder_length,
        seed=args.seed,
        output_dir=args.output,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        stage3_steps=args.stage3_steps,
        design_samples=args.design_samples,
        ranking_samples=args.ranking_samples,
    )


if __name__ == "__main__":
    main()
