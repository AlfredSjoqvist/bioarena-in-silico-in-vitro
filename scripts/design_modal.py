#!/usr/bin/env python3
"""
Run parallel TREM2 binder designs on Modal GPUs.
This is the main script for generating many designs quickly during the hackathon.

Usage:
    modal run scripts/design_modal.py --n-designs 20 --binder-length 100
    modal run scripts/design_modal.py --n-designs 10 --binder-length 80 --gpu H100
"""

import json
import os
import time

import modal

app = modal.App("trem2-binder-design")

# Container image with all dependencies
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

# Persistent volume for storing results
results_volume = modal.Volume.from_name("trem2-results", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


@app.function(
    gpu="B200",
    image=mosaic_image,
    timeout=3600,  # 1 hour per design
    volumes={"/results": results_volume},
)
def design_single_binder(binder_length: int, seed: int) -> dict:
    """Design a single TREM2 binder on a GPU."""
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
    print(f"Starting design: {design_id}")

    # Load models
    model = Boltz2()
    mpnn = ProteinMPNN.from_pretrained()

    # Features
    features, structure_writer = model.binder_features(
        binder_length=binder_length,
        chains=[TargetChain(TREM2_SEQUENCE, use_msa=False)],
    )

    # Cysteine bias
    cys_idx = TOKENS.index("C")
    cys_bias = np.zeros((binder_length, 20))
    cys_bias[:, cys_idx] = -1e6

    # Design loss
    design_loss = model.build_multisample_loss(
        loss=(
            1.0 * sp.BinderTargetContact()
            + 1.0 * sp.WithinBinderContact()
            + 10.0 * InverseFoldingSequenceRecovery(
                mpnn, temp=jax.numpy.array(0.001), bias=jnp.array(cys_bias),
            )
            + 0.05 * sp.TargetBinderPAE()
            + 0.05 * sp.BinderTargetPAE()
            + 0.025 * sp.IPTMLoss()
            + 0.4 * sp.WithinBinderPAE()
            + 0.025 * sp.pTMEnergy()
            + 0.1 * sp.PLDDTLoss()
        ),
        features=features,
        recycling_steps=1,
        num_samples=4,
    )

    # Stage 1: Soft PSSM
    print("Stage 1: Soft PSSM (100 steps)")
    _, PSSM = simplex_APGM(
        loss_function=design_loss,
        n_steps=100,
        x=jax.nn.softmax(
            0.5 * jax.random.gumbel(
                key=jax.random.key(seed), shape=(binder_length, 20),
            )
        ),
        stepsize=0.2 * np.sqrt(binder_length),
        momentum=0.3,
    )

    # Stage 2: Sharpen
    print("Stage 2: Sharpening (50 steps)")
    PSSM_sharp, _ = simplex_APGM(
        loss_function=design_loss,
        n_steps=50,
        x=PSSM,
        stepsize=0.5 * np.sqrt(binder_length),
        scale=1.25,
        logspace=True,
        momentum=0.0,
    )

    # Stage 3: Final sharpen
    print("Stage 3: Final sharpening (15 steps)")
    PSSM_final, _ = simplex_APGM(
        loss_function=design_loss,
        n_steps=15,
        x=PSSM_sharp,
        stepsize=0.5 * np.sqrt(binder_length),
        scale=1.4,
        logspace=True,
        momentum=0.0,
    )

    # Extract sequence
    binder_sequence = "".join(TOKENS[i] for i in PSSM_final.argmax(-1))
    print(f"Sequence: {binder_sequence}")

    # Stage 4: Refold and rank
    print("Stage 4: Refolding (6 samples, 3 recycling)")
    prediction = model.predict(
        PSSM=jax.nn.one_hot(
            jnp.array([TOKENS.index(c) for c in binder_sequence]), 20
        ),
        features=features,
        writer=structure_writer,
        recycling_steps=3,
        key=jax.random.key(seed),
    )

    iptm = float(prediction.iptm)
    plddt_mean = float(prediction.plddt.mean())

    # Save to persistent volume
    pdb_str = prediction.st.make_pdb_string()
    with open(f"/results/{design_id}.pdb", "w") as f:
        f.write(pdb_str)

    result = {
        "design_id": design_id,
        "binder_sequence": binder_sequence,
        "binder_length": binder_length,
        "seed": seed,
        "iptm": iptm,
        "plddt_mean": plddt_mean,
    }
    with open(f"/results/{design_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Done: {design_id} | ipTM={iptm:.4f} | pLDDT={plddt_mean:.4f}")
    return result


@app.local_entrypoint()
def main(
    n_designs: int = 20,
    binder_length: int = 100,
    seed_start: int = 0,
):
    """Launch parallel binder designs on Modal."""
    print(f"Launching {n_designs} TREM2 binder designs (L={binder_length})")
    print(f"Seeds: {seed_start} to {seed_start + n_designs - 1}")

    seeds = list(range(seed_start, seed_start + n_designs))

    # Launch all designs in parallel via Modal's map
    results = []
    for result in design_single_binder.map(
        [binder_length] * n_designs,
        seeds,
    ):
        results.append(result)
        print(f"  Completed: {result['design_id']} | ipTM={result['iptm']:.4f}")

    # Sort by ipTM
    results.sort(key=lambda r: r["iptm"], reverse=True)

    print(f"\n{'='*60}")
    print(f"ALL DESIGNS COMPLETE ({len(results)} total)")
    print(f"{'='*60}")
    for i, r in enumerate(results):
        print(f"  #{i+1}: {r['design_id']} | ipTM={r['iptm']:.4f} | pLDDT={r['plddt_mean']:.4f}")

    # Save summary
    os.makedirs("results/designs", exist_ok=True)
    summary_path = f"results/designs/batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary: {summary_path}")
