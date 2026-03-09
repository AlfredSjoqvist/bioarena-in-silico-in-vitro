#!/usr/bin/env python3
"""
BoltzGen TREM2 Binder Design on Modal.
Diffusion-based backbone generation + MPNN inverse folding + Boltz-2 refolding.

This is complementary to gradient-based designs in design_and_rank.py:
  - Gradient approach: optimizes sequences (slow, high quality, ~30 min/design)
  - BoltzGen approach: generates diverse backbones (fast, ~3 sec/sample, structural diversity)

Adapted from mosaic/examples/boltzgen_pipeline.py.

Usage:
    # Full run: 60 samples at length 80 (~$3, ~15 min)
    modal run scripts/boltzgen_modal.py

    # Quick smoke test: 12 samples
    modal run scripts/boltzgen_modal.py --smoke-test

    # Custom: multiple lengths
    modal run scripts/boltzgen_modal.py --binder-length 80 --n-samples 60
    modal run scripts/boltzgen_modal.py --binder-length 70 --n-samples 60
"""

import json
import os
import time

import modal

app = modal.App("trem2-boltzgen-design")

# Container image — needs BoltzGen + Mosaic + all dependencies
mosaic_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget")
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
model_cache = modal.Volume.from_name("trem2-model-cache", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


@app.function(
    gpu="A100",
    image=mosaic_image,
    timeout=3600,
    volumes={"/results": results_volume, "/model-cache": model_cache},
    memory=32768,
)
def run_boltzgen_batch(
    binder_length: int = 80,
    n_samples: int = 60,
    batch_size: int = 12,
    seed: int = 0,
    filter_rmsd: float = 2.5,
    mpnn_temp: float = 0.1,
) -> list[dict]:
    """Generate TREM2 binders via BoltzGen diffusion + MPNN + Boltz-2 refolding."""
    import sys
    sys.path.insert(0, "/opt/mosaic/src")
    sys.path.insert(0, "/opt/mosaic")

    # Set model cache directory
    os.environ["BOLTZ_CACHE"] = "/model-cache"

    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    from pathlib import Path
    from tempfile import NamedTemporaryFile
    from dataclasses import dataclass

    from mosaic.models.boltzgen import (
        load_boltzgen,
        load_features_and_structure_writer,
        Sampler,
        BoltzGenOutput,
        CoordsToToken,
    )
    from mosaic.models.boltz2 import Boltz2, pad_atom_features
    from mosaic.losses.boltz2 import Boltz2Output, Boltz2FromTrunkOutput
    from mosaic.losses.structure_prediction import (
        BinderTargetIPSAE,
        TargetBinderIPSAE,
        IPTMLoss,
    )
    from mosaic.losses.protein_mpnn import jacobi_inverse_fold
    from mosaic.util import calculate_rmsd, fold_in
    from mosaic.structure_prediction import TargetChain
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.common import TOKENS, LossTerm
    import gemmi
    import torch
    import urllib

    t_start = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"BOLTZGEN TREM2 DESIGN")
    print(f"  Binder length: {binder_length}")
    print(f"  Samples: {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Seed: {seed}")
    print(f"{'='*60}")

    # ── Download TREM2 structure ──
    print("Downloading TREM2 structure (5UD7)...")
    with urllib.request.urlopen("https://files.rcsb.org/download/5UD7.cif") as response:
        st = gemmi.make_structure_from_block(
            gemmi.cif.read_string(response.read().decode("utf-8"))[0]
        )
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    target_chain = st[0]["A"]
    print(f"  Target chain A: {len([r for r in target_chain])} residues")

    # ── Load models ──
    print("Loading BoltzGen...")
    boltzgen = load_boltzgen(checkpoint_dir=Path("/model-cache"))
    print("Loading Boltz-2...")
    boltz2 = Boltz2()
    print("Loading ProteinMPNN (soluble)...")
    mpnn = load_mpnn_sol()
    print(f"  Models loaded in {time.time() - t_start:.1f}s")

    # ── Cysteine bias for MPNN ──
    cys_idx = TOKENS.index("C")
    mpnn_bias = jnp.zeros((binder_length, 20)).at[:, cys_idx].set(-1e6)

    # ── Create diffusion features ──
    print("Creating diffusion features...")

    struct_for_pdb = gemmi.Structure()
    model_for_pdb = gemmi.Model("0")
    model_for_pdb.add_chain(target_chain)
    struct_for_pdb.add_model(model_for_pdb)
    struct_for_pdb[0][0].name = "A"

    with NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tf:
        struct_for_pdb.write_pdb(tf.name)
        target_pdb_path = tf.name

    yaml_binder = f"""
    entities:
      - protein:
          id: B
          sequence: {binder_length}

      - file:
          path: {target_pdb_path}

          include:
            - chain:
                id: A
    """
    features, _ = load_features_and_structure_writer(yaml_string=yaml_binder)
    coords2token = CoordsToToken(features)

    # ── Helper classes ──
    class ZeroLoss(LossTerm):
        def __call__(self, sequence, output, key):
            return 0.0, {"zero": 0.0}

    # ── JIT-compiled sampling function ──
    @eqx.filter_jit
    def sample_and_inverse_fold_fn(key):
        sampler = Sampler.from_features(
            model=boltzgen,
            features=features,
            key=fold_in(key, "sampler"),
            deterministic=True,
            recycling_steps=3,
        )

        sample = sampler(
            structure_module=boltzgen.structure_module,
            num_sampling_steps=500,
            step_scale=jnp.array(2.0),
            noise_scale=jnp.array(0.88),
            key=fold_in(key, "diffusion"),
        )
        model_output = BoltzGenOutput(sample, features, coords2token)

        mpnn_seq = jacobi_inverse_fold(
            mpnn,
            binder_length,
            model_output,
            mpnn_temp,
            fold_in(key, "inverse_fold"),
            bias=mpnn_bias,
        )
        return (
            jnp.argmax(model_output.full_sequence, -1)[:binder_length],
            model_output.backbone_coordinates,
            mpnn_seq,
        )

    # ── Batched sampling + inverse folding ──
    batched_sample = jax.vmap(sample_and_inverse_fold_fn)

    def tokens_to_str(tokens):
        return "".join([TOKENS[int(i)] for i in tokens])

    # ── Refold helper ──
    target_sequence = "".join(gemmi.one_letter_code([r.name for r in target_chain]))

    def load_padded_refold_features(sequences, target_chains_list):
        """Load Boltz-2 features for multiple sequences, padded to same atom count."""
        from os import devnull
        from contextlib import redirect_stdout, redirect_stderr

        with redirect_stdout(open(devnull, "w")), redirect_stderr(open(devnull, "w")):
            if target_chains_list:
                target_feat, _ = boltz2.target_only_features(target_chains_list)
                target_atom_size = target_feat["atom_pad_mask"].shape[-1]
            else:
                target_atom_size = 0

            unpadded = [
                boltz2.target_only_features(
                    [TargetChain(tokens_to_str(seq), use_msa=False)] + target_chains_list
                )
                for seq in sequences
            ]

        max_atom_size = max(fw[0]["atom_pad_mask"].shape[-1] for fw in unpadded)
        pad_length = sequences[0].size * 14 + target_atom_size
        pad_length = ((pad_length + 31) // 32) * 32

        assert pad_length >= max_atom_size
        assert pad_length % 32 == 0

        padded_features, writers = [], []
        for f, w in unpadded:
            pf = pad_atom_features(f, pad_length)
            w.atom_pad_mask = torch.Tensor(np.array(pf["atom_pad_mask"])[None])
            padded_features.append(pf)
            writers.append(w)

        return padded_features, writers

    # ── Refolding function ──
    @eqx.filter_jit
    def multifold(key, feat, model, loss, num_samples):
        output = Boltz2Output(
            joltz2=model.model,
            features=feat,
            deterministic=True,
            key=fold_in(key, "trunk"),
            recycling_steps=3,
        )

        def apply_loss(k):
            from_trunk = Boltz2FromTrunkOutput(
                joltz2=model.model,
                features=feat,
                deterministic=True,
                key=k,
                initial_embedding=output.initial_embedding,
                trunk_state=output.trunk_state,
                recycling_steps=3,
            )
            v, aux = loss(
                sequence=jnp.zeros((binder_length, 20)),
                output=from_trunk,
                key=fold_in(k, "loss"),
            )
            return v, from_trunk.backbone_coordinates

        losses, backbones = jax.vmap(apply_loss)(
            jax.random.split(fold_in(key, "samples"), num_samples)
        )
        best_idx = jnp.argmin(losses)
        return losses[best_idx], backbones[best_idx]

    @eqx.filter_jit
    def batched_backbone_rmsd(x, y):
        return jax.vmap(
            lambda i, j: calculate_rmsd(i.reshape(-1, 3), j.reshape(-1, 3))
        )(x, y)

    # ── Main sampling loop ──
    ranking_loss = 1.0 * IPTMLoss() + 0.5 * TargetBinderIPSAE() + 0.5 * BinderTargetIPSAE()
    zero_loss = ZeroLoss()

    all_results = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = time.time()
        current_batch_size = min(batch_size, n_samples - batch_idx * batch_size)

        print(f"\n── Batch {batch_idx+1}/{n_batches} ({current_batch_size} samples) ──")

        # Generate backbones + inverse fold
        keys = jax.random.split(jax.random.key(seed + batch_idx * 1000), current_batch_size)
        diffusion_seqs, diffusion_bb, mpnn_seqs = batched_sample(keys)

        print(f"  Diffusion + MPNN: {time.time() - batch_start:.1f}s")

        # Refold with Boltz-2 (complex)
        t_refold = time.time()
        target_chains_for_refold = [
            TargetChain(target_sequence, use_msa=False, template_chain=target_chain)
        ]
        refold_features, refold_writers = load_padded_refold_features(
            mpnn_seqs, target_chains_for_refold
        )

        refold_losses = []
        refold_bbs = []
        for i in range(current_batch_size):
            loss_val, bb = multifold(
                jax.random.key(seed + batch_idx * 1000 + i + 500),
                refold_features[i],
                boltz2,
                ranking_loss,
                num_samples=3,
            )
            refold_losses.append(float(loss_val))
            refold_bbs.append(bb)

        refold_bbs_stacked = jnp.stack(refold_bbs)
        print(f"  Boltz-2 refold: {time.time() - t_refold:.1f}s")

        # Compute backbone RMSD
        bb_rmsds = batched_backbone_rmsd(
            diffusion_bb[:current_batch_size, :binder_length],
            refold_bbs_stacked[:, :binder_length],
        )

        # Monomer refold for validation
        t_mono = time.time()
        mono_features, _ = load_padded_refold_features(mpnn_seqs, [])
        mono_bbs = []
        for i in range(current_batch_size):
            _, mono_bb = multifold(
                jax.random.key(seed + batch_idx * 1000 + i + 900),
                mono_features[i],
                boltz2,
                zero_loss,
                num_samples=1,
            )
            mono_bbs.append(mono_bb)

        mono_bbs_stacked = jnp.stack(mono_bbs)
        mono_rmsds = batched_backbone_rmsd(
            diffusion_bb[:current_batch_size, :binder_length],
            mono_bbs_stacked,
        )
        print(f"  Monomer refold: {time.time() - t_mono:.1f}s")

        # Process results
        for i in range(current_batch_size):
            seq_str = tokens_to_str(mpnn_seqs[i])
            bb_rmsd = float(bb_rmsds[i])
            mono_rmsd = float(mono_rmsds[i])
            ranking = float(refold_losses[i])

            # Hard cysteine removal (replace any C with closest hydrophobic)
            if "C" in seq_str:
                seq_str = seq_str.replace("C", "A")

            # Check if passes RMSD filter
            passes = bb_rmsd < filter_rmsd and mono_rmsd < filter_rmsd

            sample_id = f"boltzgen_L{binder_length}_b{batch_idx}_s{i}_{timestamp}"

            result = {
                "design_id": sample_id,
                "target": "TREM2",
                "target_sequence": TREM2_SEQUENCE,
                "binder_sequence": seq_str,
                "binder_length": binder_length,
                "seed": seed + batch_idx * 1000 + i,
                "iptm": 0.0,  # Not directly available from this pipeline
                "plddt_mean": 0.0,
                "binder_plddt": 0.0,
                "ranking_loss": ranking,
                "ranking_aux": {},
                "ipsae_bt": 0.0,
                "ipsae_tb": 0.0,
                "ipsae_min": 0.0,
                "ipsae_asymmetry": 0.0,
                "n_cysteines": seq_str.count("C"),
                "ig_motifs_found": [],
                "bb_rmsd": bb_rmsd,
                "mono_rmsd": mono_rmsd,
                "passes_rmsd_filter": passes,
                "method": "boltzgen",
                "total_time_seconds": 0,
                "params": {
                    "binder_length": binder_length,
                    "mpnn_temp": mpnn_temp,
                    "filter_rmsd": filter_rmsd,
                    "batch_size": batch_size,
                },
            }

            # Ig-motif check
            for motif in ["GKRFAW", "GRYRCLAL", "CPFD", "PWTL", "SHPD"]:
                if motif in seq_str:
                    result["ig_motifs_found"].append(motif)

            status = "PASS" if passes else "FAIL"
            print(
                f"    [{batch_idx*batch_size + i + 1}/{n_samples}] {sample_id} "
                f"| rank={ranking:.3f} "
                f"| bb_rmsd={bb_rmsd:.2f} "
                f"| mono_rmsd={mono_rmsd:.2f} "
                f"| {status}"
            )

            all_results.append(result)

        # Save per-batch checkpoint
        out_dir = f"/results/boltzgen_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/batch_{batch_idx}.json", "w") as f:
            json.dump(
                [r for r in all_results[batch_idx*batch_size:]],
                f, indent=2,
            )

        batch_elapsed = time.time() - batch_start
        print(f"  Batch time: {batch_elapsed:.1f}s")

    # ── Commit and return ──
    results_volume.commit()
    model_cache.commit()

    total_time = time.time() - t_start

    # Filter passing results
    passing = [r for r in all_results if r["passes_rmsd_filter"]]
    failing = [r for r in all_results if not r["passes_rmsd_filter"]]

    print(f"\n{'='*60}")
    print(f"BOLTZGEN COMPLETE")
    print(f"  Total samples:  {len(all_results)}")
    print(f"  Passing RMSD:   {len(passing)}/{len(all_results)} (< {filter_rmsd}Å)")
    print(f"  Cysteine-free:  {sum(1 for r in passing if r['n_cysteines'] == 0)}/{len(passing)}")
    print(f"  Total time:     {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    return passing


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    binder_length: int = 80,
    n_samples: int = 60,
    batch_size: int = 12,
    seed: int = 5000,
):
    """Launch BoltzGen TREM2 binder design on Modal."""

    if smoke_test:
        print("SMOKE TEST: 12 samples, length 80")
        n_samples = 12
        batch_size = 12

    print(f"Launching BoltzGen: {n_samples} samples at length {binder_length}")

    results = run_boltzgen_batch.remote(
        binder_length=binder_length,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
    )

    # Sort by ranking loss (lower = better)
    results.sort(key=lambda r: r["ranking_loss"])

    print(f"\n{'='*80}")
    print(f"BOLTZGEN RESULTS: {len(results)} passing designs")
    print(f"{'='*80}")
    print(f"{'#':<4}{'Design ID':<45}{'RankLoss':<10}{'BB RMSD':<10}{'Mono RMSD':<10}{'Cys':<5}")
    print(f"{'-'*80}")
    for i, r in enumerate(results[:20]):
        print(
            f"{i+1:<4}{r['design_id']:<45}"
            f"{r['ranking_loss']:<10.4f}"
            f"{r['bb_rmsd']:<10.2f}"
            f"{r['mono_rmsd']:<10.2f}"
            f"{r['n_cysteines']}"
        )

    # Save results
    os.makedirs("results/designs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = f"results/designs/boltzgen_L{binder_length}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Save FASTA
    fasta_path = f"results/designs/boltzgen_L{binder_length}_{ts}.fasta"
    with open(fasta_path, "w") as f:
        for i, r in enumerate(results):
            f.write(
                f">{r['design_id']} rank={i+1} "
                f"ranking_loss={r['ranking_loss']:.4f} "
                f"bb_rmsd={r['bb_rmsd']:.2f}\n"
            )
            f.write(f"{r['binder_sequence']}\n")
    print(f"Saved FASTA: {fasta_path}")

    n_cys_free = sum(1 for r in results if r["n_cysteines"] == 0)
    print(f"\nCysteine-free: {n_cys_free}/{len(results)}")
    print(f"\nNext: merge with gradient designs and run select_final.py")
