#!/usr/bin/env python3
"""
BoltzGen TREM2 Binder Pipeline on Modal — Structural Diversity.

BoltzGen generates diverse backbone structures via diffusion sampling, then
ProteinMPNN assigns sequences via Jacobi inverse folding. Boltz-2 refolds
and scores with ipSAE. This produces fundamentally different topologies than
the gradient-based PSSM optimization in design_trem2_v3.py.

Adapted from mosaic/examples/boltzgen_pipeline.py (marimo notebook).

Usage:
    # Default: 24 samples at length 80
    modal run scripts/boltzgen_trem2_modal.py

    # Custom
    modal run scripts/boltzgen_trem2_modal.py --n-samples 36 --binder-length 70

    # Smoke test
    modal run scripts/boltzgen_trem2_modal.py --smoke-test
"""

import json
import os
import time

import modal

app = modal.App("trem2-boltzgen-v3")

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

results_volume = modal.Volume.from_name("trem2-results-v3", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)

IG_MOTIFS = ["GKRFAW", "GRYRCLAL", "CPFD", "PWTL", "SHPD", "YRCL"]


@app.function(
    gpu="A100",
    image=mosaic_image,
    timeout=7200,  # 2 hours (BoltzGen batches take longer)
    volumes={"/results": results_volume},
    retries=1,
)
def run_boltzgen_batch(
    binder_length: int = 80,
    n_samples: int = 24,
    batch_size: int = 12,
    filter_rmsd: float = 2.5,
    seed: int = 0,
) -> list[dict]:
    """Run BoltzGen backbone diffusion + MPNN inverse folding + Boltz-2 refolding."""
    import sys
    sys.path.insert(0, "/opt/mosaic/src")
    sys.path.insert(0, "/opt/mosaic")

    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    import gemmi
    import urllib.request
    from tempfile import NamedTemporaryFile
    from contextlib import redirect_stdout, redirect_stderr
    from os import devnull

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
    import torch

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"BOLTZGEN TREM2 PIPELINE")
    print(f"Binder length={binder_length}, Samples={n_samples}, Seed={seed}")
    print(f"{'='*60}")

    t_start = time.time()

    # ── Load models ──
    print("Loading BoltzGen...")
    boltzgen = load_boltzgen()
    print("Loading Boltz-2...")
    boltz2 = Boltz2()
    print("Loading ProteinMPNN (soluble)...")
    mpnn = load_mpnn_sol()
    print(f"Models loaded in {time.time() - t_start:.1f}s")

    # ── Cysteine bias ──
    cys_idx = TOKENS.index("C")
    mpnn_bias = jnp.zeros((binder_length, 20)).at[:, cys_idx].set(-1e6)
    mpnn_temp = 0.1

    # ── Download TREM2 structure ──
    print("Downloading TREM2 structure (5UD7)...")
    with urllib.request.urlopen("https://files.rcsb.org/download/5UD7.cif") as response:
        st = gemmi.make_structure_from_block(
            gemmi.cif.read_string(response.read().decode("utf-8"))[0]
        )
    st.remove_ligands_and_waters()
    st.remove_empty_chains()
    target_chain = st[0]["A"]  # Chain A is TREM2

    # ── Helper: Load diffusion features ──
    def load_diffusion_features(binder_len, tgt_chain):
        struct = gemmi.Structure()
        model = gemmi.Model("0")
        model.add_chain(tgt_chain)
        struct.add_model(model)
        struct[0][0].name = "A"

        with NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tf:
            struct.write_pdb(tf.name)
            yaml_binder = f"""
            entities:
              - protein:
                  id: B
                  sequence: {binder_len}

              - file:
                  path: {tf.name}

                  include:
                    - chain:
                        id: A
            """
            features, _ = load_features_and_structure_writer(yaml_string=yaml_binder)
        return features

    # ── Helper: Load padded refold features ──
    def load_padded_refold_features(sequences, folding_model, target_chains=[]):
        with redirect_stdout(open(devnull, "w")), redirect_stderr(open(devnull, "w")):
            if target_chains:
                target_feat, _ = folding_model.target_only_features(target_chains)
                target_atom_size = target_feat["atom_pad_mask"].shape[-1]
            else:
                target_atom_size = 0

            unpadded_features_writers = [
                folding_model.target_only_features(
                    [
                        TargetChain(tokens_to_str(seq), use_msa=False),
                        *target_chains,
                    ]
                )
                for seq in sequences
            ]
        max_atom_size = max(fw[0]["atom_pad_mask"].shape[-1] for fw in unpadded_features_writers)
        pad_length = sequences[0].size * 14 + target_atom_size
        pad_length = ((pad_length + 31) // 32) * 32
        assert pad_length >= max_atom_size
        assert pad_length % 32 == 0

        padded_features, writers = [], []
        for f, w in unpadded_features_writers:
            padded_f = pad_atom_features(f, pad_length)
            w.atom_pad_mask = torch.Tensor(np.array(padded_f["atom_pad_mask"])[None])
            padded_features.append(padded_f)
            writers.append(w)
        return padded_features, writers

    def tokens_to_str(tokens):
        return "".join([TOKENS[i] for i in tokens])

    # ── Helper: Batched backbone RMSD ──
    @eqx.filter_jit
    def batched_backbone_rmsd(x, y):
        return jax.vmap(
            lambda i, j: calculate_rmsd(i.reshape(-1, 3), j.reshape(-1, 3))
        )(x, y)

    # ── Helper: Multifold (pick best of multiple diffusion samples) ──
    class ZeroLoss(LossTerm):
        def __call__(self, sequence, output, key):
            return 0.0, {"zero": 0.0}

    class FoldOutput(eqx.Module):
        loss: float
        structure_coordinates: jax.Array
        backbone_coordinates: jax.Array

    @eqx.filter_jit
    def multifold(key, features, model, loss, num_samples, binder_len):
        output = Boltz2Output(
            joltz2=model.model,
            features=features,
            deterministic=True,
            key=fold_in(key, "trunk"),
            recycling_steps=3,
        )

        def apply_loss_to_single_sample(key):
            from_trunk_output = Boltz2FromTrunkOutput(
                joltz2=model.model,
                features=features,
                deterministic=True,
                key=key,
                initial_embedding=output.initial_embedding,
                trunk_state=output.trunk_state,
                recycling_steps=3,
            )
            v, aux = loss(
                sequence=jnp.zeros((binder_len, 20)),
                output=from_trunk_output,
                key=fold_in(key, "loss"),
            )
            return FoldOutput(v, from_trunk_output.structure_coordinates, from_trunk_output.backbone_coordinates)

        output = jax.vmap(apply_loss_to_single_sample)(
            jax.random.split(fold_in(key, "samples"), num_samples)
        )
        indmin = jnp.argmin(output.loss)
        return jax.tree.map(lambda v: v[indmin], output)

    # ── Helper: Sample and inverse fold ──
    @eqx.filter_jit
    def sample_and_inverse_fold(
        key, binder_len, features, coords2token, sampler, structure_module,
        num_sampling_steps=500, mpnn_model=mpnn, bias=mpnn_bias, temp=mpnn_temp,
    ):
        sample = sampler(
            structure_module=structure_module,
            num_sampling_steps=num_sampling_steps,
            step_scale=jnp.array(2.0),
            noise_scale=jnp.array(0.88),
            key=fold_in(key, "sampler"),
        )
        model_output = BoltzGenOutput(sample, features, coords2token)
        mpnn_seq = jacobi_inverse_fold(
            mpnn_model, binder_len, model_output, temp,
            fold_in(key, "inverse fold"), bias=bias,
        )
        return (
            jnp.argmax(model_output.full_sequence, -1)[:binder_len],
            model_output.backbone_coordinates,
            mpnn_seq,
        )

    # ── Main BoltzGen pipeline ──
    def run_pipeline_batch(num_samples, binder_len, tgt_chain, key):
        diffusion_features = load_diffusion_features(binder_len, tgt_chain)
        coords2token = CoordsToToken(diffusion_features)
        sampler = Sampler.from_features(
            model=boltzgen,
            features=diffusion_features,
            key=fold_in(key, "sampler"),
            deterministic=True,
            recycling_steps=3,
        )

        diffusion_seqs, diffusion_bb, mpnn_seqs = jax.vmap(
            lambda k: sample_and_inverse_fold(
                k, binder_len, diffusion_features, coords2token,
                sampler, boltzgen.structure_module,
            )
        )(jax.random.split(fold_in(key, "diffusion"), num_samples))

        target_sequence = "".join(
            gemmi.one_letter_code([_r.name for _r in tgt_chain])
        )
        refold_complex_features, refold_writers = load_padded_refold_features(
            mpnn_seqs, boltz2,
            [TargetChain(target_sequence, use_msa=False, template_chain=tgt_chain)],
        )

        ranking_loss = 1.0 * IPTMLoss() + 0.5 * TargetBinderIPSAE() + 0.5 * BinderTargetIPSAE()

        refold_outputs = jax.vmap(
            lambda k, feat: multifold(k, feat, model=boltz2, loss=ranking_loss, num_samples=5, binder_len=binder_len)
        )(
            jax.random.split(fold_in(key, "refold"), num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_complex_features),
        )

        # Monomer refolding for RMSD check
        refold_alone_features, _ = load_padded_refold_features(mpnn_seqs, boltz2, [])
        zero_loss = ZeroLoss()
        refold_alone_outputs = jax.vmap(
            lambda k, feat: multifold(k, feat, model=boltz2, loss=zero_loss, num_samples=1, binder_len=binder_len)
        )(
            jax.random.split(fold_in(key, "monomer"), num_samples),
            jax.tree.map(lambda *feat: jnp.stack(feat), *refold_alone_features),
        )

        backbone_rmsd = batched_backbone_rmsd(diffusion_bb, refold_outputs.backbone_coordinates)
        backbone_rmsd_binder = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len],
            refold_outputs.backbone_coordinates[:, :binder_len],
        )
        backbone_rmsd_binder_alone = batched_backbone_rmsd(
            diffusion_bb[:, :binder_len],
            refold_alone_outputs.backbone_coordinates,
        )

        results = []
        for i in range(num_samples):
            seq = tokens_to_str(mpnn_seqs[i])
            n_cys = seq.count("C")
            ig_found = [m for m in IG_MOTIFS if m in seq]
            rmsd = float(backbone_rmsd_binder[i])
            rmsd_alone = float(backbone_rmsd_binder_alone[i])
            rank_loss = float(refold_outputs.loss[i])

            passes = (
                rmsd < filter_rmsd
                and rmsd_alone < filter_rmsd
                and n_cys == 0
                and not ig_found
            )

            refold_struct = refold_writers[i](refold_outputs.structure_coordinates[i])

            results.append({
                "seq": seq,
                "diffusion_seq": tokens_to_str(diffusion_seqs[i]),
                "ranking_loss": rank_loss,
                "bb_rmsd_binder": rmsd,
                "bb_rmsd_binder_alone": rmsd_alone,
                "n_cysteines": n_cys,
                "ig_motifs_found": ig_found,
                "passes_filters": passes,
                "struct": refold_struct,
            })

        return results

    # ── Run batches ──
    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        actual_batch = min(batch_size, n_samples - batch_idx * batch_size)
        print(f"\nBatch {batch_idx + 1}/{n_batches}: generating {actual_batch} samples...")
        t0 = time.time()
        batch_key = jax.random.key(seed + batch_idx * 1000)
        batch_results = run_pipeline_batch(actual_batch, binder_length, target_chain, batch_key)
        all_samples.extend(batch_results)
        print(f"  Done in {time.time() - t0:.1f}s")

    # ── Sort by ranking loss (lower = better) ──
    all_samples.sort(key=lambda s: s["ranking_loss"] + (1e6 if not s["passes_filters"] else 0))

    # ── Save results ──
    out_dir = f"/results/boltzgen_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    output_results = []
    for i, sample in enumerate(all_samples):
        design_id = f"boltzgen_L{binder_length}_b{i}_{timestamp}"
        status = "PASS" if sample["passes_filters"] else "FAIL"

        # Save PDB
        sample["struct"].write_pdb(f"{out_dir}/{design_id}.pdb")

        # Save FASTA
        with open(f"{out_dir}/{design_id}.fasta", "w") as f:
            f.write(f">{design_id} rank_loss={sample['ranking_loss']:.4f} rmsd={sample['bb_rmsd_binder']:.2f}\n")
            f.write(f"{sample['seq']}\n")

        result = {
            "design_id": design_id,
            "pipeline_version": "boltzgen_v3",
            "target": "TREM2",
            "target_sequence": TREM2_SEQUENCE,
            "binder_sequence": sample["seq"],
            "binder_length": binder_length,
            "seed": seed,
            "ranking_loss": sample["ranking_loss"],
            "bb_rmsd_binder": sample["bb_rmsd_binder"],
            "bb_rmsd_binder_alone": sample["bb_rmsd_binder_alone"],
            "n_cysteines": sample["n_cysteines"],
            "ig_motifs_found": sample["ig_motifs_found"],
            "passes_filters": sample["passes_filters"],
            # Placeholder fields for compatibility with select_final.py
            "iptm": 0.0,
            "plddt_mean": 0.0,
            "binder_plddt": 0.0,
            "ipsae_min": 0.0,
            "ipsae_asymmetry": 0.0,
        }
        output_results.append(result)

        print(f"  [{i+1:>2}/{len(all_samples)}] {design_id} "
              f"| loss={sample['ranking_loss']:.3f} "
              f"| rmsd={sample['bb_rmsd_binder']:.2f} "
              f"| cys={sample['n_cysteines']} "
              f"| {status}")

    # Save batch JSON
    with open(f"{out_dir}/batch_boltzgen_{timestamp}.json", "w") as f:
        json.dump(output_results, f, indent=2)

    total_time = time.time() - t_start
    n_passing = sum(1 for s in all_samples if s["passes_filters"])

    print(f"\n{'='*60}")
    print(f"BOLTZGEN COMPLETE")
    print(f"  Total samples:  {len(all_samples)}")
    print(f"  Passing filters: {n_passing}/{len(all_samples)}")
    print(f"  Total time:     {total_time:.1f}s")
    print(f"  Output dir:     {out_dir}")
    print(f"{'='*60}")

    results_volume.commit()
    return output_results


@app.local_entrypoint()
def main(
    smoke_test: bool = False,
    binder_length: int = 80,
    n_samples: int = 24,
    batch_size: int = 12,
    seed: int = 9999,
):
    """Run BoltzGen TREM2 binder pipeline on Modal."""
    if smoke_test:
        print("SMOKE TEST: 4 samples")
        results = run_boltzgen_batch.remote(
            binder_length=70,
            n_samples=4,
            batch_size=4,
            seed=42,
        )
        n_pass = sum(1 for r in results if r["passes_filters"])
        print(f"\nSmoke test: {n_pass}/{len(results)} passed filters")
        for r in results:
            print(f"  {r['design_id']}: loss={r['ranking_loss']:.3f} cys={r['n_cysteines']}")
        return

    print(f"Launching BoltzGen: {n_samples} samples at L={binder_length}")
    results = run_boltzgen_batch.remote(
        binder_length=binder_length,
        n_samples=n_samples,
        batch_size=batch_size,
        seed=seed,
    )

    # Save locally
    os.makedirs("results/designs", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    local_path = f"results/designs/batch_boltzgen_{ts}.json"
    with open(local_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved local copy: {local_path}")

    n_pass = sum(1 for r in results if r["passes_filters"])
    print(f"\nFinal: {n_pass}/{len(results)} passed all filters")
