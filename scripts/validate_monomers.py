#!/usr/bin/env python3
"""
Phase 2: Monomer Validation on Modal.
Refolds each binder sequence ALONE (without TREM2) to verify it folds independently.

Filters:
  - Monomer pLDDT > 80: binder must be confident on its own
  - Backbone RMSD < 2.0 Å: binder should fold similarly alone and in complex

Usage:
    # Validate top 10 designs from Phase 1
    modal run scripts/validate_monomers.py --batch results/designs/batch_v2_XXXXXXXX.json --top 10

    # Validate specific design IDs
    modal run scripts/validate_monomers.py --design-ids trem2_L80_s0_20260228 trem2_L100_s100_20260228
"""

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("trem2-monomer-validation")

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


@app.function(
    gpu="A100",
    image=mosaic_image,
    timeout=1800,  # 30 min per monomer refold
    volumes={"/results": results_volume},
)
def validate_monomer(design_id: str, binder_sequence: str, binder_length: int) -> dict:
    """Refold a binder sequence alone and compute validation metrics."""
    import sys
    sys.path.insert(0, "/opt/mosaic/src")
    sys.path.insert(0, "/opt/mosaic")

    import jax
    import jax.numpy as jnp
    import numpy as np

    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain
    from mosaic.common import TOKENS

    print(f"\n{'='*60}")
    print(f"MONOMER VALIDATION: {design_id}")
    print(f"Sequence ({binder_length} AA): {binder_sequence}")
    print(f"{'='*60}")

    t_start = time.time()

    # Load model
    print("Loading Boltz-2...")
    model = Boltz2()

    one_hot_seq = jax.nn.one_hot(
        jnp.array([TOKENS.index(c) for c in binder_sequence]), 20
    )

    # ── Refold binder as monomer ──
    print("Generating monomer features...")
    mono_features, mono_writer = model.target_only_features(
        chains=[TargetChain(sequence=binder_sequence, use_msa=False)]
    )

    print("Refolding as monomer (3 recycling steps)...")
    mono_pred = model.predict(
        PSSM=one_hot_seq,
        features=mono_features,
        writer=mono_writer,
        recycling_steps=3,
        key=jax.random.key(0),
    )

    mono_plddt = float(mono_pred.plddt.mean())
    print(f"  Monomer pLDDT: {mono_plddt:.4f}")

    # ── Load complex structure and compute RMSD ──
    complex_pdb_path = f"/results/{design_id}/{design_id}.pdb"
    rmsd = float("nan")

    if os.path.exists(complex_pdb_path):
        try:
            import gemmi

            # Extract binder backbone CA coords from complex
            complex_st = gemmi.read_pdb(complex_pdb_path)
            complex_model = complex_st[0]
            binder_chain = complex_model[0]  # First chain is always the binder
            complex_ca = []
            for residue in binder_chain:
                ca = residue.find_atom("CA", "\0")
                if ca:
                    complex_ca.append([ca.pos.x, ca.pos.y, ca.pos.z])
            complex_ca = np.array(complex_ca)

            # Extract binder backbone CA coords from monomer
            mono_pdb_str = mono_pred.st.make_pdb_string()
            mono_st = gemmi.read_pdb_string(mono_pdb_str)
            mono_model = mono_st[0]
            mono_chain = mono_model[0]
            mono_ca = []
            for residue in mono_chain:
                ca = residue.find_atom("CA", "\0")
                if ca:
                    mono_ca.append([ca.pos.x, ca.pos.y, ca.pos.z])
            mono_ca = np.array(mono_ca)

            # Compute aligned RMSD using Mosaic's Kabsch implementation
            if len(complex_ca) == len(mono_ca) and len(complex_ca) > 0:
                from mosaic.util import calculate_rmsd
                rmsd = float(calculate_rmsd(
                    jnp.array(complex_ca),
                    jnp.array(mono_ca),
                ))
                print(f"  Complex vs. monomer RMSD: {rmsd:.2f} Å")
            else:
                print(f"  WARNING: CA count mismatch (complex={len(complex_ca)}, mono={len(mono_ca)})")
        except Exception as e:
            print(f"  WARNING: RMSD calculation failed: {e}")
    else:
        print(f"  WARNING: Complex PDB not found at {complex_pdb_path}")

    # ── Save monomer PDB ──
    out_dir = f"/results/{design_id}"
    os.makedirs(out_dir, exist_ok=True)
    mono_pdb_str = mono_pred.st.make_pdb_string()
    with open(f"{out_dir}/{design_id}_monomer.pdb", "w") as f:
        f.write(mono_pdb_str)

    total_time = time.time() - t_start

    # ── Determine pass/fail ──
    plddt_pass = mono_plddt > 80.0
    rmsd_pass = rmsd < 2.0 if not np.isnan(rmsd) else False
    passed = plddt_pass and rmsd_pass

    result = {
        "design_id": design_id,
        "binder_sequence": binder_sequence,
        "binder_length": binder_length,
        "monomer_plddt": mono_plddt,
        "monomer_complex_rmsd": rmsd if not np.isnan(rmsd) else None,
        "plddt_pass": plddt_pass,
        "rmsd_pass": rmsd_pass,
        "passed": passed,
        "validation_time_seconds": total_time,
    }

    with open(f"{out_dir}/{design_id}_monomer_validation.json", "w") as f:
        json.dump(result, f, indent=2)

    results_volume.commit()

    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    print(f"    pLDDT: {mono_plddt:.4f} ({'PASS' if plddt_pass else 'FAIL'} > 80)")
    rmsd_str = f"{rmsd:.2f}" if not np.isnan(rmsd) else "N/A"
    print(f"    RMSD:  {rmsd_str} ({'PASS' if rmsd_pass else 'FAIL'} < 2.0)")
    print(f"    Time:  {total_time:.1f}s")

    return result


@app.local_entrypoint()
def main(
    batch: str = "",
    design_ids: list[str] = [],
    top: int = 10,
    output: str = "results/designs/monomer_validation.json",
):
    """Validate top designs by monomer refolding."""

    # Collect designs to validate
    designs_to_validate = []

    if batch:
        print(f"Loading batch results from: {batch}")
        with open(batch) as f:
            batch_results = json.load(f)

        # Filter: cysteine-free only, sort by ranking loss
        batch_results = [r for r in batch_results if r.get("n_cysteines", 0) == 0]
        batch_results.sort(key=lambda r: r.get("ranking_loss", float("inf")))

        for r in batch_results[:top]:
            designs_to_validate.append({
                "design_id": r["design_id"],
                "binder_sequence": r["binder_sequence"],
                "binder_length": r["binder_length"],
            })

    elif design_ids:
        # Load individual design JSONs from results volume
        for did in design_ids:
            json_path = Path(f"results/designs/{did}.json")
            if json_path.exists():
                with open(json_path) as f:
                    data = json.load(f)
                designs_to_validate.append({
                    "design_id": data["design_id"],
                    "binder_sequence": data["binder_sequence"],
                    "binder_length": data["binder_length"],
                })
            else:
                print(f"  WARNING: Design not found: {did}")

    if not designs_to_validate:
        print("No designs to validate! Provide --batch or --design-ids.")
        return

    total = len(designs_to_validate)
    print(f"\nValidating {total} designs by monomer refolding...")

    # Launch in parallel on Modal
    results = []
    for result in validate_monomer.map(
        [d["design_id"] for d in designs_to_validate],
        [d["binder_sequence"] for d in designs_to_validate],
        [d["binder_length"] for d in designs_to_validate],
    ):
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        rmsd_str = f"{result['monomer_complex_rmsd']:.2f}" if result["monomer_complex_rmsd"] is not None else "N/A"
        print(
            f"  [{len(results)}/{total}] {result['design_id']} "
            f"| pLDDT={result['monomer_plddt']:.4f} "
            f"| RMSD={rmsd_str} "
            f"| {status}"
        )

    # Summary
    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"MONOMER VALIDATION COMPLETE")
    print(f"  Passed: {n_passed}/{total}")
    print(f"{'='*60}")

    print(f"\n{'Design ID':<45}{'pLDDT':<10}{'RMSD':<10}{'Status':<8}")
    print(f"{'-'*70}")
    for r in results:
        rmsd_str = f"{r['monomer_complex_rmsd']:.2f}" if r["monomer_complex_rmsd"] is not None else "N/A"
        status = "PASS" if r["passed"] else "FAIL"
        print(
            f"  {r['design_id']:<43} "
            f"{r['monomer_plddt']:<10.4f}"
            f"{rmsd_str:<10}"
            f"{status}"
        )

    # Save results
    os.makedirs(Path(output).parent, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved validation results: {output}")
