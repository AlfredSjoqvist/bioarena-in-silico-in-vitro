#!/usr/bin/env python3
"""
Fast Boltz-2 scoring pipeline on Modal.
Give it a binder sequence, get back ipSAE score in ~5-10 min.

Usage:
    # Score a single binder
    modal run scripts/boltz2_score_modal.py --binder-seq "MDEEKRLLAVFANFDKSVSE..."

    # Score multiple binders from a file (one sequence per line)
    modal run scripts/boltz2_score_modal.py --fasta-file binders.fasta

    # Score with more samples (slower but more accurate)
    modal run scripts/boltz2_score_modal.py --binder-seq "MDEE..." --num-samples 10
"""

import json
import os
import time
from pathlib import Path

import modal

app = modal.App("boltz2-trem2-scoring")

# Container image with Boltz-2 + IPSAE dependencies
boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    .pip_install(
        "boltz",
        "numpy",
        "biopython",
        "gemmi",
    )
    .run_commands(
        "git clone https://github.com/DunbrackLab/IPSAE.git /opt/IPSAE",
    )
    .env({
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    })
)

# Cache model weights across runs
model_cache = modal.Volume.from_name("boltz2-model-cache", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


@app.function(
    gpu="A100",
    image=boltz_image,
    timeout=1800,
    volumes={"/cache": model_cache},
)
def score_binder(binder_seq: str, num_samples: int = 5, binder_name: str = "binder") -> dict:
    """Score a single binder sequence against TREM2 using Boltz-2."""
    import subprocess
    import tempfile
    import numpy as np

    t_start = time.time()
    print(f"Scoring {binder_name}: {binder_seq[:40]}... ({len(binder_seq)} aa)")

    # Create temp directory for this job
    work_dir = tempfile.mkdtemp()
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(input_dir)
    os.makedirs(output_dir)

    # Write FASTA input for Boltz-2 (binder + TREM2 as two chains)
    fasta_path = os.path.join(input_dir, f"{binder_name}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">A|protein|binder\n{binder_seq}\n")
        f.write(f">B|protein|TREM2\n{TREM2_SEQUENCE}\n")

    # Run Boltz-2 prediction
    print(f"Running Boltz-2 ({num_samples} samples)...")
    cmd = [
        "boltz", "predict",
        fasta_path,
        "--out_dir", output_dir,
        "--num_samples", str(num_samples),
        "--cache", "/cache",
        "--override",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        print(f"Boltz-2 stderr: {result.stderr}")
        return {"error": result.stderr, "binder_name": binder_name}

    print(f"Boltz-2 finished in {time.time() - t_start:.0f}s")

    # Find output files
    pred_dir = os.path.join(output_dir, "boltz_results", binder_name, "predictions", binder_name)
    if not os.path.exists(pred_dir):
        # Try to find the actual output directory
        for root, dirs, files in os.walk(output_dir):
            for f_name in files:
                if f_name.endswith(".cif"):
                    pred_dir = root
                    break

    # Collect all sample scores
    scores_all = []
    for sample_idx in range(num_samples):
        # Look for structure and PAE files
        cif_candidates = [
            os.path.join(pred_dir, f"{binder_name}_model_{sample_idx}.cif"),
            os.path.join(pred_dir, f"model_{sample_idx}.cif"),
        ]
        pae_candidates = [
            os.path.join(pred_dir, f"pae_{binder_name}_model_{sample_idx}.npz"),
            os.path.join(pred_dir, f"pae_model_{sample_idx}.npz"),
        ]

        cif_file = None
        pae_file = None
        for c in cif_candidates:
            if os.path.exists(c):
                cif_file = c
                break
        for p in pae_candidates:
            if os.path.exists(p):
                pae_file = p
                break

        if not cif_file or not pae_file:
            continue

        # Run IPSAE scoring
        ipsae_cmd = [
            "python", "/opt/IPSAE/ipsae.py",
            pae_file, cif_file, "10", "10",
        ]
        ipsae_result = subprocess.run(ipsae_cmd, capture_output=True, text=True)

        if ipsae_result.returncode == 0:
            # Parse IPSAE output
            sample_scores = {"sample": sample_idx}
            for line in ipsae_result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Try tab-separated format
                parts = line.split("\t")
                if len(parts) >= 2:
                    try:
                        sample_scores[parts[0].strip()] = float(parts[1].strip())
                    except (ValueError, IndexError):
                        pass
                # Try colon-separated format
                elif ":" in line:
                    key, val = line.split(":", 1)
                    try:
                        sample_scores[key.strip()] = float(val.strip())
                    except ValueError:
                        pass

            scores_all.append(sample_scores)
            print(f"  Sample {sample_idx}: {sample_scores}")
        else:
            print(f"  Sample {sample_idx} IPSAE error: {ipsae_result.stderr[:200]}")

    if not scores_all:
        # If IPSAE parsing failed, try to get raw confidence from Boltz
        print("IPSAE parsing failed, returning raw output")
        return {
            "binder_name": binder_name,
            "binder_seq": binder_seq,
            "binder_length": len(binder_seq),
            "error": "Could not parse IPSAE scores",
            "raw_stdout": ipsae_result.stdout[:500] if 'ipsae_result' in dir() else "no output",
            "time_seconds": time.time() - t_start,
        }

    # Aggregate scores across samples (take best sample)
    best_ipsae = 0.0
    best_sample = {}
    for s in scores_all:
        ipsae_val = s.get("ipSAE", s.get("ipsae", 0))
        if ipsae_val > best_ipsae:
            best_ipsae = ipsae_val
            best_sample = s

    total_time = time.time() - t_start

    result = {
        "binder_name": binder_name,
        "binder_seq": binder_seq,
        "binder_length": len(binder_seq),
        "best_ipsae": best_ipsae,
        "best_sample": best_sample,
        "all_samples": scores_all,
        "num_samples": num_samples,
        "time_seconds": total_time,
    }

    print(f"\n{'='*50}")
    print(f"RESULT: {binder_name}")
    print(f"  Best ipSAE: {best_ipsae:.4f}")
    print(f"  Time: {total_time:.0f}s")
    print(f"{'='*50}")

    return result


@app.local_entrypoint()
def main(
    binder_seq: str = "",
    fasta_file: str = "",
    num_samples: int = 5,
):
    """Score binder(s) against TREM2."""

    binders = []

    if binder_seq:
        binders.append(("binder_0", binder_seq.strip()))

    elif fasta_file:
        # Parse FASTA file
        current_name = None
        current_seq = []
        with open(fasta_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name and current_seq:
                        binders.append((current_name, "".join(current_seq)))
                    current_name = line[1:].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(line)
            if current_name and current_seq:
                binders.append((current_name, "".join(current_seq)))
    else:
        print("ERROR: Provide --binder-seq or --fasta-file")
        return

    print(f"Scoring {len(binders)} binder(s) against TREM2 with Boltz-2 ({num_samples} samples)")
    print(f"Target: TREM2 ({len(TREM2_SEQUENCE)} aa)")
    print()

    if len(binders) == 1:
        # Single binder - run directly
        name, seq = binders[0]
        result = score_binder.remote(seq, num_samples=num_samples, binder_name=name)
        print(json.dumps(result, indent=2))
    else:
        # Multiple binders - run in parallel
        results = []
        for result in score_binder.map(
            [seq for _, seq in binders],
            [num_samples] * len(binders),
            [name for name, _ in binders],
        ):
            results.append(result)
            ipsae = result.get("best_ipsae", 0)
            print(f"  {result['binder_name']}: ipSAE={ipsae:.4f} ({result.get('time_seconds', 0):.0f}s)")

        # Sort by ipSAE
        results.sort(key=lambda r: r.get("best_ipsae", 0), reverse=True)

        print(f"\n{'='*60}")
        print(f"LEADERBOARD ({len(results)} binders)")
        print(f"{'='*60}")
        for i, r in enumerate(results):
            ipsae = r.get("best_ipsae", 0)
            marker = " <<<" if ipsae >= 0.6 else ""
            print(f"  #{i+1}: {r['binder_name']} | ipSAE={ipsae:.4f} | {r['binder_length']}aa{marker}")

        # Save results
        os.makedirs("results/scores", exist_ok=True)
        out_path = f"results/scores/boltz2_scores_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")
