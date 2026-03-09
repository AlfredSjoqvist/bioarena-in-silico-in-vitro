"""
Boltz-2 TREM2 Scoring API on Modal.
Each binder gets its own A100 GPU and scores run in parallel.
CORS enabled for browser access.

Deploy:  modal deploy scripts/boltz2_api.py
Test:    modal serve scripts/boltz2_api.py
"""

import json
import os
import subprocess
import tempfile
import time

import modal

app = modal.App("boltz2-trem2-api")

boltz_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("boltz", "numpy", "gemmi", "fastapi[standard]")
    .run_commands("git clone https://github.com/DunbrackLab/IPSAE.git /opt/IPSAE")
    .env({
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
    })
)

web_image = modal.Image.debian_slim().pip_install("fastapi[standard]")

model_cache = modal.Volume.from_name("boltz2-model-cache", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


def run_ipsae(pae_file: str, structure_file: str) -> dict:
    """Run IPSAE scoring and parse output."""
    print(f"  [IPSAE] Running: python /opt/IPSAE/ipsae.py {pae_file} {structure_file} 10 10")
    cmd = ["python", "/opt/IPSAE/ipsae.py", pae_file, structure_file, "10", "10"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(f"  [IPSAE] returncode={result.returncode}")
    if result.stdout:
        print(f"  [IPSAE] stdout: {result.stdout[:500]}")
    if result.stderr:
        print(f"  [IPSAE] stderr: {result.stderr[:500]}")

    scores = {}
    if result.returncode != 0:
        scores["error"] = result.stderr[:500]
        return scores

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                scores[parts[0].strip()] = float(parts[1].strip())
            except (ValueError, IndexError):
                pass
        elif ":" in line:
            key, val = line.split(":", 1)
            try:
                scores[key.strip()] = float(val.strip())
            except ValueError:
                pass

    if not scores:
        scores["raw_output"] = result.stdout[:1000]

    print(f"  [IPSAE] Parsed scores: {scores}")
    return scores


@app.function(
    gpu="A100",
    image=boltz_image,
    timeout=1800,
    volumes={"/cache": model_cache},
)
def score_single(binder_seq: str, binder_name: str, num_samples: int) -> dict:
    """Score a single binder on its own GPU."""
    t_start = time.time()
    print(f"[{binder_name}] === START SCORING ===")
    print(f"[{binder_name}] Seq: {binder_seq[:40]}... ({len(binder_seq)} aa)")
    print(f"[{binder_name}] Samples: {num_samples}")

    work_dir = tempfile.mkdtemp()
    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")
    os.makedirs(input_dir)
    os.makedirs(output_dir)

    fasta_path = os.path.join(input_dir, f"{binder_name}.fasta")
    with open(fasta_path, "w") as f:
        f.write(f">A|protein|binder\n{binder_seq}\n")
        f.write(f">B|protein|TREM2\n{TREM2_SEQUENCE}\n")

    print(f"[{binder_name}] FASTA written to {fasta_path}")

    cmd = [
        "boltz", "predict", fasta_path,
        "--out_dir", output_dir,
        "--diffusion_samples", str(num_samples),
        "--cache", "/cache",
        "--override",
    ]
    print(f"[{binder_name}] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)

    print(f"[{binder_name}] Boltz-2 returncode={proc.returncode}")
    if proc.stdout:
        print(f"[{binder_name}] Boltz-2 stdout (last 500): ...{proc.stdout[-500:]}")
    if proc.stderr:
        print(f"[{binder_name}] Boltz-2 stderr (last 500): ...{proc.stderr[-500:]}")

    if proc.returncode != 0:
        return {
            "error": f"Boltz-2 failed: {proc.stderr[:500]}",
            "binder_name": binder_name,
            "binder_seq": binder_seq,
            "binder_length": len(binder_seq),
            "time_seconds": round(time.time() - t_start, 1),
        }

    # List ALL output files for debugging
    print(f"[{binder_name}] === OUTPUT FILE TREE ===")
    all_files = []
    for root, dirs, files in os.walk(output_dir):
        for fname in files:
            full = os.path.join(root, fname)
            size = os.path.getsize(full)
            rel = os.path.relpath(full, output_dir)
            print(f"  {rel} ({size} bytes)")
            all_files.append((root, fname, full, size))

    if not all_files:
        print(f"[{binder_name}] WARNING: No output files found!")
        return {
            "error": "Boltz-2 produced no output files",
            "binder_name": binder_name,
            "binder_seq": binder_seq,
            "binder_length": len(binder_seq),
            "boltz_stdout": proc.stdout[-1000:],
            "time_seconds": round(time.time() - t_start, 1),
        }

    # Try to run IPSAE on all CIF+PAE pairs
    all_scores = []
    cif_files = [(r, f, p) for r, f, p, _ in all_files if f.endswith(".cif")]
    npz_files = [(r, f, p) for r, f, p, _ in all_files if f.endswith(".npz")]
    json_files = [(r, f, p) for r, f, p, _ in all_files if f.endswith(".json")]

    print(f"[{binder_name}] Found: {len(cif_files)} CIF, {len(npz_files)} NPZ, {len(json_files)} JSON")

    # Strategy 1: Match CIF to PAE by name pattern
    for cif_root, cif_name, cif_path in cif_files:
        sample_id = cif_name.replace(".cif", "")
        pae_candidates = [
            os.path.join(cif_root, f"pae_{sample_id}.npz"),
            os.path.join(cif_root, f"{sample_id}_pae.npz"),
            os.path.join(cif_root, cif_name.replace(".cif", "_pae.npz")),
        ]
        # Also check parent/sibling dirs
        parent = os.path.dirname(cif_root)
        pae_candidates += [
            os.path.join(parent, f"pae_{sample_id}.npz"),
            os.path.join(cif_root, "pae", f"{sample_id}.npz"),
        ]

        matched = False
        for pae_path in pae_candidates:
            if os.path.exists(pae_path):
                print(f"[{binder_name}] Matched: {cif_name} <-> {os.path.basename(pae_path)}")
                scores = run_ipsae(pae_path, cif_path)
                scores["sample_file"] = cif_name
                scores["pae_file"] = os.path.basename(pae_path)
                all_scores.append(scores)
                matched = True
                break

        if not matched:
            print(f"[{binder_name}] WARNING: No PAE match for {cif_name}")

    # Strategy 2: If no matches, try brute-force pairing by order
    if not all_scores and npz_files and cif_files:
        print(f"[{binder_name}] Fallback: brute-force pairing {len(npz_files)} NPZ with {len(cif_files)} CIF")
        npz_sorted = sorted(npz_files, key=lambda x: x[1])
        cif_sorted = sorted(cif_files, key=lambda x: x[1])
        for (nr, nf, np_), (cr, cf, cp) in zip(npz_sorted, cif_sorted):
            print(f"[{binder_name}] Trying: {nf} + {cf}")
            scores = run_ipsae(np_, cp)
            scores["sample_file"] = cf
            scores["pae_file"] = nf
            all_scores.append(scores)

    # Strategy 3: Check if there's a JSON confidence file instead
    if not all_scores and json_files:
        print(f"[{binder_name}] Checking JSON files for confidence scores...")
        for jr, jf, jp in json_files:
            try:
                with open(jp) as f:
                    data = json.load(f)
                print(f"[{binder_name}] JSON {jf} keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                if isinstance(data, dict):
                    for key in ["confidence_score", "iptm", "ptm", "ranking_score"]:
                        if key in data:
                            print(f"[{binder_name}]   {key}: {data[key]}")
            except Exception as e:
                print(f"[{binder_name}] JSON parse error {jf}: {e}")

    print(f"[{binder_name}] === SCORING SUMMARY ===")
    print(f"[{binder_name}] Total IPSAE results: {len(all_scores)}")
    for i, s in enumerate(all_scores):
        print(f"[{binder_name}]   Sample {i}: {s}")

    best_ipsae = 0.0
    best_sample = {}
    for s in all_scores:
        val = s.get("ipSAE", s.get("ipsae", s.get("IPSAE", 0)))
        if isinstance(val, (int, float)) and val > best_ipsae:
            best_ipsae = val
            best_sample = s

    total_time = time.time() - t_start
    print(f"[{binder_name}] FINAL: ipSAE={best_ipsae:.4f} in {total_time:.0f}s")

    return {
        "binder_name": binder_name,
        "binder_seq": binder_seq,
        "binder_length": len(binder_seq),
        "target": "TREM2",
        "target_length": len(TREM2_SEQUENCE),
        "best_ipsae": best_ipsae,
        "best_sample": best_sample,
        "all_samples": all_scores,
        "num_samples": num_samples,
        "time_seconds": round(total_time, 1),
        "verdict": "LIKELY BINDER" if best_ipsae >= 0.6 else "UNLIKELY BINDER" if best_ipsae > 0 else "NO SCORE",
        "debug": {
            "output_files": [os.path.relpath(p, output_dir) for _, _, p, _ in all_files],
            "cif_count": len(cif_files),
            "npz_count": len(npz_files),
            "json_count": len(json_files),
            "ipsae_results_count": len(all_scores),
        },
    }


@app.function(image=web_image, timeout=1800)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="Boltz-2 TREM2 Scoring API")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health():
        return {"status": "ok", "target": "TREM2", "model": "Boltz-2", "parallel": True}

    @api.post("/score")
    def score(body: dict):
        num_samples = body.get("num_samples", 5)

        # Batch mode
        if "binders" in body:
            binders = body["binders"]
            seqs, names = [], []
            for b in binders:
                seq = b.get("binder_seq", "").strip().upper()
                name = b.get("binder_name", f"binder_{len(seqs)}")
                if not seq:
                    continue
                invalid = set(seq) - set("ARNDCQEGHILKMFPSTWYV")
                if invalid:
                    continue
                seqs.append(seq)
                names.append(name)

            if not seqs:
                return {"error": "No valid binder sequences provided"}

            t_start = time.time()
            results = list(score_single.map(seqs, names, [num_samples] * len(seqs)))
            return {
                "results": results,
                "num_binders": len(results),
                "total_time_seconds": round(time.time() - t_start, 1),
            }

        # Single mode
        binder_seq = body.get("binder_seq", "").strip().upper()
        binder_name = body.get("binder_name", "binder")
        if not binder_seq:
            return {"error": "binder_seq is required"}
        invalid = set(binder_seq) - set("ARNDCQEGHILKMFPSTWYV")
        if invalid:
            return {"error": f"Invalid amino acids: {invalid}"}
        return score_single.remote(binder_seq, binder_name, num_samples)

    return api
