"""
BoltzGen TREM2 Binder Design API on Modal.
Generates novel binder sequences for TREM2 using Mosaic's Boltz-2 gradient optimization.
Each design runs on its own A100 GPU in parallel. CORS enabled.

Deploy:  modal deploy scripts/boltzgen_api.py
Test:    modal serve scripts/boltzgen_api.py
"""

import json
import os
import time

import modal

app = modal.App("boltzgen-trem2-design")

# Heavy image with Mosaic + JAX CUDA + all dependencies
mosaic_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install("git", "curl", "build-essential")
    .pip_install("uv")
    .run_commands(
        "git clone https://github.com/escalante-bio/mosaic.git /opt/mosaic",
        "cd /opt/mosaic && uv pip install --system --group jax-cuda -e '.'",
    )
    .pip_install("fastapi[standard]")
    .env({
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        "PYTHONPATH": "/opt/mosaic/src:/opt/mosaic",
    })
)

web_image = modal.Image.debian_slim().pip_install("fastapi[standard]")

model_cache = modal.Volume.from_name("boltzgen-model-cache", create_if_missing=True)

TREM2_SEQUENCE = (
    "AHNTTVFQGVAGQSLQVSCPYDSMKHWGRRKAWCRQLGE"
    "LLDRGDKQASDQQLGFVTQREAQPPPKAVLAHSQSLDL"
    "SKIPKTKVMF"
)


@app.function(
    gpu="A100:1",
    image=mosaic_image,
    timeout=3600,
    volumes={"/cache": model_cache},
)
def design_single(
    binder_length: int,
    seed: int,
    stage1_steps: int,
    stage2_steps: int,
    stage3_steps: int,
    design_samples: int,
    ranking_samples: int,
) -> dict:
    """Run a single binder design on one GPU."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from mosaic.models.boltz2 import Boltz2
    from mosaic.structure_prediction import TargetChain
    from mosaic.optimizers import simplex_APGM
    from mosaic.common import TOKENS as MOSAIC_TOKENS
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    import mosaic.losses.structure_prediction as sp

    design_id = f"trem2_L{binder_length}_s{seed}"
    t_start = time.time()

    try:
        print(f"[{design_id}] Loading models...")
        model = Boltz2()
        mpnn = ProteinMPNN.from_pretrained()

        print(f"[{design_id}] Generating features for binder_length={binder_length}...")
        features, structure_writer = model.binder_features(
            binder_length=binder_length,
            chains=[TargetChain(TREM2_SEQUENCE, use_msa=False)],
        )

        # Cysteine bias
        cys_idx = MOSAIC_TOKENS.index("C")
        cys_bias = np.zeros((binder_length, 20))
        cys_bias[:, cys_idx] = -1e6

        # Design loss
        print(f"[{design_id}] Building loss function...")
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
            num_samples=design_samples,
        )

        # Stage 1: Soft PSSM
        print(f"[{design_id}] Stage 1: Soft PSSM ({stage1_steps} steps)")
        t1 = time.time()
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
        print(f"[{design_id}]   Stage 1 done ({time.time()-t1:.0f}s)")

        # Stage 2: Sharpen
        print(f"[{design_id}] Stage 2: Sharpen ({stage2_steps} steps)")
        t2 = time.time()
        PSSM_sharp, _ = simplex_APGM(
            loss_function=design_loss,
            n_steps=stage2_steps,
            x=PSSM,
            stepsize=0.5 * np.sqrt(binder_length),
            scale=1.25,
            logspace=True,
            momentum=0.0,
        )
        print(f"[{design_id}]   Stage 2 done ({time.time()-t2:.0f}s)")

        # Stage 3: Final sharpen
        print(f"[{design_id}] Stage 3: Final sharpen ({stage3_steps} steps)")
        t3 = time.time()
        PSSM_final, _ = simplex_APGM(
            loss_function=design_loss,
            n_steps=stage3_steps,
            x=PSSM_sharp,
            stepsize=0.5 * np.sqrt(binder_length),
            scale=1.4,
            logspace=True,
            momentum=0.0,
        )
        print(f"[{design_id}]   Stage 3 done ({time.time()-t3:.0f}s)")

        # Extract sequence
        binder_seq = "".join(MOSAIC_TOKENS[i] for i in PSSM_final.argmax(-1))
        print(f"[{design_id}] Sequence: {binder_seq[:50]}...")

        # Stage 4: Refold and get confidence
        print(f"[{design_id}] Stage 4: Refolding ({ranking_samples} samples)")
        t4 = time.time()
        iptm = 0.0
        plddt = 0.0
        try:
            prediction = model.predict(
                PSSM=jax.nn.one_hot(
                    jnp.array([MOSAIC_TOKENS.index(c) for c in binder_seq]), 20
                ),
                features=features,
                writer=structure_writer,
                recycling_steps=3,
                key=jax.random.key(seed),
            )
            iptm = float(prediction.iptm)
            plddt = float(prediction.plddt.mean())
            print(f"[{design_id}]   ipTM={iptm:.4f}, pLDDT={plddt:.1f} ({time.time()-t4:.0f}s)")
        except Exception as e:
            print(f"[{design_id}]   Refolding error: {e}")

        design_time = time.time() - t_start
        print(f"[{design_id}] DONE in {design_time:.0f}s")

        return {
            "design_id": design_id,
            "binder_seq": binder_seq,
            "binder_length": binder_length,
            "seed": seed,
            "iptm": iptm,
            "plddt_mean": plddt,
            "design_time_seconds": round(design_time, 1),
            "params": {
                "stage1_steps": stage1_steps,
                "stage2_steps": stage2_steps,
                "stage3_steps": stage3_steps,
                "design_samples": design_samples,
                "ranking_samples": ranking_samples,
            },
        }

    except Exception as e:
        import traceback
        return {
            "design_id": design_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "design_time_seconds": round(time.time() - t_start, 1),
        }


@app.function(image=web_image, timeout=3600)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI(title="BoltzGen TREM2 Design API")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @api.get("/health")
    def health():
        return {"status": "ok", "service": "BoltzGen TREM2 Designer", "model": "Mosaic/Boltz-2", "parallel": True}

    @api.post("/design")
    def design(body: dict):
        binder_length = body.get("binder_length", 100)
        seed = body.get("seed", 42)
        num_designs = min(body.get("num_designs", 1), 5)
        stage1_steps = body.get("stage1_steps", 100)
        stage2_steps = body.get("stage2_steps", 50)
        stage3_steps = body.get("stage3_steps", 15)
        design_samples = body.get("design_samples", 4)
        ranking_samples = body.get("ranking_samples", 6)

        if binder_length < 40 or binder_length > 250:
            return {"error": "binder_length must be between 40 and 250"}

        t_start = time.time()
        seeds = [seed + i for i in range(num_designs)]

        results = list(design_single.map(
            [binder_length] * num_designs,
            seeds,
            [stage1_steps] * num_designs,
            [stage2_steps] * num_designs,
            [stage3_steps] * num_designs,
            [design_samples] * num_designs,
            [ranking_samples] * num_designs,
        ))

        return {
            "target": "TREM2",
            "designs": results,
            "num_designs": len(results),
            "total_time_seconds": round(time.time() - t_start, 1),
        }

    return api
