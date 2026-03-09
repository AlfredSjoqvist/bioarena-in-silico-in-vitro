# bioArena: De Novo Protein Binder Design for TREM2

Computational protein design pipeline for generating high-affinity binders targeting **TREM2** (Triggering Receptor Expressed on Myeloid cells 2), a key therapeutic target in Alzheimer's disease and neuroinflammation. Built for the [bioArena In Silico to In Vitro hackathon](https://www.bioarena.ai/), where top-scoring designs are experimentally validated by [Adaptyv Bio](https://www.adaptyvbio.com/).

## Why TREM2?

TREM2 is an immune receptor on microglia whose loss-of-function variants (R47H, R62H) are among the strongest genetic risk factors for late-onset Alzheimer's disease. Designing de novo binders that engage TREM2's ectodomain could unlock new therapeutic modalities beyond traditional antibody discovery.

## Approach

This project implements a **gradient-based protein design pipeline** using [Boltz-2](https://github.com/jwohlwend/boltz) differentiable structure prediction, adapted from the strategy that won the [Adaptyv Nipah competition](https://www.adaptyvbio.com/).

### Design Pipeline

```
Random Sequence → Soft PSSM Optimization → Logspace Sharpening → Hard Sequence → Refold & Rank
     (init)         (100 steps, APGM)        (50+15 steps)        (extract)     (ipSAE scoring)
```

**Stage 1** — Initialize a soft position-specific scoring matrix (PSSM) and optimize it against Boltz-2's predicted structure using simplex accelerated projected gradient descent with momentum.

**Stage 2–3** — Progressively sharpen the PSSM toward a discrete amino acid sequence using L2 regularization in logspace (scales 1.25 → 1.4).

**Stage 4** — Extract the hard sequence, refold the complex with increased diffusion samples and recycling steps, then score with ipSAE.

### Key Design Decisions

- **Anti-Ig-fold loss** (helix bias weight = 2.5): TREM2 itself is an Ig-fold protein. Without explicit secondary structure steering, the optimizer converges on Ig-like binders that co-fold poorly. Helix loss breaks this local minimum.
- **Triple cysteine prevention**: MPNN bias (−1e6) + PSSM hard clamp + post-filter. Disulfide bonds complicate E. coli expression and reduce experimental success rates.
- **Inverse folding recovery weight = 10.0**: Prevents the optimizer from gaming structural metrics with physically implausible sequences.
- **Binder lengths 70–85 AA**: Shorter than typical designs because TREM2's ectodomain is only 87 residues. Keeps the binder–target size ratio reasonable.

### Scoring: ipSAE

Designs are ranked by **ipSAE** (interaction prediction Score from Aligned Errors), which improves on AlphaFold's ipTM by:
1. Only counting residue pairs with PAE below a confidence cutoff
2. Computing the distance normalization factor from high-confidence interface residues, not full chain length

Primary metric is `ipSAE_min = min(binder→target, target→binder)` to catch asymmetric false positives. Threshold: >0.6 likely binder, >0.8 high confidence.

## Architecture

```
scripts/
├── design_binder.py        # Core Boltz-2 gradient optimization (single design)
├── design_modal.py         # Parallel GPU execution on Modal (batch designs)
├── design_turbo.py         # Maximum-diversity launcher (4 variant strategies, 40 designs)
├── boltzgen_modal.py       # Fast diffusion backbone sampling + MPNN (complementary path)
├── score_ipsae.py          # ipSAE scoring wrapper
├── validate_monomers.py    # Monomer refold validation (pLDDT > 80, RMSD < 2.0 Å)
├── rank_and_select.py      # Multi-metric ranking with diversity filtering
└── select_final.py         # Final top-10 selection with hard filters

configs/
├── design_config.yaml      # Optimization hyperparameters, loss weights, submission criteria
└── scoring_config.yaml     # ipSAE cutoffs and selection thresholds

frontend/                   # React UI for real-time design submission and scoring
├── index.html
└── server.js               # Node.js proxy to Modal API endpoints

data/trem2/
├── 5UD7.pdb                # Wild-type apo TREM2 (2.2 Å crystal structure)
├── 6YYE.pdb                # TREM2 + scFv complex (reference epitope)
└── trem2_chainA.pdb        # Extracted target chain
```

### Compute Infrastructure

All GPU workloads run on **[Modal](https://modal.com/)** serverless GPUs (A100/B200). Designs execute in parallel via `modal.Function.map()` with persistent volumes for result coordination. Each design takes ~30 minutes and costs ~$3 on B200.

### Three-Phase Validation

1. **Design** — Gradient-based sequence optimization against Boltz-2 structure prediction
2. **Monomer validation** — Refold each binder alone to verify independent folding (pLDDT > 80, backbone RMSD < 2.0 Å vs. complex conformation)
3. **Selection** — Composite ranking (0.6×ipSAE + 0.3×ipTM + 0.1×pLDDT), greedy diversity filtering (Levenshtein distance ≥ 10), hard filters (zero cysteines, no Ig-fold motifs)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Modal for GPU compute
modal setup

# Download TREM2 structures
python scripts/download_trem2.py

# Run a single design locally (requires GPU)
python scripts/design_binder.py --binder-length 80 --seed 42

# Run 20 parallel designs on Modal
modal run scripts/design_modal.py --n-designs 20 --binder-length 80

# Score all designs with ipSAE
python scripts/score_ipsae.py --input results/designs/ --output results/scores/

# Rank and select top 10 for submission
python scripts/rank_and_select.py --scores results/scores/ --top 10 --output results/submissions/
```

## Target Structure

| Property | Value |
|----------|-------|
| **Protein** | TREM2 ectodomain (Ig-like V-type domain) |
| **PDB** | 5UD7 (apo, 2.2 Å), 6YYE (scFv complex) |
| **Residues** | 19–131 (87 AA hackathon construct) |
| **Fold** | Beta-sandwich, 9 beta-strands |
| **Key epitope** | Arg-rich basic patch (R47, R62, R76, R77) |
| **Disease variants** | R47H, R62H (Alzheimer's risk) |

## Tools & References

- **[Boltz-2](https://github.com/jwohlwend/boltz)** — Differentiable structure prediction
- **[Mosaic](https://github.com/escalante-bio/mosaic)** — Gradient-based protein design framework
- **[IPSAE](https://github.com/DunbrackLab/IPSAE)** — Interface prediction scoring
- **[Modal](https://modal.com/)** — Serverless GPU compute
- **[Adaptyv Bio](https://www.adaptyvbio.com/)** — Experimental validation partner

## License

MIT
