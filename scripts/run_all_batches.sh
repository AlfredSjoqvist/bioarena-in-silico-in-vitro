#!/bin/bash
# TREM2 v3 Design Campaign — Full Batch Orchestration
#
# Launches multiple design batches with different configurations for maximum diversity.
# Run from the bioarena-in-silico-in-vitro/ directory.
#
# Usage:
#   bash scripts/run_all_batches.sh           # Full campaign (~$90, ~90 min)
#   bash scripts/run_all_batches.sh smoke     # Smoke test only (~$1, ~5 min)
#   bash scripts/run_all_batches.sh quick     # Quick run: Batch A only (~$37, ~30 min)

set -e
cd "$(dirname "$0")/.."

echo "============================================================"
echo "  TREM2 v3 DESIGN CAMPAIGN"
echo "============================================================"
echo ""

MODE="${1:-full}"

if [ "$MODE" = "smoke" ]; then
    echo "--- SMOKE TEST ---"
    echo "Running 1 design with reduced steps to verify pipeline..."
    modal run scripts/design_trem2_v3.py --smoke-test
    echo ""
    echo "Smoke test complete. If OK, run: bash scripts/run_all_batches.sh"
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# BATCH A: Primary helical binders (highest priority)
# 15 designs at [70, 80, 85] AA with strong HelixLoss
# Estimated: ~$37, ~30 min
# ══════════════════════════════════════════════════════════════
echo "--- BATCH A: Primary Helical Binders ---"
echo "  Lengths: [70, 80, 85], HelixLoss=2.5, 5 seeds each"
echo "  Estimated: ~\$37, ~30 min"
echo ""
modal run scripts/design_trem2_v3.py \
    --lengths 70 80 85 \
    --n-per-length 5 \
    --helix-weight 2.5 \
    --seed-offset 0

echo ""
echo "Batch A complete."
echo ""

if [ "$MODE" = "quick" ]; then
    echo "Quick mode: skipping remaining batches."
    echo "Run selection: python scripts/select_final.py --batch results/designs/batch_v3_*.json --output results/submission --skip-validation"
    exit 0
fi

# ══════════════════════════════════════════════════════════════
# BATCH B: Diversity variant (different helix weight + seeds)
# 10 designs at [70, 80] AA with moderate HelixLoss
# Estimated: ~$25, ~30 min
# ══════════════════════════════════════════════════════════════
echo "--- BATCH B: Diversity Variant ---"
echo "  Lengths: [70, 80], HelixLoss=2.0, 5 seeds each"
echo "  Estimated: ~\$25, ~30 min"
echo ""
modal run scripts/design_trem2_v3.py \
    --lengths 70 80 \
    --n-per-length 5 \
    --helix-weight 2.0 \
    --seed-offset 500

echo ""
echo "Batch B complete."
echo ""

# ══════════════════════════════════════════════════════════════
# BATCH C: Longer exploratory (for back-face / flat surface binding)
# 5 designs at 100 AA with light HelixLoss
# Estimated: ~$15, ~35 min
# ══════════════════════════════════════════════════════════════
echo "--- BATCH C: Longer Exploratory ---"
echo "  Lengths: [100], HelixLoss=1.5, 5 seeds"
echo "  Estimated: ~\$15, ~35 min"
echo ""
modal run scripts/design_trem2_v3.py \
    --lengths 100 \
    --n-per-length 5 \
    --helix-weight 1.5 \
    --seed-offset 1000

echo ""
echo "Batch C complete."
echo ""

# ══════════════════════════════════════════════════════════════
# BATCH D: BoltzGen structural diversity
# 24 backbone samples via diffusion + MPNN inverse folding
# Estimated: ~$10, ~20 min
# ══════════════════════════════════════════════════════════════
echo "--- BATCH D: BoltzGen Structural Diversity ---"
echo "  24 samples at L=80 via BoltzGen diffusion"
echo "  Estimated: ~\$10, ~20 min"
echo ""
modal run scripts/boltzgen_trem2_modal.py \
    --binder-length 80 \
    --n-samples 24

echo ""
echo "Batch D complete."
echo ""

# ══════════════════════════════════════════════════════════════
# POST-DESIGN: Summary
# ══════════════════════════════════════════════════════════════
echo "============================================================"
echo "  ALL BATCHES COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. (Optional) Validate top designs:"
echo "     modal run scripts/validate_monomers.py \\"
echo "       --batch results/designs/batch_v3_*.json --top 25"
echo ""
echo "  2. Select final 10:"
echo "     python scripts/select_final.py \\"
echo "       --batch results/designs/batch_v3_*.json results/designs/batch_boltzgen_*.json \\"
echo "       --output results/submission --skip-validation"
echo ""
echo "  3. Submit to bioArena:"
echo "     cat results/submission/submission.fasta"
echo ""
