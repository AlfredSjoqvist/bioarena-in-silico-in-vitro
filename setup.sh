#!/bin/bash
# bioArena Hackathon Setup Script
# Run this BEFORE the hackathon to have everything ready

set -e
echo "=== bioArena Hackathon: TREM2 Binder Design Setup ==="
echo ""

# 1. Clone scoring tool
echo "--- Step 1: Cloning IPSAE scoring tool ---"
if [ ! -d "tools/IPSAE" ]; then
    mkdir -p tools
    git clone https://github.com/DunbrackLab/IPSAE.git tools/IPSAE
    echo "  Done: tools/IPSAE"
else
    echo "  Already exists: tools/IPSAE"
fi

# 2. Clone Mosaic (Boltz-2 binder design framework)
echo ""
echo "--- Step 2: Cloning Mosaic (Boltz-2 design framework) ---"
if [ ! -d "mosaic" ]; then
    git clone https://github.com/escalante-bio/mosaic.git
    echo "  Done: mosaic/"
else
    echo "  Already exists: mosaic/"
fi

# 3. Install base Python dependencies
echo ""
echo "--- Step 3: Installing Python dependencies ---"
pip install -r requirements.txt
echo "  Done"

# 4. Setup Modal
echo ""
echo "--- Step 4: Setting up Modal (GPU compute) ---"
echo "  If not already authenticated, run: modal setup"
modal token list 2>/dev/null && echo "  Modal already authenticated" || echo "  Run 'modal setup' to authenticate"

# 5. Download TREM2 structure
echo ""
echo "--- Step 5: Downloading TREM2 PDB structure ---"
python scripts/download_trem2.py

# 6. Setup Mosaic (requires GPU or will be done on Modal)
echo ""
echo "--- Step 6: Mosaic setup ---"
echo "  Mosaic requires GPU for full setup."
echo "  On a GPU machine, run:"
echo "    cd mosaic && uv sync --group jax-cuda"
echo "  For CPU-only testing:"
echo "    cd mosaic && uv sync --group jax-cpu"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Pre-hackathon checklist:"
echo "  [ ] ROWAN_API_KEY set (get from labs.rowansci.com)"
echo "  [ ] OPENROUTER_API_KEY set (get from openrouter.ai/settings/keys)"
echo "  [ ] Modal authenticated (modal setup)"
echo "  [ ] TREM2 PDB downloaded (check data/trem2/)"
echo "  [ ] Mosaic cloned (check mosaic/)"
echo "  [ ] IPSAE cloned (check tools/IPSAE/)"
