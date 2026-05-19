#!/bin/bash
# RS Mapping Pilot Experiment
# Run when GPU server is accessible (VPN connected)
#
# Usage on nolf-02 or nolf-03:
#   cd /path/to/mas_GRDH_copy/scripts
#   bash run_rs_pilot.sh
#
# Or direct:
#   python test_rs_comparison.py \
#       --ckpt ../models/v1-5-pruned-emaonly.ckpt \
#       --config ../configs/stable-diffusion/ldm.yaml \
#       --test_prompts test_prompts.txt \
#       --n_prompts 3 \
#       --outdir ./rs_comparison

CKPT="../models/v1-5-pruned-emaonly.ckpt"
CONFIG="../configs/stable-diffusion/ldm.yaml"
PROMPTS="test_prompts.txt"
OUTDIR="./rs_comparison"

# On remote server with conda
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate CGIS
fi

mkdir -p "$OUTDIR"

python test_rs_comparison.py \
    --ckpt "$CKPT" \
    --config "$CONFIG" \
    --test_prompts "$PROMPTS" \
    --n_prompts 3 \
    --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/experiment.log"

echo "Done. Results in $OUTDIR/"
