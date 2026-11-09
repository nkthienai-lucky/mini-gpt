#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$HOME/ai-roadmap/.venv/bin/python" ]]; then
    PYTHON="$HOME/ai-roadmap/.venv/bin/python"
  else
    PYTHON="python"
  fi
fi
CHECKPOINT="checkpoints/smoke.pt"

"$PYTHON" -m pytest -q
"$PYTHON" train.py --device cpu --max-steps 20 --eval-interval 10 --checkpoint "$CHECKPOINT"
"$PYTHON" sample.py \
  --checkpoint "$CHECKPOINT" \
  --prompt "ROMEO:" \
  --max-new-tokens 32 \
  --temperature 0 \
  --benchmark \
  --device cpu
