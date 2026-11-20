#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

echo "==> Setting up environment in ${VENV_DIR}"

if command -v uv >/dev/null 2>&1; then
    echo "==> Using uv"
    uv venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    uv pip install -r requirements.txt
else
    echo "==> uv not found; falling back to python3 -m venv + pip"
    python3 -m venv "${VENV_DIR}"
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
fi

echo ""
echo "==> Installation complete!"
echo "Activate the environment with:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "Note: openlanguagedata/flores_plus is a gated dataset on Hugging Face."
echo "Before running the evaluation:"
echo "  1. Request access at https://huggingface.co/datasets/openlanguagedata/flores_plus (auto-approved)."
echo "  2. Log in via: huggingface-cli login  (or export HF_TOKEN=hf_...)"

