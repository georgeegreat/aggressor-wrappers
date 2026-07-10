#!/usr/bin/env bash
# Create or update the AGGRESSOR conda environment for aggressor_wrappers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="AGGRESSOR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH; install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Updating existing conda env: $ENV_NAME"
  conda env update -n "$ENV_NAME" -f "$ROOT/environment.yml" --prune
else
  echo "Creating conda env: $ENV_NAME"
  conda env create -f "$ROOT/environment.yml"
fi

conda activate "$ENV_NAME"
pip install -e "$ROOT/.[test]"

echo ""
echo "AGGRESSOR env is ready. Next steps:"
echo "  conda activate $ENV_NAME"
echo "  pip install pyrosetta --find-links https://west.rosettacommons.org/pyrosetta/quarterly/release"
echo "  export KEY_MODELLER=\"YOUR_LICENSE_KEY\"   # see README"
echo "  Rscript -e 'install.packages(\"appnn\", repos=\"https://cloud.r-project.org\")'   # if needed"
