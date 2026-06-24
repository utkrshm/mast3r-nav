#!/bin/bash

# Force-preload the conda-managed libpng so it wins over pip-bundled libpng
# in pillow.libs/ which is too old and lacks png_set_cICP (needed by opencv 4.12+).
# LD_PRELOAD overrides RPATH embedded in pip opencv binaries.

# Determine project root relative to this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

ENV_DIR=""
if [ -n "$CONDA_PREFIX" ] && [ -d "$CONDA_PREFIX/lib" ]; then
    ENV_DIR="$CONDA_PREFIX"
elif [ -d "$PROJECT_ROOT/.pixi/envs/default/lib" ]; then
    ENV_DIR="$PROJECT_ROOT/.pixi/envs/default"
fi

if [ -n "$ENV_DIR" ] && [ -f "$ENV_DIR/lib/libpng16.so.16" ]; then
    export LD_PRELOAD="$ENV_DIR/lib/libpng16.so.16:$LD_PRELOAD"
fi

