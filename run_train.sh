#!/bin/bash

# Check for --clear-cache flag
for arg in "$@"; do
    if [ "$arg" == "--clear-cache" ]; then
        echo "Clearing dataset cache..."
        rm -rf /scratch2/utkarsh.malaiya/datasets/gs2_modified/data_splits/gs2/train/dataset_*.pkl
        rm -rf /scratch2/utkarsh.malaiya/datasets/gs2_modified/data_splits/gs2/train/dataset_*.lmdb
        rm -rf /scratch2/utkarsh.malaiya/datasets/gs2_modified/data_splits/gs2/val/dataset_*.pkl
        rm -rf /scratch2/utkarsh.malaiya/datasets/gs2_modified/data_splits/gs2/val/dataset_*.lmdb
        echo "Cache cleared!"
        break
    fi
done

source scripts/activate.sh

cd libs/control/visualnav_transformer/train

python train.py --config ./config/mast3r-nav.yaml