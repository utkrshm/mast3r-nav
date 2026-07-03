"""Script to compare the costmaps generated from the original costmaps implementation, with the costmaps from the vectorized implementation"""

# Each costmap (original and vectorized) .npz file is of the shape (n_images, H, W), where each value encodes the distance to goal of each pixel.

import numpy as np
from pathlib import Path
import pathlib
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import glob

DIR = Path("../data/droid-w_modified/trajectories")

org_costmaps = glob.glob("**/costmaps_*.npz", root_dir=DIR, recursive=True)
vec_costmaps = glob.glob("**/vectorized_costmaps_*.npz", root_dir=DIR, recursive=True)
new_vec_costmaps = glob.glob("**/single_batch_costmaps_*.npz", root_dir=DIR, recursive=True)

# org_costmaps = vec_costmaps
vec_costmaps = new_vec_costmaps

print(org_costmaps[0], vec_costmaps[0])

def pair_costmaps(org_list, vec_list):
    org_map = {Path(f).parent.name: DIR / f for f in org_list}
    vec_map = {Path(f).parent.name: DIR / f for f in vec_list}
    
    pairs = []
    for parent, org_path in org_map.items():
        if parent in vec_map:
            pairs.append((org_path, vec_map[parent]))
    return pairs

pairs = pair_costmaps(org_costmaps, vec_costmaps)
print(f"Found {len(pairs)} matched pairs; original list has {len(org_costmaps)} costmaps, the vectorized one had {len(vec_costmaps)}")

print("Comparing across original and vectorized costmaps")

for org_path, vec_path in pairs:
    org_data = np.load(org_path)
    vec_data = np.load(vec_path)
    
    org_cm = org_data["costmaps"]
    vec_cm = vec_data["costmaps"]
    
    diff = np.abs(org_cm - vec_cm)
    norm_diff = np.divide(diff, org_cm, where=org_cm!=0)
    total_pixels = diff.size
    non_matching = np.sum(diff != 0)

    max_diff_idx = np.unravel_index(np.argmax(diff, axis=None), diff.shape)
    if non_matching > 0:
        print(f"Costmaps mismatch details:")
        print(f"  Non-matching pixels: {non_matching} / {total_pixels} ({non_matching/total_pixels*100:.4f}%)")
        print(f"  Mean difference: {np.mean(diff):.6f}")
        print(f"  Max difference:  {np.max(diff):.6f}")

        print(f"Normalized mean diff (non-zero): {np.mean(norm_diff):.6f}")
        print(f"Normalized max diff (non-zero):  {np.max(norm_diff):.6f}")

        print(f"Maximum difference occurs at index: {max_diff_idx}, where original costmap has value {org_cm[max_diff_idx]}, vectorized costmap has value {vec_cm[max_diff_idx]}; image index {max_diff_idx[0]}")

        print(f"Total images in this run: {org_cm.shape[0]}")

    else:
        print("✓ Costmaps are bit-identical!")

    # print("\n--- Metadata Comparison ---")
    # print("Original Metadata:")
    # print(org_data["metadata"])
    # print("\nVectorized Metadata:")
    # print(vec_data["metadata"])
    # print("---------------------------\n")
