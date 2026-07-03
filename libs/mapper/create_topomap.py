import os
import time
import numpy as np
import cv2
import json
from tqdm import tqdm
from typing import Tuple
import networkx as nx
from itertools import combinations
# import open3d as o3d
from natsort import natsorted
import blosc2
import pickle
import sys
from pathlib import Path
import shutil
from enum import Enum
from PIL import Image
from dataclasses import dataclass
import h5py
import glob

from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
import torch
import torchvision.transforms as tfm

# Disable torch warnings
import warnings
warnings.simplefilter("ignore")

# Hydra imports
import hydra
from omegaconf import DictConfig, OmegaConf

# Add third-party libraries to path
BASE_DIR = Path(__file__).parent.parent.parent
MAST3R_PATH = BASE_DIR / "libs" / "matcher" / "mast3r"

# Add to Python path if not already there
if str(MAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(MAST3R_PATH))

# Import geometry utilities
from libs.common.geometry_utils import (
    farthest_point_sampling_o3d,
    resize_to_divisible,
    pixel_to_camera_3d,
    get_mask_centroid
)

from libs.common.graph_utils import load_compressed_graph_chunked

# Import our MASt3R wrapper
from libs.mast3r_utils import MASt3RInference

class NodeCullingMode(Enum):
    NONE = "NONE"
    FPS = "FPS"

class EdgeCullingMode(Enum):
    NONE = "NONE"
    EMST_SINGLE = "EMST_SINGLE"
    DELAUNAY_3D = "DELAUNAY_3D"

@dataclass
class CostmapData:
    costmaps: np.ndarray  # shape: (N_images, H, W)
    metadata: dict

    @staticmethod
    def from_npz(npz_path):
        """
        Docstring for from_npz
        
        :param npz_path: Description
        """
        data = np.load(npz_path, allow_pickle=True)
        costmaps = data['costmaps']
        metadata = json.loads(data['metadata'].item())
        return CostmapData(costmaps=costmaps, metadata=metadata)
    
    def get_costmap(self):
        return self.costmaps
    
    def get_metadata(self):
        return self.metadata

class MapTopological3DPoints:
    def __init__(self, img_dir: str, out_dir: str, cfg: DictConfig):
        self.cfg = cfg
        # self.cfg.update(cfg)
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        print("\n" + "="*80)
        print("INITIALIZING TOPOLOGICAL MAPPER")
        print("="*80)
        print(f"\nConfiguration:\n{OmegaConf.to_yaml(cfg)}")

        # Directory paths
        self.img_dir = Path(img_dir)
        self.scene_dir = self.img_dir.parent
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Image configuration
        self.W = cfg.image.width
        self.H = cfg.image.height
        self.device = cfg.model.device
        self.img_match_window_size = cfg.graph.inter_image_match_window_size

        # Load and sort image names
        self.img_names = natsorted(os.listdir(self.img_dir))
        self.img_paths = [self.img_dir / img_name for img_name in self.img_names]
        print(f"Found {len(self.img_paths)} images in {self.img_dir}")
        
        # Apply subsampling
        self.img_paths = self._subsample_images()
        print(f"After subsampling, {len(self.img_paths)} images will be used.")

        # optionally copy them (useful when original images are in a different dir)
        if cfg.processing.copy_images:
            self._copy_images_to_output_dir()

        # init other variables
        self.G, self.nodeID_to_imgRegionIdx = None, None
        self.inter_image_edges = {}
        self.intra_image_edges = {}
        self.pixel_to_node_id = {}  # Mapping for sparse graph
        self.force_recompute_graph = cfg.processing.force_recompute_graph

        # Output file paths
        self.pc_npz_path = self.out_dir / "nodes_mast3r_points.npz"
        self.graph_intra_path = self.out_dir / "graph_intra_edges.pickle"
        self.graph_inter_path = self.out_dir / "graph_just_inter_edges.pickle"
        self.graph_path = self.out_dir / "graph_mast3r_intra_edges_with_weights.pickle"

        # Load MASt3R model using wrapper
        self.model_path = cfg.model.path
        self.mast3r_match_subsample = self.cfg.model.subsample_or_initxy1
        self.batch_size = self.cfg.model.get("batch_size", 64)
        self.mast3r = MASt3RInference(model_path=self.model_path, device=self.device)
    
    def _get_base_graph_filename(self):
        """Generate filename for base graph (without goal)"""
        ec_mode = self.cfg.graph.edge_culling_mode
        nc_mode = self.cfg.graph.node_culling_mode
        nc_factor = self.cfg.graph.node_culling_factor
        w, h = self.cfg.image.width, self.cfg.image.height
        
        return (
            f"graph_base_"
            f"{w}x{h}_"
            f"EC_{ec_mode}_"
            f"NC_{nc_mode}_"
            f"NCF_{nc_factor}.pkl"
        )
    
    def _get_goal_graph_filename(self, goal_img_idx: int = None):
        """Generate filename for graph with goal distances"""
        ec_mode = self.cfg.graph.edge_culling_mode
        nc_mode = self.cfg.graph.node_culling_mode
        nc_factor = self.cfg.graph.node_culling_factor
        w, h = self.cfg.image.width, self.cfg.image.height
        
        # if goal_img_idx is None:
        #     goal_img_idx = self.G.graph.get('goal_img_idx', 'unknown')
        
        return (
            f"graph_with_distances_to_goal_"
            f"{w}x{h}_"
            f"EC_{ec_mode}_"
            f"NC_{nc_mode}_"
            f"NCF_{nc_factor}.pkl"
        )
    
    def _get_costmap_filename(self):
        """Generate descriptive filename for graph based on config"""
        ec_mode = self.cfg.graph.edge_culling_mode
        nc_mode = self.cfg.graph.node_culling_mode
        nc_factor = self.cfg.graph.node_culling_factor
        w, h = self.cfg.image.width, self.cfg.image.height
        
        return (
            f"single_batch_costmaps_"
            f"{w}x{h}_"
            f"EC_{ec_mode}_"
            f"NC_{nc_mode}_"
            f"NCF_{nc_factor}"
        )
    
    def _default_config(self) -> dict:
        """Return default configuration (private method)"""
        return {
            "W": 320,
            "H": 240,
            "image_match_window_size": 3,
            "model_path": None,
            "device": "cuda",
            "force_recompute_graph": False,
            "subsample_start_idx": 0,
            "subsample_end_idx": None,
            "subsample_step": 1,
            "copy_images": False,
            "edge_culling_mode": EdgeCullingMode.NONE.value,
            "node_culling_mode": NodeCullingMode.NONE.value,
            "node_culling_factor": 10,
        }

    def _subsample_images(self) -> list:
        """Subsample images based on configuration (private method)"""
        start_idx = self.cfg.processing.subsample_start_idx
        end_idx = self.cfg.processing.subsample_end_idx
        step = self.cfg.processing.subsample_step

        return self.img_paths[start_idx:end_idx:step]
    
    def _copy_images_to_output_dir(self):
        """Copy images to output directory (private method)"""
        out_img_dir = self.out_dir / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file extension from first image
        extension = Path(self.img_paths[0]).suffix
        
        print(f"Copying {len(self.img_paths)} images to {out_img_dir}")
        for i, img_path in enumerate(self.img_paths):
            output_path = out_img_dir / f"{i:04d}{extension}"
            shutil.copy2(img_path, output_path)
    
    def create_map_topo(self):
        """
        """
        # Step 1: Compute or load point clouds
        if (not self.pc_npz_path.exists() and not self.graph_path.exists()) or self.force_recompute_graph:
            pc_dict = self.compute_and_save_point_clouds(save_as_npz=True)
        else:
            print(f"Using Precomputed point clouds from {self.pc_npz_path}")
            pc_dict = np.load(self.pc_npz_path)
        
        # Step 2: Create base graph with nodes
        self.G = self.create_base_graph_with_nodes(pc_dict)
        print(f"\nGraph just after creation: {self.G}")

        # Step 3 : Add inter-image edges
        if not self.graph_intra_path.exists() or self.force_recompute_graph:
            self.G = self.add_inter_image_edges_to_graph()
        
        # # Step 4 : Add intra-image edges
        if not self.graph_inter_path.exists() or self.force_recompute_graph:
            self.G = self.add_intra_image_edges_to_graph()
    
    def compute_and_save_point_clouds(self, save_as_npz: bool = True) -> dict:
        """
        Compute 3D point clouds for all images using MASt3R model.
        
        Args:
            save_to_disk: Whether to save point clouds to compressed NPZ file
            
        Returns:
            dict: Mapping from image path (str) to point cloud (np.ndarray of shape H x W x 3)
        """
        # point clouds dictionary
        pc_dict = {}
        
        # Batched self-matching: all images in one GPU call using configured batch size
        print(f"Running batched point cloud inference on {len(self.img_paths)} images (batch_size={self.batch_size})...")
        all_pts3d = self.mast3r.get_pts3d(
            self.img_paths, resize=(self.H, self.W), batch_size=self.batch_size
        )  # (N, H, W, 3)
        for img_idx, img_path in enumerate(
            tqdm(self.img_paths, desc="Storing MASt3R point clouds")
        ):
            pc_dict[str(img_path)] = all_pts3d[img_idx]  # (H, W, 3)
        
        # Optionally save to disk
        if save_as_npz:
            np.savez_compressed(self.pc_npz_path, **pc_dict)
        
        return pc_dict
    
    def pixel_coord_to_global_node_id(self, img_idx, px, py):
        # img_idx should be according to 0-indexing
        H, W = self.H, self.W

        # node id of the first node belonging this image
        img_st_node_id = img_idx * (H * W)
        node_id = img_st_node_id + (py * W + px)
        return node_id

    def create_base_graph_with_nodes(self, pc_dict):
        """
        Creates a Sparse graph based on mast3r matches
        """
        G = nx.Graph()
        G.graph['cfg'] = OmegaConf.to_container(self.cfg, resolve=True)
        
        # Collect all match pairs while preserving correspondence
        all_match_pairs = []
        num_matches_per_pair = []  # Track number of matches per image pair for statistics
        nc_factor = self.cfg.graph.node_culling_factor
        
        print("First pass: collecting all match pairs...")
        img_st_idx = 0
        img_end_idx = len(self.img_paths)
        match_window_size = self.img_match_window_size
        
        from mast3r.fast_nn import fast_reciprocal_NNs

        # Pre-collect all (i, j) pairs for a single batched inference call
        _pair_indices = []
        _imgs0, _imgs1 = [], []
        for i in range(img_st_idx, img_end_idx):
            for j in range(i + 1, min(i + 1 + match_window_size, img_end_idx)):
                _pair_indices.append((i, j))
                _imgs0.append(self.img_paths[i])
                _imgs1.append(self.img_paths[j])

        print(f"Running batched match inference on {len(_pair_indices)} pairs (batch_size={self.batch_size})...")
        try:
            _match_out = self.mast3r.infer(
                _imgs0, _imgs1, resize=(self.H, self.W), batch_size=self.batch_size
            )
        except Exception as e:
            print(f"Batched match inference failed: {e}")
            raise

        _border = 3
        _subsample = self.cfg.model.subsample_or_initxy1

        for _pk, (i, j) in enumerate(
            tqdm(_pair_indices, desc="Extracting matches from batch")
        ):
            try:
                desc1 = _match_out["pred1"]["desc"][_pk].detach()
                desc2 = _match_out["pred2"]["desc"][_pk].detach()

                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1, desc2,
                    subsample_or_initxy1=_subsample,
                    device=self.device,
                    dist="dot",
                    block_size=2**13,
                )

                # Border filtering
                H0, W0 = _match_out["view1"]["true_shape"][_pk]
                H1, W1 = _match_out["view2"]["true_shape"][_pk]
                _valid = (
                    (matches_im0[:, 0] >= _border) & (matches_im0[:, 0] < int(W0) - _border) &
                    (matches_im0[:, 1] >= _border) & (matches_im0[:, 1] < int(H0) - _border) &
                    (matches_im1[:, 0] >= _border) & (matches_im1[:, 0] < int(W1) - _border) &
                    (matches_im1[:, 1] >= _border) & (matches_im1[:, 1] < int(H1) - _border)
                )
                matches_im0 = matches_im0[_valid]
                matches_im1 = matches_im1[_valid]

                num_matches = len(matches_im0)
                num_matches_per_pair.append(num_matches)

                # Vectorized pixel tuple construction (replaces per-match loop)
                _px0 = matches_im0[:, 0].astype(int)
                _py0 = matches_im0[:, 1].astype(int)
                _px1 = matches_im1[:, 0].astype(int)
                _py1 = matches_im1[:, 1].astype(int)
                matches_per_pair = [
                    ((i, _px0[k], _py0[k]), (j, _px1[k], _py1[k]))
                    for k in range(num_matches)
                ]

                # Sample match pairs (preserves correspondence)
                matches_per_pair = matches_per_pair[::nc_factor]
                all_match_pairs.extend(matches_per_pair)

            except Exception as e:
                print(f"Error extracting matches between {i} and {j}: {e}")
                continue
        
        print(f"Found {len(all_match_pairs)} match pairs across all images")
        if len(num_matches_per_pair) > 0:
            avg_matches = np.mean(num_matches_per_pair)
            min_matches = np.min(num_matches_per_pair)
            max_matches = np.max(num_matches_per_pair)
            print(f"Matches per image pair - Avg: {avg_matches:.1f}, Min: {min_matches}, Max: {max_matches}")
        
        sampled_match_pairs = all_match_pairs
        # sampled_match_pairs = all_match_pairs[::nc_factor]
        
        print(f"Sampled {len(sampled_match_pairs)} match pairs (every {nc_factor}th pair)")
        
        # Extract unique pixels from sampled match pairs
        unique_pixels = set()
        for pair in sampled_match_pairs:
            unique_pixels.add(pair[0]) 
            unique_pixels.add(pair[1])
        
        print(f"Extracted {len(unique_pixels)} unique pixels from sampled pairs")
        
        # Create nodes
        pixel_to_node_id = {}
        for img_idx, px, py in tqdm(unique_pixels, desc="Creating DA nodes"):
            # Node ID calculation
            node_id = self.pixel_coord_to_global_node_id(img_idx, px, py)
            
            key = str(self.img_paths[img_idx])
            pcd = pc_dict[key]  # (H, W, 3)
            
            # Get 3D coordinate for this pixel
            coord_3d = pcd[py, px]  # Note: pcd is indexed as [y, x]
            
            # Create node
            node_attrs = {
                "map": [img_idx, py * self.W + px],  # [image_idx, pixel_index]
                "coord_mast3r": coord_3d,
                "pixel": [px, py],
                "type": "da"
            }
            
            G.add_node(node_id, **node_attrs)
            pixel_to_node_id[(img_idx, px, py)] = node_id
            # node_id_to_pixel[node_id] = (img_idx, px, py)
        
        # Store both mappings
        self.pixel_to_node_id = pixel_to_node_id
        self.sampled_match_pairs = sampled_match_pairs  # Store for guaranteed edge creation
        # self.node_id_to_pixel = node_id_to_pixel
        
        print(f"Created sparse graph with {G.number_of_nodes()} DA nodes using old node IDs")
        print(f"Stored {len(self.sampled_match_pairs)} match pairs for edge creation")
        
        # Compute nodes per image statistics
        nodes_per_image = {}
        for img_idx, px, py in unique_pixels:
            nodes_per_image[img_idx] = nodes_per_image.get(img_idx, 0) + 1
        
        if len(nodes_per_image) > 0:
            counts = list(nodes_per_image.values())
            avg_nodes = np.mean(counts)
            min_nodes = np.min(counts)
            max_nodes = np.max(counts)
            print(f"DA nodes per image - Avg: {avg_nodes:.1f}, Min: {min_nodes}, Max: {max_nodes}")
        
        return G

    def add_inter_image_edges_to_graph(self):
        da_edges = []

        print(f"Adding Inter-Image edges from {len(self.sampled_match_pairs)} stored match pairs...")
        
        for pair in tqdm(self.sampled_match_pairs, desc="Creating DA edges from match pairs"):
            # pixel_i = (img_idx_i, px_i, py_i)
            pixel_i, pixel_j = pair
            
            # Get node IDs for both pixels in the pair
            node_i = self.pixel_coord_to_global_node_id(pixel_i[0], pixel_i[1], pixel_i[2])
            node_j = self.pixel_coord_to_global_node_id(pixel_j[0], pixel_j[1], pixel_j[2])
            
            # Both nodes should exist since we created them from sampled pairs
            if node_i in self.G.nodes and node_j in self.G.nodes:
                da_edges.append((int(node_i), int(node_j), {'edge_type': 'da', 'weight': 0}))
            else:
                print(f"Warning: Missing nodes {node_i} or {node_j} for pair {pair}")
        
        print(f"Created {len(da_edges)} DA edges from stored pairs")
        print(f"Expected {len(self.sampled_match_pairs)} edges, got {len(da_edges)} edges")

        # Add DA edges to graph
        self.G.add_edges_from(da_edges)
        print(f"\n\nNumber of nodes and edges: {len(self.G.nodes())}, {self.G.number_of_edges()}")

        return self.G

    def add_intra_image_edges_to_graph(self):
        """
        Connect DA nodes within the same image using spatial relationships
        """
        # Group DA nodes by image
        da_nodes_per_img = {}
        for node_id in self.G.nodes():
            node = self.G.nodes[node_id]
            img_id = node['map'][0]
            
            if img_id not in da_nodes_per_img:
                da_nodes_per_img[img_id] = []
            da_nodes_per_img[img_id].append(node_id)
        print(f"Adding intra-image edges for {len(da_nodes_per_img)} images")

        # adding DA intra-image edges
        for img_id in tqdm(
            da_nodes_per_img.keys(), desc="connecting da nodes intra edges"
        ):
            da_nodes = da_nodes_per_img[img_id]
            edge_attrs = {"edge_type": "da_intra"}  # You can customize these attributes
            edge_culling_mode = EdgeCullingMode(self.cfg.graph.edge_culling_mode)
            # --- EMST_SINGLE mode ---
            if edge_culling_mode == EdgeCullingMode.EMST_SINGLE:
                edges = self._create_emst_edges(da_nodes, img_id)
            # --- DELAUNAY_3D mode ---
            elif edge_culling_mode == EdgeCullingMode.DELAUNAY_3D:
                edges = self._create_delaunay_3d_edges(da_nodes, img_id)
            # --- Default: all-to-all ---
            else:
                edges = self._create_complete_graph_edges(da_nodes, img_id)
        
            # Add all intra-image edges to the graph
            self.G.add_edges_from(edges)

        print(
            f"Final graph has {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges"
        )

        return self.G
    
    def _create_emst_edges(self, da_nodes: list, img_id: int) -> list:
        """Create edges using Euclidean Minimum Spanning Tree"""
        if len(da_nodes) <= 1:
            return []
        
        # Get 3D coordinates
        coords = np.array([
            self.G.nodes[nid]["coord_mast3r"] 
            for nid in da_nodes
        ])
        
        # Compute distance matrix
        dist_matrix = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], 
            axis=2
        )

        # === DEBUG: Essential checks ===
        # print(
        #     f"EMST DEBUG img {img_id}: nodes={len(da_nodes)}, dist_matrix min/max={dist_matrix.min():.3f}/{dist_matrix.max():.3f}, has_nan/inf={np.isnan(dist_matrix).any()}/{np.isinf(dist_matrix).any()}"
        # )
        
        # Compute MST
        mst = minimum_spanning_tree(dist_matrix).toarray()
        
        # Build edge list with attributes
        edges = []
        for i in range(len(da_nodes)):
            for j in range(len(da_nodes)): 
                if i != j and mst[i, j] > 0:
                    edge_weight = dist_matrix[i, j]
                    edges.append((
                        da_nodes[i], 
                        da_nodes[j], 
                        {
                            'edge_type': 'da_intra',
                            'weight': edge_weight,
                        }
                    ))
        
        # # Check connectivity using NetworkX
        # mst_graph = nx.Graph()
        # mst_graph.add_nodes_from(range(len(da_nodes)))
        # mst_graph.add_edges_from(
        #     [
        #         (i, j)
        #         for i in range(len(da_nodes))
        #         for j in range(len(da_nodes))
        #         if i != j and mst[i, j] > 0
        #     ]
        # )
        # is_connected = nx.is_connected(mst_graph)

        # print(
        #     f"EMST edges: {len(da_da_intra_edges)}, expected: {len(da_nodes) - 1}, connected: {is_connected}"
        # )
        
        # print(f"Created {len(edges)} EMST edges for image {img_id} "
        #     f"(expected {len(da_nodes) - 1})")
        
        return edges
    
    def _create_delaunay_3d_edges(self, da_nodes: list, img_id: int) -> list:
        """Create edges using 3D Delaunay triangulation"""
        if len(da_nodes) <= 3:
            print(f"Not enough DA nodes for 3D Delaunay in image {img_id}")
            return []
        
        # Get 3D coordinates
        coords = np.array([
            self.G.nodes[nid]["coord_mast3r"] 
            for nid in da_nodes
        ])

        # Compute distance matrix
        dist_matrix = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], 
            axis=2
        )

        try:
            tri = Delaunay(coords)
            
            # Extract unique edges from simplices
            edges_set = set()
            for simplex in tri.simplices:
                # Each simplex is a tetrahedron (4 vertices)
                for i in range(4):
                    for j in range(i + 1, 4):
                        a, b = simplex[i], simplex[j]
                        node_a = da_nodes[a]
                        node_b = da_nodes[b]
                        # Sort to ensure uniqueness
                        edge = tuple(sorted((node_a, node_b)))
                        edge_with_attrs = edge + ({'edge_type': 'da_intra', 'weight': dist_matrix[a, b]},)
                        edges_set.add(edge_with_attrs)
            
            # Convert to list with attributes
            edges = list(edges_set)

            # print(f"Created {len(edges)} 3D Delaunay edges for image {img_id}")
            return edges
            
        except Exception as e:
            print(f"3D Delaunay failed for image {img_id}: {e}")
            return []

    def _create_complete_graph_edges(self, da_nodes: list, img_id: int) -> list:
        """Create complete graph (all-to-all connections) with distance-based weights"""
        if len(da_nodes) <= 1:
            print(f"Only one DA node in image {img_id}, no intra-image edges needed")
            return []
        
        # Get 3D coordinates
        coords = np.array([
            self.G.nodes[nid]["coord_mast3r"] 
            for nid in da_nodes
        ])
        
        # Compute pairwise distance matrix
        dist_matrix = np.linalg.norm(
            coords[:, None, :] - coords[None, :, :], 
            axis=2
        )
        
        # Create all-to-all edges with weights
        edges = [
            (da_nodes[i], da_nodes[j], {'edge_type': 'da_intra', 'weight': dist_matrix[i, j]})
            for i, j in combinations(range(len(da_nodes)), 2)
        ]
        
        # print(f"Created {len(edges)} complete graph edges for image {img_id}")
        return edges

    def get_goal_from_episode(self):
        """
        Infer goal from episode folder using semantic mask and centroid.
        
        Uses get_goal_info() to find the goal image and mask, then computes
        the centroid to get goal pixel coordinates.
        """
        from libs.common.geometry_utils import get_goal_info, get_mask_centroid
        
        task_type = self.cfg.goal.get("task_type", "original")
        goal_img_idx, goal_mask, goal_instance_id = get_goal_info(
            str(self.scene_dir), 
            task_type
        )
        
        # Resize mask if needed
        if goal_mask.shape != (self.H, self.W):
            print(f"Resizing goal mask from {goal_mask.shape} to ({self.H}, {self.W})")
            goal_mask = cv2.resize(goal_mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        
        # Get centroid
        centroid = get_mask_centroid(goal_mask)
        if centroid is None:
            raise ValueError(f"Goal mask is empty in episode {self.scene_dir}")
        
        goal_px, goal_py = centroid
        
        print(f"Inferred goal from episode: img_idx={goal_img_idx}, pixel=({goal_px}, {goal_py})")
        
        # Store as instance variables for compute_distances_to_goal_node
        self.inferred_goal_img_idx = goal_img_idx
        self.inferred_goal_px = goal_px
        self.inferred_goal_py = goal_py
        self.inferred_goal_mask = goal_mask
        
        return goal_img_idx, goal_px, goal_py

    def compute_distances_to_goal_node(
        self,
        goal_img_idx: int = None,
        goal_px: int = None,
        goal_py: int = None
    ):
        """
        Compute distances from all nodes to a goal node.
        
        Args:
            goal_img_idx: Goal image index (optional, uses config/inferred if None)
            goal_px: Goal pixel x coordinate (optional)
            goal_py: Goal pixel y coordinate (optional)
            
        Returns:
            Costmaps array if compute_costmaps is enabled, else None
        """
        # Determine goal coordinates: explicit args > inferred > config
        if not goal_img_idx:
            if hasattr(self, 'inferred_goal_img_idx'):
                goal_img_idx = self.inferred_goal_img_idx
                goal_px = self.inferred_goal_px
                goal_py = self.inferred_goal_py
            else:
                goal_img_idx = self.cfg.goal.image_idx
                goal_px = self.cfg.goal.pixel_x
                goal_py = self.cfg.goal.pixel_y

            # If multiple scenes are to be processed, and the costmaps need to be compiled (implying that this is map compilation for the training stage), then set the goal image index to the last image frame
            if (self.cfg.scenes.multi_scene and self.cfg.scenes.compile_costmaps):
                goal_img_idx = len(self.img_paths) - 1

        print(f"Goal: img_idx={goal_img_idx}, pixel=({goal_px}, {goal_py})")

        # Load point cloud data if needed
        if os.path.exists(self.pc_npz_path):
            pc_dict = np.load(self.pc_npz_path)
        else:
            pc_dict = self.compute_and_save_point_clouds(save_as_npz=False)
        
        # Calculate expected goal node ID
        expected_goal_node_id = self.pixel_coord_to_global_node_id(goal_img_idx, goal_px, goal_py)
        
        # Add or find goal node
        if expected_goal_node_id in self.G.nodes:
            print(f"Goal node {expected_goal_node_id} already exists")
            goal_node_id = expected_goal_node_id
        else:
            # Add the goal node to the graph
            goal_node_id = self.add_goal_node(goal_img_idx, goal_px, goal_py, pc_dict)
            print(f"Added goal node {goal_node_id} at pixel ({goal_px}, {goal_py}) in image {goal_img_idx}")
        
        # Store goal info in graph (snake_case)
        self.G.graph['goal_img_idx'] = goal_img_idx
        self.G.graph['goal_node_coords'] = (goal_px, goal_py)
        self.G.graph['goal_node_id'] = goal_node_id
        
        # Connect DA nodes to goal
        self.connect_da_to_goal_node(goal_img_idx, goal_node_id)
        
        # Run Dijkstra from goal to all nodes
        path_lengths = self.get_single_source_paths(self.G, source_node=goal_node_id, weight='weight')
        self.all_path_lengths = path_lengths
        self.G.graph['all_path_lengths'] = {'weight': path_lengths}

        # Compute costmaps if enabled
        if self.cfg.goal.get('compute_costmaps', True):
            print(f"\nComputing distance-to-goal costmaps for all images...")
            img_costmaps = self.compute_all_image_costmaps(pc_dict)
            metadata = {
                'goal_img_idx': goal_img_idx,
                'goal_pixel': (goal_px, goal_py),
                'goal_node_id': goal_node_id,
                'cfg': OmegaConf.to_container(self.cfg, resolve=True),
                'goal_coord_3d': self.G.nodes[goal_node_id]['coord_mast3r'].tolist(),
                'image_paths': [str(path) for path in self.img_paths],
                'shape': list(img_costmaps.shape),
            }
            self.save_costmaps(img_costmaps, metadata)
            return img_costmaps

        return None

    def add_goal_node(self, goal_img_idx, goal_px, goal_py, pc_dict):
        """
        Add goal node to the sparse graph using OLD node ID calculation
        """
        # In case the goal img idx is higher than the total number of frames in the scene, then set the goal as the last frame
        try:
            key = str(self.img_paths[goal_img_idx])
        except IndexError:
            key = str(self.img_paths[len(self.img_paths) - 1])
        pcd = pc_dict[key]
        coord_3d = pcd[goal_py, goal_px]
        
        # Use OLD node ID calculation instead of sequential
        goal_node_id = self.pixel_coord_to_global_node_id(goal_img_idx, goal_px, goal_py)
        
        goal_attrs = {
            "map": [goal_img_idx, goal_py * self.W + goal_px],
            "coord_mast3r": coord_3d,
            "pixel": [goal_px, goal_py],
            "type": "goal"
        }
        
        self.G.add_node(goal_node_id, **goal_attrs)
        
        # Update mapping
        self.pixel_to_node_id[(goal_img_idx, goal_px, goal_py)] = goal_node_id
        
        # Store goal info in graph
        self.G.graph['goal_img_idx'] = goal_img_idx
        self.G.graph['goal_node_coords'] = (goal_px, goal_py)
        self.G.graph['goal_node_id'] = goal_node_id
        
        return goal_node_id

    def connect_da_to_goal_node(self, img_idx, goal_node_id):
        """
        Connect goal node to DA nodes in the same image (sparse graph version)
        """
        goal_node = self.G.nodes[goal_node_id]
        
        # Find all DA nodes in the same image
        img_da_node_ids = []
        for node_id, node_data in self.G.nodes(data=True):
            if (node_data['map'][0] == img_idx and 
                node_data.get('type') == 'da'):
                img_da_node_ids.append(node_id)
        
        print(f"Found {len(img_da_node_ids)} DA nodes in image {img_idx}")
        
        # if goal node is da node then it's already connected to all other da nodes of the image
        if goal_node["type"] == "da":
            print(f"{goal_node_id = } is a DA node")
            self.G.nodes[goal_node_id]["type"] = "goal"

            # Update the edgeType for all edges between goal_node_id and DA nodes in the image
            for da_node_id in img_da_node_ids:
                if self.G.has_edge(goal_node_id, da_node_id):
                    self.G.edges[goal_node_id, da_node_id]["edge_type"] = "goal_da_intra"
            return

        # updating the goal node type
        self.G.nodes[goal_node_id]["type"] = "goal"
        pts3d_goal_node = goal_node["coord_mast3r"]

        edge_weights = {}

        # Finding distance from goal_node to all DA nodes in the goal image
        for da_node_id in img_da_node_ids:
            da_node = self.G.nodes[da_node_id]
            pts3d_da_node = da_node["coord_mast3r"]
            edge_weights[da_node_id] = np.linalg.norm(pts3d_goal_node - pts3d_da_node)

        # Adding all the edge along with edge_weights from goal to all the da nodes
        self.G.add_edges_from(
            [
                (
                    goal_node_id,
                    da_node_id,
                    {
                        "edge_type": "goal_da_intra",
                        "weight": edge_weights[da_node_id],
                    },
                )
                for da_node_id in img_da_node_ids
            ]
        )
        
        print(f"Connected goal node {goal_node_id} to {len(img_da_node_ids)} DA nodes")
        return self.G

    def get_single_source_paths(self, G, source_node, weight=None, maxVal=1e6):
        """
        Compute shortest path lengths from a single source node to all other nodes.

        Args:
            G: NetworkX graph
            source_node: Source node ID
            weight: Edge weight attribute to use (e.g., 'margin')
            maxVal: Value to use for unreachable nodes

        Returns:
            dict: Dictionary mapping target nodes to their shortest path lengths from source_node
        """
        # Use NetworkX's single_source_dijkstra_path_length
        path_lengths = nx.single_source_dijkstra_path_length(
            G, source_node, weight=weight
        )

        # Fill in unreachable nodes with maxVal
        for node in G.nodes():
            if node not in path_lengths:
                path_lengths[node] = maxVal

        return path_lengths

    def compute_all_image_costmaps(self, pc_dict):
        img_costmaps = []
        num_images = len(self.img_paths)
        for i in tqdm(range(0, num_images), desc="computing non-da to goal distances"):
            # Get point cloud for current image
            pts3d = pc_dict[str(self.img_paths[i])] # (H, W, 3)

            # compute costmap for current image
            costmap = self.compute_single_image_costmap(i, pts3d) # (H, W)
            img_costmaps.append(costmap)
        
        img_costmaps = np.stack(img_costmaps, axis=0) # (N_images, H, W)
        return img_costmaps
    
    def compute_single_image_costmap(self, img_idx, pts3d, max_dist=1e6):
        """
        Compute distance-to-goal costmap for a single image using GPU acceleration.
        
        Args:
            img_idx: Image index
            pts3d: Point cloud of shape (H, W, 3)
            max_dist: Maximum distance for unreachable/invalid pixels
            
        Returns:
            np.ndarray: Costmap of shape (H, W) with distance to goal for each pixel
        """
        H, W = self.H, self.W
        
        costmap = np.full((H, W), max_dist, dtype=np.float32)
        
        pts3d_flat = pts3d.reshape(H * W, 3)  # (H*W, 3)

        # old costmap direct/non-direct edges classification code        
        # Step 1: Collect DA and Non-DA pixel information
        # da_pixel_indices = []
        # da_distances = [] 
        # nonda_pixel_indices = []        
        # for y in range(H):
        #     for x in range(W):
        #         node_id = self.pixel_coord_to_global_node_id(img_idx, x, y)
        #         linear_idx = y * W + x
        #         if node_id in self.G.nodes:
        #             node_type = self.G.nodes[node_id].get('type', 'unknown')
        #             if node_type in ['da', 'goal']:
        #                 da_pixel_indices.append(linear_idx)
        #                 da_distances.append(self.all_path_lengths[node_id])
        #                 costmap[y, x] = self.all_path_lengths[node_id]
        #             else:
        #                 nonda_pixel_indices.append(linear_idx)
        #         else:
        #             nonda_pixel_indices.append(linear_idx)
        # da_pixel_indices = np.array(da_pixel_indices, dtype=np.int32)
        # da_distances = np.array(da_distances, dtype=np.float32)
        # nonda_pixel_indices = np.array(nonda_pixel_indices, dtype=np.int32)

        # Vectorized DA/Non-DA classification: iterate over graph nodes (small set)
        # instead of all H*W pixels. Mathematically identical to the old row-major nested loop.
        
        # Original code loops over all (x, y) computing `node_id = img_idx * H * W + y * W + x`
        # Since `linear_idx = y * W + x`, we have `node_id = base_id + linear_idx`
        _base_id = img_idx * (H * W)
        _da_idx_list, _da_dist_list = [], []
        
        for nid in self.G.nodes:
            # Get current image's nodes
            if _base_id <= nid < _base_id + H * W:
                _ntype = self.G.nodes[nid].get('type', 'unknown')
                if _ntype in ['da', 'goal']:
                    # For DA/goal nodes, linear_idx is simply `nid - base_id`
                    _da_idx_list.append(nid - _base_id)
                    _da_dist_list.append(self.all_path_lengths[nid])

        if _da_idx_list:
            # Sorting the resulting indices ensures they are in the exact same ascending 
            # `linear_idx` (row-major) order as the original nested loop.
            _order = np.argsort(_da_idx_list)
            da_pixel_indices = np.array(_da_idx_list, dtype=np.int32)[_order]
            da_distances = np.array(_da_dist_list, dtype=np.float32)[_order]
        else:
            da_pixel_indices = np.array([], dtype=np.int32)
            da_distances = np.array([], dtype=np.float32)

        # The non-DA pixels are the complement of the DA pixels within the [0, H*W) range, computed efficiently using np.setdiff1d.
        nonda_pixel_indices = np.setdiff1d(
            np.arange(H * W, dtype=np.int32), da_pixel_indices
        )

        # Write DA distances into costmap (mirrors original costmap[y, x] = ...)
        da_y = da_pixel_indices // W
        da_x = da_pixel_indices % W
        costmap[da_y, da_x] = da_distances
        
        # print(f"  Image {img_idx}: {len(da_pixel_indices)} DA pixels, {len(nonda_pixel_indices)} Non-DA pixels")
        
        if len(da_pixel_indices) == 0:
            print(f"  Warning: No DA nodes in image {img_idx}, returning max distances")
            return costmap
        
        if len(nonda_pixel_indices) == 0:
            return costmap
        
        # Step 2: distance computation for Non-DA pixels
        nonda_distances = self.compute_nonda_distances(
            pts3d_flat, 
            nonda_pixel_indices, 
            da_pixel_indices,
            da_distances,
            max_dist
        )
        
        # Step 3: Fill costmap with Non-DA distances
        nonda_y = nonda_pixel_indices // W
        nonda_x = nonda_pixel_indices % W
        costmap[nonda_y, nonda_x] = nonda_distances
        
        return costmap

    def compute_nonda_distances( self,
        pts3d_flat: np.ndarray,  # (H*W, 3)
        nonda_indices: np.ndarray,  # (N_nonda,)
        da_indices: np.ndarray,  # (N_da,)
        da_distances: np.ndarray,  # (N_da,)
        max_dist: float = 1e6
    ) -> np.ndarray:
        """
        Compute distances for Non-DA pixels
        
        Args:
            pts3d_flat: All 3D points in image (H*W, 3)
            nonda_indices: Indices of Non-DA pixels
            da_indices: Indices of DA pixels
            da_distances: Distance-to-goal for DA pixels (N_da,)
            max_dist: Maximum distance for invalid pixels
            
        Returns:
            np.ndarray: Distances for Non-DA pixels (N_nonda,)
        """
        device = torch.device(self.device)
        
        # Get 3D coordinates
        nonda_pts3d = pts3d_flat[nonda_indices]  # (N_nonda, 3)
        da_pts3d = pts3d_flat[da_indices]  # (N_da, 3)
        
        # Convert to torch tensors
        nonda_pts3d = torch.from_numpy(nonda_pts3d).float().to(device)  # (N_nonda, 3)
        da_pts3d = torch.from_numpy(da_pts3d).float().to(device)  # (N_da, 3)
        da_distances = torch.from_numpy(da_distances).float().to(device)  # (N_da,)
        
        # Check for invalid depths (z < 0)
        nonda_valid = nonda_pts3d[:, 2] >= 0  # (N_nonda,)
        da_valid = da_pts3d[:, 2] >= 0  # (N_da,)
        
        # Compute pairwise 3D distances: (N_nonda, N_da)
        # Broadcasting: (N_nonda, 1, 3) - (1, N_da, 3) -> (N_nonda, N_da, 3)
        diff = nonda_pts3d.unsqueeze(1) - da_pts3d.unsqueeze(0)  # (N_nonda, N_da, 3)
        euclidean_dists = torch.norm(diff, dim=2)  # (N_nonda, N_da)
        
        # Add DA distances to goal
        total_dists = euclidean_dists + da_distances.unsqueeze(0)  # (N_nonda, N_da)
        
        # Mask out invalid DA points (set to max_dist)
        total_dists[:, ~da_valid] = max_dist
        
        # Find minimum distance to goal via any DA point
        min_dists, _ = torch.min(total_dists, dim=1)  # (N_nonda,)
        
        # Set invalid Non-DA points to max_dist
        min_dists[~nonda_valid] = max_dist
        
        # Move back to CPU and convert to numpy
        nonda_distances = min_dists.cpu().numpy()
        
        return nonda_distances

    def save_costmaps(self, costmaps: np.ndarray, metadata: dict, filename=None):
        """
        Save costmaps to compressed NPZ file.
        
        Args:
            costmaps: Array of shape (N_images, H, W)
            filename: Output filename
        """
        if filename is None:
            filename = self._get_costmap_filename()

        save_path = self.out_dir / filename
        
        np.savez_compressed(
            save_path,
            costmaps=costmaps,
            metadata=json.dumps(metadata)
        )
        
        print(f"✓ Saved costmaps to {save_path}")
    
    @staticmethod
    def load_costmap_file(filepath):
        """
        Load a costmap .npz file and return (costmap, metadata).
        Args:
            filepath (str or Path): Path to the .npz file.
        Returns:
            costmap (np.ndarray): The costmap array.
            metadata (dict): Metadata dictionary.
        """
        data = np.load(filepath, allow_pickle=True)
        costmap = data["costmaps"]
        metadata = json.loads(data["metadata"].item())
        return costmap, metadata

    def save_compressed_graph_chunked(self, path, graph=None):
        """
        Simple and reliable compression using blosc2 with manual chunking for large data
        """
        if graph is None:
            graph = self.G

        t_start = time.time()
        graph_data = self.decompose_graph_data(graph)

        # 1. Serialize data
        t_serialize = time.time()
        serialized = pickle.dumps(graph_data, protocol=pickle.HIGHEST_PROTOCOL)
        original_size = len(serialized)
        # print(f"Serialization took: {time.time() - t_serialize:.3f}s")
        # print(f"Original serialized size: {original_size / 1e6:.2f} MB")

        # 2. Compress using blosc2 with manual chunking for large data
        t_compress = time.time()
        compressed_path = f"{path}.b2s"
        # print("Compressing using blosc2 with manual chunking...")

        # blosc2 size limit (~2GB)
        BLOSC2_MAX_SIZE = 2000000000

        if original_size <= BLOSC2_MAX_SIZE:
            # Single compression for smaller data
            compressed = blosc2.compress(serialized, codec=blosc2.Codec.ZSTD, clevel=9)

            # Save with format marker
            with open(compressed_path, "wb") as f:
                f.write(b"SINGLE")  # 6-byte format marker
                f.write(len(compressed).to_bytes(8, "little"))  # size info
                f.write(compressed)

            compressed_size = len(compressed)
            compression_method = "single_blosc2"

        else:
            # Manual chunking for large data
            chunk_size = 1024 * 1024 * 1024  # 1GB chunks
            # chunk_size = BLOSC2_MAX_SIZE
            num_chunks = (len(serialized) + chunk_size - 1) // chunk_size
            compressed_chunks = []

            print(f"Data too large, using {num_chunks} chunks...")
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(serialized))
                chunk = serialized[start_idx:end_idx]

                # Compress each chunk with your preferred settings
                compressed_chunk = blosc2.compress(
                    chunk, codec=blosc2.Codec.ZSTD, clevel=9
                )
                compressed_chunks.append(compressed_chunk)
                # print(f"  Chunk {i+1}/{num_chunks}: {len(chunk):,} -> {len(compressed_chunk):,} bytes")

            # Save chunked data with format marker
            with open(compressed_path, "wb") as f:
                f.write(b"CHUNKS")  # 6-byte format marker
                f.write(num_chunks.to_bytes(4, "little"))  # number of chunks
                f.write(original_size.to_bytes(8, "little"))  # original size

                # Write chunk sizes
                for chunk in compressed_chunks:
                    f.write(len(chunk).to_bytes(4, "little"))

                # Write compressed chunks
                for chunk in compressed_chunks:
                    f.write(chunk)

            compressed_size = sum(len(chunk) for chunk in compressed_chunks)
            compression_method = "chunked_blosc2"

        # print(f"Compression ({compression_method}) took: {time.time() - t_compress:.3f}s")

        # Calculate stats
        compression_ratio = (1 - compressed_size / original_size) * 100

        # print(f"Total time: {time.time() - t_start:.3f}s")
        # print(f"Compressed size: {compressed_size:,} bytes ({compressed_size / 1e6:.2f} MB)")
        # print(f"Compression ratio: {compression_ratio:.1f}%")
        print(f"Saved to: {compressed_path} | Total Time: {time.time() - t_start:.3f}s")

        return compressed_path

    def decompose_graph_data(self, graph=None):
        t_start = time.time()

        if graph is None:
            graph = self.G

        # Basic graph structure
        graph_data = {"directed": graph.is_directed(), "graph_attrs": dict(graph.graph)}

        # Decompose nodes and their attributes
        nodes_list = list(graph.nodes(data=True))
        node_ids = []
        node_maps = []
        node_coords = []
        node_pixels = []
        node_types = []

        print(f"Processing {len(nodes_list)} nodes...")
        for node_id, attrs in nodes_list:
            node_ids.append(node_id)

            # Extract specific attributes
            node_maps.append(attrs.get("map", [0, 0]))
            node_coords.append(
                attrs.get("coord_mast3r", np.array([0, 0, 0], dtype=np.float64))
            )
            node_pixels.append(attrs.get("pixel", [0, 0]))
            node_types.append(attrs.get("type", "unknown"))

        # Convert to numpy arrays for better compression
        graph_data["node_ids"] = np.array(node_ids)
        graph_data["node_maps"] = np.array(node_maps, dtype=np.int32)
        graph_data["node_coords"] = np.array(node_coords, dtype=np.float64)
        graph_data["node_pixels"] = np.array(node_pixels, dtype=np.int32)
        graph_data["node_types"] = np.array(node_types, dtype="U20")  # String array

        # Decompose edges and their attributes
        edges_list = list(graph.edges(data=True))
        edge_sources = []
        edge_targets = []
        edge_types = []
        edge_weights = []

        print(f"Processing {len(edges_list)} edges...")
        for source, target, attrs in edges_list:
            edge_sources.append(source)
            edge_targets.append(target)
            edge_types.append(attrs.get("edge_type", "unknown"))
            edge_weights.append(attrs.get("weight", 0.0))

        # Convert to numpy arrays
        graph_data["edge_sources"] = np.array(edge_sources, dtype=np.int32)
        graph_data["edge_targets"] = np.array(edge_targets, dtype=np.int32)
        graph_data["edge_types"] = np.array(edge_types, dtype="U10")  # String array
        graph_data["edge_weights"] = np.array(edge_weights, dtype=np.float64)

        # Handle allPathLengths specially (convert dict to arrays)
        if "all_path_lengths" in graph_data["graph_attrs"]:
            path_lengths = graph_data["graph_attrs"]["all_path_lengths"]["weight"]
            path_nodes = np.array(list(path_lengths.keys()))
            path_distances = np.array(list(path_lengths.values()), dtype=np.float64)

            graph_data["path_nodes"] = path_nodes
            graph_data["path_distances"] = path_distances
            del graph_data["graph_attrs"]["all_path_lengths"]
            print(f"Converted {len(path_lengths)} path lengths to arrays")

        print(f"Data preparation took: {time.time() - t_start:.3f}s")
        return graph_data

    def load_base_graph_and_add_goal(self):
        """
        Load existing base graph, add goal from config, compute distances, save new files.
        
        This method is used for update_graph mode to avoid rebuilding the entire graph.
        """
        # Load base graph
        base_graph_path = self.scene_dir / self.cfg.goal.base_graph_path
        if not base_graph_path.exists():
            raise FileNotFoundError(f"Base graph not found: {base_graph_path}")
        
        print(f"Loading base graph from: {base_graph_path}")
        self.G = load_compressed_graph_chunked(str(base_graph_path))
        
        # Get goal from config and compute distances
        goal_img_idx = self.cfg.goal.image_idx
        goal_px = self.cfg.goal.pixel_x
        goal_py = self.cfg.goal.pixel_y
        
        # If multiple scenes are to be processed, and the costmaps need to be compiled (implying that this is map compilation for the training stage), then automatically set the goal image index to the last image frame
        if not goal_img_idx and (self.cfg.scenes.multi_scene and self.cfg.scenes.compile_costmaps):
            goal_img_idx = len(self.img_paths) - 1

        self.compute_distances_to_goal_node(goal_img_idx, goal_px, goal_py)
        
        # Save graph with goal
        graph_filename = self._get_goal_graph_filename(goal_img_idx)
        self.save_compressed_graph_chunked(str(self.out_dir / graph_filename))
        
        print(f"✓ Updated graph saved with goal node")
        return self.G

def make_topo_map(scene_dir: Path, img_dir: Path, out_dir: Path, cfg: DictConfig) -> nx.Graph:
    """
    Create topological map for a single scene.
    
    Args:
        scene_dir: Scene directory path
        img_dir: Images directory path
        out_dir: Output directory path
        cfg: Hydra configuration
        
    Returns:
        NetworkX graph with distances to goal node
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING SCENE: {scene_dir.name}")
    print(f"{'='*80}\n")

    # Get goal mode (default to "config" for backward compatibility)
    goal_mode = cfg.goal.get("mode", "config")
    print(f"Goal mode: {goal_mode}")

    # Initialize mapper
    start_time = time.time()
    topo_map = MapTopological3DPoints(str(img_dir), str(out_dir), cfg)

    # Handle update_graph mode separately (doesn't create new graph)
    if goal_mode == "update_graph":
        print("\nMode: UPDATE_GRAPH - Loading base graph and adding goal")
        topo_map.load_base_graph_and_add_goal()
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ SCENE COMPLETE in {total_time:.2f}s")
        print(f"{'='*80}\n")
        return topo_map.G

    # For all other modes: create topological map first
    print(f"\nMode: {goal_mode.upper()} - Creating topological map...")
    t1 = time.time()
    topo_map.create_map_topo()
    print(f"✓ Created topological map in {time.time() - t1:.2f}s")
    print(f"  Graph: {topo_map.G.number_of_nodes()} nodes, {topo_map.G.number_of_edges()} edges")

    # Save base graph (always, before adding goal distances)
    print("\nSaving base graph...")
    base_filename = topo_map._get_base_graph_filename()
    if cfg.compression.enabled:
        topo_map.save_compressed_graph_chunked(str(out_dir / base_filename))
    else:
        with open(out_dir / base_filename, 'wb') as f:
            pickle.dump(topo_map.G, f)

    # Compute goal distances (for config and episode modes)
    if goal_mode == "episode":
        print("\nInferring goal from episode folder...")
        topo_map.get_goal_from_episode()
        
        print("Computing distances to goal node...")
        t2 = time.time()
        topo_map.compute_distances_to_goal_node()
        print(f"✓ Computed distances in {time.time() - t2:.2f}s")
        
        # Save goal graph
        print("\nSaving goal graph...")
        goal_filename = topo_map._get_goal_graph_filename()
        if cfg.compression.enabled:
            topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
        else:
            with open(out_dir / goal_filename, 'wb') as f:
                pickle.dump(topo_map.G, f)

    elif goal_mode == "config":
        print("\nComputing distances to goal node...")
        t2 = time.time()
        topo_map.compute_distances_to_goal_node()
        print(f"✓ Computed distances in {time.time() - t2:.2f}s")
        
        # Save goal graph
        print("\nSaving goal graph...")
        goal_filename = topo_map._get_goal_graph_filename()
        if cfg.compression.enabled:
            topo_map.save_compressed_graph_chunked(str(out_dir / goal_filename))
        else:
            with open(out_dir / goal_filename, 'wb') as f:
                pickle.dump(topo_map.G, f)

    elif goal_mode == "none":
        # Base graph already saved, nothing more to do
        pass

    else:
        raise ValueError(f"Unknown goal mode: {goal_mode}. Expected: none, config, episode, update_graph")
    
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ SCENE COMPLETE in {total_time:.2f}s")
    print(f"{'='*80}\n")

    return topo_map.G

def get_scene_list(cfg: DictConfig) -> list:
    """
    Get list of scene paths based on config.
    
    Priority:
    1. scene_list_file (if provided)
    2. start_idx/end_idx/step filtering
    3. Single scene_name (when multi_scene=false)
    """
    base_dir = Path(cfg.scenes.base_dir)
    
    if not cfg.scenes.multi_scene:
        # Single scene mode
        scene_path = base_dir / cfg.scenes.scene_name
        print(f"Single scene mode: {scene_path.name}")
        return [scene_path]
    
    # Multi-scene mode: check txt file first
    list_file = cfg.scenes.get("scene_list_file")
    if list_file and Path(list_file).exists():
        with open(list_file, 'r') as f:
            names = [line.strip() for line in f if line.strip()]
        scenes = [base_dir / name for name in names if (base_dir / name).exists()]
        print(f"Multi-scene mode (from file): {len(scenes)} scenes")
        return scenes
    
    # Multi-scene mode: use start/end/step
    all_scenes = natsorted([p for p in base_dir.iterdir() if p.is_dir()], key=lambda x: x.name)
    
    start = cfg.scenes.get("start_idx", 0)
    end = cfg.scenes.get("end_idx", -1)
    step = cfg.scenes.get("step", 1)
    
    if end == -1:
        end = len(all_scenes)
    
    scenes = all_scenes[start:end:step]
    print(f"Multi-scene mode: {len(scenes)} scenes (indices {start}:{end}:{step})")
    return scenes


def compile_costmaps_into_h5_file(base_dir: Path, output_file: Path):
    # Converts the individually generated into a training compatible H5 file from which the controller can look up the costmaps linked to each frame in each trajectory.
    # Adds a new key of "pls_pixels" to the H5 files, which is then referenced in the integrate.py file in the controller repo, to load the pixel level costmaps
    print(f"Creating consolidated HDF5 file: {output_file}")
    with h5py.File(output_file, "w") as f:
        for scene_name in tqdm(os.listdir(base_dir), desc="Compiling scenes"):
            scene_path = os.path.join(base_dir, scene_name)
            
            # Find the costmaps npz file
            npz_files = glob.glob(os.path.join(scene_path, "costmaps_*.npz"))
            if not npz_files:
                # Some directories might not have costmaps computed yet
                continue
                
            npz_path = npz_files[0]
            
            try:
                # Load the costmap data
                data = np.load(npz_path, allow_pickle=True)
                # costmaps shape is typically (N_images, 240, 320)
                costmaps = data["costmaps"]
                
                num_images = costmaps.shape[0]
                for img_idx in range(num_images):
                    key = f"{scene_name}_{img_idx}"
                    
                    # Get the costmap for this frame
                    costmap_frame = costmaps[img_idx] # (H, W)
                    
                    # Downsample dynamically to match the model's expected shape of (60, 80)
                    h, w = costmap_frame.shape
                    h_downsampling_ratio = max(1, h // 60)
                    w_downsampling_ratio = max(1, w // 80)
                    
                    costmap_downsampled = costmap_frame[::h_downsampling_ratio, ::w_downsampling_ratio]
                    
                    # Create H5 group and save dataset
                    grp = f.create_group(key)
                    grp.create_dataset("pls_pixels", data=costmap_downsampled)
                    
            except Exception as e:
                print(f"Error processing {scene_name}: {e}")

    print(f"Successfully compiled dense costmaps to {output_file}!")


@hydra.main(version_base=None, config_path="../../configs/mapper", config_name="mapper_config")
def main(cfg: DictConfig):
    print("\n" + "="*80)
    print("TOPOLOGICAL MAP CREATION")
    print("="*80)
    print(f"\nUsing configuration from: {cfg}")
    print("="*80 + "\n")

    # Set environment variable for base directory
    os.environ['BASE_DIR'] = str(BASE_DIR)

    # Get scene list
    scenes = get_scene_list(cfg)
    
    if len(scenes) == 0:
        raise ValueError("No scenes found to process")
    
    # # Disable goal computation for multi-scene (base graph only)
    # if cfg.scenes.multi_scene:
    #     print("Multi-scene mode: disabling goal computation (base graph only)")
    
    # Process each scene
    results = {}
    base_out_dir = cfg.scenes.get("base_out_dir", None)
    
    for scene_num, scene_dir in enumerate(tqdm(scenes, desc="Processing scenes", unit="scene")):
        img_dir = scene_dir / "images"
        if scene_dir.name == 'q5QZSEeHe5g_0000000_plant_32_':
            continue
        
        # Determine output directory
        if base_out_dir is not None:
            out_dir = Path(base_out_dir) / scene_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = scene_dir  # Output to same scene directory
        
        if not img_dir.exists():
            print(f"⚠ Skipping {scene_dir.name}: no images/ folder")
            results[scene_dir.name] = False
            continue
        
        print(f"\nScene: {scene_dir.name}")
        print(f"Images: {img_dir}")
        print(f"Output: {out_dir}\n")
        
        graph = make_topo_map(scene_dir, img_dir, out_dir, cfg)
        results[scene_dir.name] = graph is not None
    
    # Compiling the costmaps into H5 file for training
    if cfg.scenes.multi_scene and cfg.scenes.compile_costmaps:
        print("Accidentally entered new code; debug")
        if cfg.scenes.compiled_costmaps_file:
            compiled_costmaps_file = Path(cfg.scenes.compiled_costmaps_file)
        else:
            compiled_costmaps_file = base_out_dir / "compiled_costmaps.h5"
            print(f"No path to a compiled costmaps file supplied; saving the compiled costmaps to {compiled_costmaps_file}")
        
        assert not compiled_costmaps_file.is_dir(), "Please make sure that `compiled_costmaps_file` is not a directory, but the path to a file"
        compile_costmaps_into_h5_file(
            base_dir = cfg.scenes.base_dir,
            output_file = cfg.scenes.compiled_costmaps_file
        )

    # Summary
    successful = sum(results.values())
    print(f"\n{'='*80}")
    print(f"✓ COMPLETE: {successful}/{len(results)} scenes processed successfully")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()