"""
MASt3R Inference Module

Provides a unified interface for MASt3R model operations including:
- Model loading
- Batched 3D point cloud inference (self-matching)
- Batched image pair matching
- Depth conversion utilities
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Union, List

import numpy as np
import torch
import torchvision.transforms as tfm
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

# Add MASt3R to path
BASE_DIR = Path(__file__).parent.parent.parent
MAST3R_PATH = BASE_DIR / "libs" / "matcher" / "mast3r"
if str(MAST3R_PATH) not in sys.path:
    sys.path.insert(0, str(MAST3R_PATH))

# Default model path (inside the mast3r repo)
# DEFAULT_MODEL_PATH = MAST3R_PATH / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
DEFAULT_MODEL_CANDIDATES = (
    BASE_DIR / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    MAST3R_PATH / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
)

class MASt3RInference:
    """
    Unified interface for MASt3R model operations with batched inference support.

    Provides simple methods for:
    - get_pts3d(): Get 3D point cloud from single image (self-matching)
    - get_matches(): Get 2D-2D matches between two images
    - infer(): Raw inference on image pair

    Example — batched inference::

        mast3r = MASt3RInference(device="cuda")
        output = mast3r.infer(imgs0_paths, imgs1_paths, resize=(320, 240), batch_size=8)
        pts3d = mast3r.get_pts3d([img_path1, img_path2], resize=(320, 240))
        matches0, matches1 = mast3r.get_matches(img0_path, img1_path)
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize MASt3R model.

        Args:
            model_path: Path to model weights. Uses default if None.
            device: Compute device ("cuda" or "cpu")
        """
        from mast3r.model import AsymmetricMASt3R

        self.device = device
        # self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.model_path = self._resolve_model_path(model_path)

        logger.info(f"Loading MASt3R model from {self.model_path}")
        self.model = AsymmetricMASt3R.from_pretrained(str(self.model_path)).to(device)
        logger.info("MASt3R model loaded successfully")

        # Image normalization
        self.normalize = tfm.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Path:
        """Resolve the local MASt3R checkpoint path before calling from_pretrained."""
        if model_path:
            resolved = Path(model_path).expanduser().resolve()
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(f"MASt3R checkpoint not found at: {resolved}")

        for candidate in DEFAULT_MODEL_CANDIDATES:
            resolved = candidate.resolve()
            if resolved.is_file():
                return resolved

        searched = "\n".join(f"- {candidate.resolve()}" for candidate in DEFAULT_MODEL_CANDIDATES)
        raise FileNotFoundError(
            "MASt3R checkpoint not found. Searched:\n"
            f"{searched}\n"
            "Download the checkpoint to ./checkpoints/ or pass model_path explicitly."
        )

    @staticmethod
    def _to_list(x) -> list:
        """
        Ensure *x* is a list.

        Passing a single image (``str``, ``Path``, or ``np.ndarray``) is
        silently promoted to a one-element list for backward compatibility.
        """
        if isinstance(x, (str, Path, np.ndarray)):
            return [x]
        return list(x)

    def _load_single(self, img: Union[np.ndarray, str, Path],
                     resize: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        Load one image and return a normalised ``(1, 3, H, W)`` tensor on CPU.

        Args:
            img:    Image as a file path (``str`` / ``Path``) or numpy RGB array.
            resize: Optional ``(H, W)`` target size applied with antialias resize.

        Returns:
            Tensor of shape ``(1, 3, H, W)``, values in ``[-1, 1]``.
        """
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        t = tfm.ToTensor()(img)
        if resize is not None:
            t = tfm.Resize(resize, antialias=True)(t)
        return self.normalize(t).unsqueeze(0)  # (1, 3, H, W)

    def _prepare_pairs(self, imgs0: List, imgs1: List,
                       resize: Optional[Tuple[int, int]] = None) -> list:
        """
        Build the list of ``(view0_dict, view1_dict)`` tuples expected by
        ``dust3r.inference.inference()``.

        Each view dict contains:

        - ``"img"``        – ``(1, 3, H, W)`` float tensor on ``self.device``
        - ``"true_shape"`` – ``(1, 2)`` int32 numpy array ``[[H, W]]``
        - ``"idx"``        – integer pair index
        - ``"instance"``   – integer instance id (same as idx)

        Args:
            imgs0:  List of N first images (path or np.ndarray).
            imgs1:  List of N second images (path or np.ndarray). Must match ``len(imgs0)``.
            resize: Optional ``(H, W)`` resize applied to every image.

        Returns:
            List of N ``(view_dict, view_dict)`` tuples.
        """
        assert len(imgs0) == len(imgs1), (
            f"imgs0 and imgs1 must have the same length, got {len(imgs0)} vs {len(imgs1)}"
        )
        pairs = []
        for i, (im0, im1) in enumerate(zip(imgs0, imgs1)):
            t0 = self._load_single(im0, resize).to(self.device)  # (1, 3, H, W)
            t1 = self._load_single(im1, resize).to(self.device)
            h, w = t0.shape[-2], t0.shape[-1]
            view0 = {"img": t0, "true_shape": np.int32([[h, w]]), "idx": i, "instance": i}
            view1 = {"img": t1, "true_shape": np.int32([[h, w]]), "idx": i, "instance": i}
            pairs.append((view0, view1))
        return pairs

    def infer(self, imgs0: List, imgs1: List,
              resize: Optional[Tuple[int, int]] = None,
              batch_size: int = 8) -> dict:
        """
        Run batched MASt3R inference over N image pairs.

        Images are preprocessed one-by-one then grouped into batches of
        ``batch_size`` pairs for the GPU forward pass, keeping peak VRAM
        bounded regardless of the total number of pairs.

        Args:
            img0: First image (np.ndarray, path, or Path)
            img1: Second image (np.ndarray, path, or Path)
            resize: Optional resize dimensions (H, W)

        Returns:
            dict: Raw MASt3R output containing view1, view2, pred1, pred2
        """
        imgs0 = self._to_list(imgs0)
        imgs1 = self._to_list(imgs1)

        from dust3r.inference import inference

        pairs = self._prepare_pairs(imgs0, imgs1, resize)
        with torch.no_grad():
            output = inference(pairs, self.model, self.device,
                               batch_size=batch_size, verbose=False)
        return output

    def get_pts3d(self, imgs: List,
                  resize: Optional[Tuple[int, int]] = None,
                  batch_size: int = 8) -> np.ndarray:
        """
        Get 3D point cloud from single image using self-matching.

        Args:
            img: Input image (np.ndarray, path, or Path)
            resize: Optional resize dimensions (H, W)

        Returns:
            pts3d: (H, W, 3) numpy array of 3D points
        """
        imgs = self._to_list(imgs)
        output = self.infer(imgs, imgs, resize=resize, batch_size=batch_size)
        return output["pred1"]["pts3d"].cpu().numpy()

    def get_matches(self, img0, img1,
                    resize: Optional[Tuple[int, int]] = None,
                    subsample: int = 8,
                    border: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get 2D-2D matches between two images or lists of images.

        Args:
            img0: First image
            img1: Second image
            resize: Optional resize dimensions (H, W)
            subsample: Subsampling factor for fast_reciprocal_NNs
            border: Border pixels to ignore
        Returns:
            matches_im0: (N, 2) array of match coordinates in img0 [x, y]
            matches_im1: (N, 2) array of match coordinates in img1 [x, y]
        """
        from mast3r.fast_nn import fast_reciprocal_NNs

        # Always pass lists; _to_list handles both single-image and list inputs
        output = self.infer(self._to_list(img0), self._to_list(img1), resize)

        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()

        # Find 2D-2D matches
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1, desc2,
            subsample_or_initxy1=subsample,
            device=self.device,
            dist="dot",
            block_size=2**13,
        )

        # Filter border matches
        H0, W0 = view1["true_shape"][0]
        H1, W1 = view2["true_shape"][0]

        valid = (
            (matches_im0[:, 0] >= border) & (matches_im0[:, 0] < int(W0) - border) &
            (matches_im0[:, 1] >= border) & (matches_im0[:, 1] < int(H0) - border) &
            (matches_im1[:, 0] >= border) & (matches_im1[:, 0] < int(W1) - border) &
            (matches_im1[:, 1] >= border) & (matches_im1[:, 1] < int(H1) - border)
        )

        return matches_im0[valid], matches_im1[valid]

    @staticmethod
    def pts3d_to_depth(pts3d: np.ndarray) -> np.ndarray:
        """
        Convert a pts3d array to a depth map using the Z channel.
        Args:
            pts3d: (H, W, 3) point cloud
        Returns:
            depth: (H, W) depth map
        """
        Z = pts3d[..., 2]
        return np.where(Z > 0, Z, 0.0).astype(np.float32)

