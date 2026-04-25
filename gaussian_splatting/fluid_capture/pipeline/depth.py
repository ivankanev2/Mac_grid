"""Monocular depth estimation via Depth Anything V2 (HuggingFace).

We use the HuggingFace ``transformers`` interface rather than the
original repo because it handles model loading, weight caching and
device placement uniformly, and the V2 weights are first-class on the Hub.

Default model is the Base variant (~97 M params) — it's the sweet spot
between quality and speed on Apple Silicon MPS.  Swap in Small-hf for
~3× speedup on rougher quality, or Large-hf for ~3× slowdown on
noticeably better quality.

Output is *relative* depth: larger values mean farther away, but the
absolute scale is arbitrary.  For our pipeline this is fine — the
simulator will be told what world-scale to use at voxelisation time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from PIL import Image


def _select_device(requested: Optional[str]) -> str:
    """Pick a device: explicit request, then MPS, then CUDA, then CPU."""
    if requested:
        return requested
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DepthEstimator:
    """Wrapper around Depth Anything V2 for per-frame depth prediction."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Base-hf",
        device: Optional[str] = None,
    ):
        # Lazy-import transformers so the package import doesn't pull it in
        # for callers that only need frames/segmentation/etc.
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = _select_device(device)
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray) -> np.ndarray:
        """Predict depth from an (H, W, 3) RGB uint8 image.

        Returns a (H, W) float32 array of relative depth values (larger =
        farther away).  The output is resized back to the input
        resolution by bicubic interpolation.
        """
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got {image_rgb.shape}")
        if image_rgb.dtype != np.uint8:
            arr = image_rgb
            if arr.max() <= 1.0:
                arr = (arr * 255.0)
            image_rgb = arr.astype(np.uint8)

        H, W = image_rgb.shape[:2]
        pil = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # outputs.predicted_depth is (1, H', W'); resize to original.
        predicted = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)
        return depth
