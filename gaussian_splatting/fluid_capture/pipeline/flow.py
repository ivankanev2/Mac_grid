"""Optical flow estimation via RAFT (torchvision).

We use ``torchvision.models.optical_flow.raft_large`` with the default
pretrained weights.  RAFT requires inputs whose H and W are divisible
by 8, so we resize, run the model, then resize the predicted flow back
to the original image size and rescale the flow vectors accordingly.

Output is the per-pixel pixel-space displacement from frame A to frame
B, in units of *pixels* (not normalised).  ``flow[r, c, 0]`` is the
horizontal (column-direction) displacement; ``flow[r, c, 1]`` is the
vertical (row-direction) displacement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF


def _select_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class FlowEstimator:
    """Wrapper around torchvision RAFT for image-pair optical flow."""

    def __init__(self, device: Optional[str] = None):
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        self.device = _select_device(device)
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).to(self.device).eval()
        self.transforms = self.weights.transforms()

    @staticmethod
    def _to_chw(image_rgb: np.ndarray) -> torch.Tensor:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) RGB image, got {image_rgb.shape}")
        if image_rgb.dtype == np.uint8:
            t = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        else:
            t = torch.from_numpy(image_rgb.astype(np.float32)).permute(2, 0, 1)
        return t

    @torch.no_grad()
    def predict(
        self,
        image_a_rgb: np.ndarray,
        image_b_rgb: np.ndarray,
    ) -> np.ndarray:
        """Predict pixel-space optical flow from frame A to frame B.

        Returns ``(H, W, 2)`` float32 with channels (du, dv) — i.e. the
        column displacement and the row displacement, both in pixels at
        the original input resolution.
        """
        if image_a_rgb.shape != image_b_rgb.shape:
            raise ValueError(
                f"Frame shapes must match; got {image_a_rgb.shape} vs {image_b_rgb.shape}"
            )

        H, W = image_a_rgb.shape[:2]
        a = self._to_chw(image_a_rgb)
        b = self._to_chw(image_b_rgb)

        # RAFT expects sizes divisible by 8.
        H8 = (H // 8) * 8
        W8 = (W // 8) * 8
        if H8 == 0 or W8 == 0:
            raise ValueError(f"Image too small for RAFT: {W}x{H}")
        a_r = TF.resize(a, [H8, W8], antialias=True)
        b_r = TF.resize(b, [H8, W8], antialias=True)

        # Apply RAFT preprocessing (normalisation + batch dim).
        a_t, b_t = self.transforms(a_r.unsqueeze(0), b_r.unsqueeze(0))
        a_t, b_t = a_t.to(self.device), b_t.to(self.device)

        flow_predictions = self.model(a_t, b_t)
        # The model returns a list of refinements; the last entry is the final flow.
        flow = flow_predictions[-1]  # (1, 2, H8, W8)

        # Resize back to original resolution and rescale flow magnitudes.
        flow = torch.nn.functional.interpolate(
            flow,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )
        # Channel 0 is horizontal (du), channel 1 is vertical (dv).
        flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        flow_np[..., 0] *= float(W) / float(W8)
        flow_np[..., 1] *= float(H) / float(H8)
        return flow_np
