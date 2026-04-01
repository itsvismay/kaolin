# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import torch

__all__ = [
    'DiscCut',
    'compute_disc_winding_number',
    'compute_winding_number',
]


@dataclass
class DiscCut:
    r"""Defines a finite disc-shaped cut surface for fracture simulation.

    The cut is a disc of given radius, centred at ``center`` and oriented by
    ``normal``.  The lifted winding number approximates a jump discontinuity
    across the disc using a steep :func:`torch.tanh` in the normal direction,
    multiplied by a radial falloff that is 1 at the centre and 0 at the disc
    boundary.  Points beyond the disc radius see ``H = 0``.

    The cut-progress parameter ``alpha`` controls the **effective radius** of
    the cut: ``effective_radius = alpha * radius``.  At ``alpha = 0`` the
    effective radius is zero and ``H = 0`` everywhere (uncut).  At
    ``alpha = 1`` the full disc is active.  This allows the fracture to be
    animated as the disc grows from the centre outward.

    .. math::
        H(x) = \tanh\!\left(\frac{s \cdot \mathbf{n} \cdot (x - c)}{r}\right)
                \cdot h\_scale \cdot \mathbf{1}\!\left[\|x_\perp - c\| \le \alpha r\right]

    where :math:`x_\perp` is the projection of :math:`x` onto the disc plane,
    :math:`r` is the disc radius, and :math:`s` is ``tanh_sharpness``.
    Points outside the active cut front get ``H = 0`` exactly (binary mask —
    no intermediate blending), so opposite sides of the cut are never mixed.

    Attributes:
        normal (torch.Tensor): ``(3,)`` unit normal to the disc plane.
        center (torch.Tensor): ``(3,)`` centre of the disc (world space).
        radius (float): Radius of the disc (world-space units).
        alpha (float): Cut progress in ``[0, 1]``.  Controls the effective
            radius: ``effective_radius = alpha * radius``.  Defaults to 1.0.
        tanh_sharpness (float): Controls how quickly ``H`` transitions across
            the cut plane.  Normalised by ``radius`` so the transition width
            is object-scale-independent.  Higher values produce a sharper
            step; the transition half-width is approximately
            ``radius / tanh_sharpness``.  Defaults to 100.0.
        h_scale (float): Saturation value of ``H``.  ``H`` is scaled so it
            saturates to ``±h_scale`` rather than ``±1``.  A larger value
            exaggerates the 4D separation between opposite sides, making RKPM
            kernels decay more steeply across the cut.  Defaults to 1.0.
    """
    normal: torch.Tensor
    center: torch.Tensor
    radius: float
    alpha: float = 1.0
    tanh_sharpness: float = 100.0
    h_scale: float = 1.0


def compute_disc_winding_number(pts, cut):
    r"""Compute the lifted winding number for a disc-shaped cut.

    Approximates a jump discontinuity across the disc using a steep
    :func:`torch.tanh` in the normal direction multiplied by a linear radial
    falloff.  The effective radius of the disc is ``cut.alpha * cut.radius``,
    so animating ``cut.alpha`` from 0 to 1 grows the cut from the centre
    outward.

    - ``cut.alpha=0``: effective radius = 0, ``H=0`` everywhere (uncut).
    - ``cut.alpha=1``: full disc active, strong jump across the disc plane.

    Args:
        pts (torch.Tensor): ``(N, 3)`` world-space query points.
        cut (DiscCut): Cut surface definition, including ``cut.alpha``.

    Returns:
        torch.Tensor: ``(N, 1)`` lifted winding number values.
    """
    normal = cut.normal.to(device=pts.device, dtype=pts.dtype)
    center = cut.center.to(device=pts.device, dtype=pts.dtype)

    # Signed distance from the disc plane — determines which side each point is on.
    signed_dist = ((pts - center) * normal).sum(dim=-1, keepdim=True)  # (N, 1)

    # Radial distance from the disc axis in the plane of the disc.
    proj_in_plane = pts - signed_dist * normal                              # (N, 3)
    radial_dist = torch.norm(proj_in_plane - center, dim=-1, keepdim=True)  # (N, 1)

    # Binary mask: points within the active cut front get the full tanh step,
    # points outside get H=0. No intermediate values — avoids the blending artefact
    # where both sides of the cut appeared at H≈0 near the disc boundary.
    effective_radius = cut.radius * cut.alpha
    inside_front = (radial_dist <= effective_radius).to(pts.dtype)  # (N, 1), 0 or 1

    # Approximate Heaviside across the disc plane via steep tanh.
    # Scale is normalised by radius so the jump width is object-scale-independent.
    scale = cut.tanh_sharpness / cut.radius
    H = torch.tanh(signed_dist * scale) * inside_front * cut.h_scale  # (N, 1)

    return H


def compute_winding_number(pts, cut):
    r"""Compute the lifted winding number for a cut surface.

    Currently supports :class:`DiscCut`.  The cut progress is read from
    ``cut.alpha``; update it before calling this function to animate the cut.

    Args:
        pts (torch.Tensor): ``(N, 3)`` world-space query points.
        cut (DiscCut): Cut surface definition.

    Returns:
        torch.Tensor: ``(N, 1)`` lifted winding number values.

    Raises:
        TypeError: If ``cut`` is not a recognised cut type.
    """
    if isinstance(cut, DiscCut):
        return compute_disc_winding_number(pts, cut)
    else:
        raise TypeError(f"Unrecognised cut type: {type(cut)}. Expected DiscCut.")
