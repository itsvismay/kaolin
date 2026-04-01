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

import torch
import pytest
from kaolin.physics.simplicits.winding_number import (
    DiscCut,
    compute_disc_winding_number, compute_winding_number,
)


class TestDiscCut:
    """Tests for DiscCut and compute_disc_winding_number."""

    @pytest.fixture
    def disc(self):
        """Unit disc in the XZ plane (normal=Y), centred at origin, radius=1."""
        return DiscCut(
            normal=torch.tensor([0., 1., 0.]),
            center=torch.tensor([0., 0., 0.]),
            radius=1.0,
        )

    def test_output_shape(self, disc):
        pts = torch.randn(50, 3)
        disc.alpha = 1.0
        H = compute_disc_winding_number(pts, disc)
        assert H.shape == (50, 1)

    def test_opposite_sides_opposite_sign(self, disc):
        """Points directly above and below the disc centre should have opposite H."""
        pts = torch.tensor([
            [0., 0.01, 0.],   # just above disc — H > 0
            [0., -0.01, 0.],  # just below disc — H < 0
        ])
        disc.alpha = 1.0
        H = compute_disc_winding_number(pts, disc)
        assert H[0, 0].item() > 0
        assert H[1, 0].item() < 0

    def test_far_outside_radius_near_zero(self, disc):
        """Points well outside the disc radius should see H ≈ 0."""
        pts = torch.tensor([
            [2.0, 0.01, 0.],   # radial dist = 2.0 >> radius=1.0
            [0.0, 0.01, 2.0],
        ])
        disc.alpha = 1.0
        H = compute_disc_winding_number(pts, disc)
        assert H.abs().max().item() == pytest.approx(0.0, abs=1e-6)

    def test_alpha_zero_is_zero_everywhere(self, disc):
        """alpha=0 means effective_radius=0, so H=0 everywhere."""
        pts = torch.randn(20, 3)
        disc.alpha = 0.0
        H = compute_disc_winding_number(pts, disc)
        assert H.abs().max().item() == pytest.approx(0.0, abs=1e-4)

    def test_alpha_grows_active_region(self, disc):
        """A point at radial dist 0.6 should be active at alpha=1 but not alpha=0.5."""
        # radial dist = 0.6, radius = 1.0 → inside at alpha=1, outside at alpha=0.5
        pts = torch.tensor([[0.6, 0.01, 0.]])
        disc.alpha = 1.0
        H_full = compute_disc_winding_number(pts, disc)
        disc.alpha = 0.5
        H_half = compute_disc_winding_number(pts, disc)
        assert H_full.abs().item() > 0.0
        assert H_half.abs().item() == pytest.approx(0.0, abs=1e-6)

    def test_device_transfer(self, disc):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        pts = torch.randn(10, 3, device='cuda')
        disc.alpha = 1.0
        H = compute_disc_winding_number(pts, disc)
        assert H.device.type == 'cuda'

    def test_dispatch_disc(self):
        cut = DiscCut(normal=torch.tensor([0., 1., 0.]),
                      center=torch.tensor([0., 0., 0.]),
                      radius=1.0)
        pts = torch.randn(10, 3)
        cut.alpha = 0.7
        H_direct = compute_disc_winding_number(pts, cut)
        H_dispatch = compute_winding_number(pts, cut)
        assert torch.allclose(H_direct, H_dispatch)

    def test_dispatch_unknown_raises(self):
        with pytest.raises(TypeError):
            compute_winding_number(torch.randn(5, 3), "not_a_cut")
