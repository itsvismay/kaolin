# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Union, Optional

import torch
import torch.nn as nn

from kaolin.ops.pointcloud import farthest_point_sampling
from kaolin.physics.materials.material_utils import to_lame
from kaolin.physics.simplicits.winding_number import compute_winding_number
from scipy.spatial import cKDTree

__all__ = [
    'SimplicitsRKPM',
]

class SimplicitsRKPM(nn.Module):
    r"""Simplicits skinning weights using Reproducing Kernel Particle Method (RKPM).

    Computes skinning weights for Simplicits physics simulation using RKPM basis
    functions. The skinning weights are derived from the eigenvectors of a
    generalized eigenvalue problem involving the mass and elastic hessian matrices
    assembled from RKPM kernel evaluations.

    Supports both 3D and 4D (cut-aware) modes. In 4D mode, points are lifted to
    ``(x, y, z, H)`` where ``H`` is the winding number across a :class:`DiscCut`,
    so eigenvectors localize to one side of the cut.

    Project link: https://research.nvidia.com/labs/sil/projects/freeform

    Args:
        num_handles (int): Number of deformation handles (non-zero eigenvectors) to use.
        num_nodes (int): Number of RKPM kernel nodes.
        kernel_type (str, optional): Type of kernel function. Currently only
            ``"gaussian"`` is supported. Defaults to ``"gaussian"``.
        radius_scale (float, optional): Scaling factor applied to node radii
            computed from nearest-neighbor distances. Defaults to 1.0.
        radius_init_kNN (int, optional): Number of nearest neighbors used to
            determine initial node radius. Defaults to 2.
        radius_min (Union[float, str, None], optional): Minimum node radius.
            Can be a float value, or a string of the form ``"Nx"`` (e.g.
            ``"3x"``) to set the minimum as a multiple of the mean
            nearest-neighbor distance among input points. Defaults to ``"3x"``.
        num_points (int, optional): Number of integration sample points for
            eigenanalysis. If None, all input points are used. Defaults to None.
        use_double (bool, optional): Whether to force double precision for all
            RKPM computations. Defaults to True.
        cutoff_factor (float, optional): Truncation radius for Gaussian kernels,
            in units of the node radius. Kernels are zeroed beyond
            ``cutoff_factor * radius``. ``None`` disables truncation
            (infinite support). Defaults to 3.0.
    """

    def __init__(
        self,
        num_handles: int,
        num_nodes: int,
        kernel_type: str = "gaussian",
        radius_scale: float = 1.0,
        radius_init_kNN: int = 2,
        radius_min: Union[float, str, None] = "3x",
        num_points: Optional[int] = None,
        use_double: bool = True,
        cutoff_factor: Optional[float] = 3.0,
    ):
        super(SimplicitsRKPM, self).__init__()
        self.num_points = num_points
        self.num_handles = num_handles
        self.num_nodes = num_nodes

        self.kernel_type = kernel_type
        self.radius_scale = radius_scale
        self.radius_init_kNN = radius_init_kNN
        self.radius_min = radius_min
        self.cutoff_factor = cutoff_factor

        self.rkpm = RKPM(num_nodes, kernel_type, cutoff_factor=cutoff_factor)

        self.use_double = use_double  # force double precision for all computations involving RKPM
        if self.use_double:
            self.rkpm.double()

        # eigenvectors
        self.evecs = nn.Parameter(torch.zeros(num_nodes, num_handles, dtype=torch.float64 if use_double else torch.float32))
        self.evecs.requires_grad = False


    def init(self, pts, yms, prs, rhos, appx_vol, cut=None, force_reinit=False):
        r"""Initializes the RKPM nodes and eigenvectors from input point cloud data.

        Selects RKPM kernel nodes via Farthest Point Sampling, computes node
        radii from nearest-neighbor distances, and performs a generalized
        eigenanalysis on the mass and stiffness matrices to determine the
        deformation modes.

        Geometry-independent computations (FPS, kD-trees, radii) are cached
        across calls so that only the eigenanalysis is recomputed when ``cut.alpha``
        changes.

        Args:
            pts (torch.Tensor): Input points of shape :math:`(N, 3)`.
            yms (torch.Tensor): Young's moduli of shape :math:`(N,)`.
            prs (torch.Tensor): Poisson's ratios of shape :math:`(N,)`.
            rhos (torch.Tensor): Densities of shape :math:`(N,)`.
            appx_vol (float): Approximate volume of the object.
            cut (DiscCut, optional): Cut surface for 4D lifting. ``None`` for 3D mode.
            force_reinit (bool): If True, recompute FPS/kD-trees even if cached.
        """
        # currently assume all integration samples have equal volume weights = appx_vol / num_points, weights cancelled out

        device = pts.device

        # --- Cache geometry-independent computations (FPS, kD-trees, radii) ---
        # These depend only on pts which is static across alpha updates.
        if hasattr(self, '_cached_nodes') and not force_reinit:
            nodes = self._cached_nodes
            node_radius = self._cached_node_radius
            sample_indices = self._cached_sample_indices
        else:
            # Use Farthest Point Sampling to determine nodes
            if pts.shape[0] < self.num_nodes:
                print("WARNING: num_nodes is less than the number of points. Using all points as nodes.")
                node_indices = torch.arange(pts.shape[0], device=device)
            else:
                node_indices = farthest_point_sampling(pts[None], self.num_nodes).squeeze(0)

            nodes = pts[node_indices]
            nodes_np = nodes.cpu().numpy()
            nodes_kdtree = cKDTree(nodes_np)

            pts_np = pts.cpu().numpy()
            pts_kdtree = cKDTree(pts_np)

            # Compute node radii (always in 3D, based on 3D positions)
            dists, _ = nodes_kdtree.query(nodes_np, k=self.radius_init_kNN + 1, workers=-1)
            node_radius = torch.tensor(dists[:, -1] * self.radius_scale, device=nodes.device, dtype=nodes.dtype)

            if isinstance(self.radius_min, float):
                node_radius = node_radius.clamp(min=self.radius_min)
            elif isinstance(self.radius_min, str):
                assert self.radius_min[-1] == "x", "radius_min must end with 'x'"
                min_dist_factor = float(self.radius_min[:-1])
                pts_dists, _ = pts_kdtree.query(pts_np, k=2, workers=-1)
                radius_min = pts_dists[:, -1].mean() * min_dist_factor
                node_radius = node_radius.clamp(min=radius_min)
            else:
                raise ValueError("Unknown radius_min")

            # Farthest Point Sampling to determine integration points
            if self.num_points is None:
                sample_indices = torch.arange(pts.shape[0], device=device)
            else:
                sample_indices = farthest_point_sampling(pts[None], self.num_points).squeeze(0)

            self._cached_nodes = nodes
            self._cached_node_radius = node_radius
            self._cached_sample_indices = sample_indices

        # --- 3D vs 4D mode ---
        # Use 4D lifting only when alpha > 0 — at alpha=0, H=0 everywhere, making
        # the 5th polynomial column identically zero (singular moment matrix).
        # moment_eps in phi/grad_phi helps at small alpha but not at exactly alpha=0.
        use_4d = cut is not None and cut.alpha > 0
        rkpm_dim_changed = False
        if use_4d:
            if self.rkpm.num_dims != 4:
                self.rkpm = RKPM(self.num_nodes, self.kernel_type, num_dims=4,
                                 cutoff_factor=self.cutoff_factor)
                if self.use_double:
                    self.rkpm.double()
                self.rkpm.to(device)
                rkpm_dim_changed = True
            H_nodes = compute_winding_number(nodes.float(), cut).to(dtype=nodes.dtype)  # (N, 1)
            nodes_for_kernels = torch.cat([nodes, H_nodes], dim=-1)  # (N, 4)
        else:
            if self.rkpm.num_dims != 3:
                self.rkpm = RKPM(self.num_nodes, self.kernel_type, num_dims=3,
                                 cutoff_factor=self.cutoff_factor)
                if self.use_double:
                    self.rkpm.double()
                self.rkpm.to(device)
                rkpm_dim_changed = True
            nodes_for_kernels = nodes  # 3D

        # Reset warm-start eigenvectors when the RKPM dimensionality changes or on force_reinit
        if force_reinit or rkpm_dim_changed:
            self._prev_evecs_full = None

        self.rkpm.set_kernels(nodes_for_kernels, node_radius)

        x = pts[sample_indices]
        yms_x = yms[sample_indices]
        prs_x = prs[sample_indices]

        if self.use_double:
            x = x.to(dtype=torch.float64)
            yms_x = yms_x.to(dtype=torch.float64)
            prs_x = prs_x.to(dtype=torch.float64)

        if use_4d:
            H_x = compute_winding_number(x.float(), cut).to(dtype=x.dtype)  # (n_sample, 1)
            x_for_eigen = torch.cat([x, H_x], dim=-1)  # (n_sample, 4)
        else:
            x_for_eigen = x

        # --- Eigenanalysis with LOBPCG warm start ---
        M = self.get_mass_matrix(x_for_eigen)
        H_mat = self.get_hessian_matrix(x_for_eigen, yms_x, prs_x)

        # Pass previous eigenvectors as initial guess (warm start); None on first call or after dim change
        X_init = getattr(self, '_prev_evecs_full', None)
        # add one for the zero eigenvalue
        evals, evecs = torch.lobpcg(A=H_mat, B=M, k=(self.num_handles + 1), largest=False, X=X_init)
        self._prev_evecs_full = evecs.detach().clone()
        self.evecs.data.copy_(evecs[:, 1:])

    def get_mass_matrix(self, x):
        r"""Computes the RKPM mass matrix.

        The mass matrix is :math:`M = \Phi^T \Phi`, where :math:`\Phi` is the
        matrix of RKPM kernel evaluations at the sample points.

        Args:
            x (torch.Tensor): Sample points of shape :math:`(n, D)` where D is 3 or 4.

        Returns:
            torch.Tensor: Mass matrix of shape :math:`(N, N)`.
        """
        phi_x = self.rkpm.phi(x)
        M = phi_x.T @ phi_x
        return M

    def get_hessian_matrix(self, x, yms, prs, reparameterize_lame=True):
        r"""Computes the RKPM stiffness (Hessian) matrix.

        Args:
            x (torch.Tensor): Sample points of shape :math:`(n, D)`.
            yms (torch.Tensor): Young's moduli at sample points of shape :math:`(n,)`.
            prs (torch.Tensor): Poisson's ratios at sample points of shape :math:`(n,)`.
            reparameterize_lame (bool, optional): If True, scales by
                :math:`\lambda + 4\mu`. If False, scales by :math:`\lambda + 3\mu`.
                Defaults to True.

        Returns:
            torch.Tensor: Stiffness matrix of shape :math:`(N, N)`.
        """
        grad_phi_x = self.rkpm.grad_phi(x)  # (n, N, D=3)
        n, N, D = grad_phi_x.shape
        J = grad_phi_x.permute(0, 2, 1).reshape(n * D, N)
        # assume the neohookean energy in wp.fem
        # scaling factor (\lambda + 4\mu)
        mus, lams = to_lame(yms, prs)
        if reparameterize_lame:
            per_point_coeff = lams + 4 * mus
        else:
            per_point_coeff = lams + 3 * mus
        per_dim_coeff = torch.kron(per_point_coeff.flatten(), torch.ones(D, device=x.device, dtype=x.dtype))
        H = J.T @ (per_dim_coeff[:, None] * J)
        return H

    def forward(self, x, cut=None):
        r"""Evaluates RKPM skinning weights at query points.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, 3)`.
            cut (DiscCut, optional): Cut surface for 4D lifting. ``None`` for 3D.

        Returns:
            torch.Tensor: Skinning weights of shape :math:`(n, C)`.
        """
        dtype = x.dtype
        x_d = x.to(dtype=torch.float64) if self.use_double else x
        # Lift to 4D only when the RKPM was initialised in 4D (alpha > 0 at last refit)
        if cut is not None and self.rkpm.num_dims == 4:
            H = compute_winding_number(x.float(), cut).to(dtype=x_d.dtype)  # (n, 1)
            x_d = torch.cat([x_d, H], dim=-1)  # (n, 4)
        return self.rkpm(x_d, self.evecs).to(dtype=dtype)

    def grad(self, x):
        r"""Computes spatial gradients of RKPM skinning weights at query points.

        Note: This computes the gradient in whatever dimensionality the RKPM is
        currently operating in (3D or 4D). For 3D Jacobians with correct chain
        rule through H(x) in 4D mode, use :meth:`grad_3d`.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, D)`.

        Returns:
            torch.Tensor: Skinning weight gradients of shape :math:`(n, D, C)`.
        """
        dtype = x.dtype
        if self.use_double:
            x = x.to(dtype=torch.float64)
        grad_phi = self.rkpm.grad_phi(x)  # (n, N, D)
        grad = torch.einsum("nNd,Nc->ndc", grad_phi, self.evecs)  # (n, D, C)
        return grad.to(dtype=dtype).permute(0, 2, 1)  # (n, C, D)

    def grad_3d(self, x, cut=None):
        r"""Analytical Jacobian of RKPM weights w.r.t. 3D positions.

        In 4D mode (alpha > 0), applies the chain rule through H(x) to correctly
        account for dH/dx. Much faster than vmap+jacrev through the full forward pass.

        Args:
            x (torch.Tensor): ``(n, 3)`` query points.
            cut (DiscCut, optional): Cut object (needed for chain rule in 4D mode);
                pass ``None`` in 3D mode.

        Returns:
            torch.Tensor: ``(n, 3, num_handles)`` Jacobian of RKPM weights w.r.t.
            3D positions.
        """
        dtype = x.dtype
        x_d = x.to(dtype=torch.float64) if self.use_double else x

        if cut is not None and self.rkpm.num_dims == 4:
            # Compute dH/dx via autograd through compute_winding_number.
            # torch.enable_grad() ensures this works even inside a torch.no_grad() context.
            with torch.enable_grad():
                x_3d = x.float().detach().requires_grad_(True)
                H = compute_winding_number(x_3d, cut)           # (n, 1)
                dH_dx = torch.autograd.grad(H.sum(), x_3d)[0]  # (n, 3)

            H_d = H.detach().to(dtype=x_d.dtype)
            x_4d = torch.cat([x_d, H_d], dim=-1)           # (n, 4)

            grad_phi = self.rkpm.grad_phi(x_4d)             # (n, N, 4)
            # grad_W_4d[n, d, c] = sum_N grad_phi[n,N,d] * evecs[N,c]
            grad_W_4d = torch.einsum("nNd,Nc->ndc", grad_phi, self.evecs)  # (n, 4, num_handles)
            grad_W_3d = grad_W_4d[:, :3, :]                 # (n, 3, num_handles)
            grad_W_H  = grad_W_4d[:, 3:, :]                 # (n, 1, num_handles)
            # Chain rule: dW/dx_3D += dW/dH * dH/dx_3D
            # grad_W_H: (n,1,C), dH_dx: (n,3) → dH_dx[...,None]: (n,3,1) → broadcast (n,3,C)
            grad_W_3d = grad_W_3d + grad_W_H * dH_dx.to(dtype=x_d.dtype).unsqueeze(-1)
        else:
            grad_phi = self.rkpm.grad_phi(x_d)              # (n, N, 3)
            grad_W_3d = torch.einsum("nNd,Nc->ndc", grad_phi, self.evecs)  # (n, 3, num_handles)

        return grad_W_3d.to(dtype=dtype).permute(0, 2, 1)  # (n, num_handles, 3)


class RKPM(nn.Module):
    r"""Reproducing Kernel Particle Method (RKPM) function module.

    Implements first-order RKPM functions with consistency correction,
    allowing kernel-based interpolation over scattered point data. The corrected
    kernel :math:`\phi_I(x)` satisfies polynomial completeness up to the
    specified polynomial degree.

    Supports configurable dimensionality (3D or 4D for cut-aware mode) and
    optional kernel truncation via ``cutoff_factor``.

    Args:
        num_nodes (int): Number of kernel nodes.
        kernel_type (str, optional): Type of base kernel function. Currently
            only ``"gaussian"`` is supported. Defaults to ``"gaussian"``.
        polynomial_degree (int, optional): Degree of polynomial basis used for
            consistency correction. Currently only degree 1 is supported.
            Defaults to 1.
        num_dims (int, optional): Spatial dimensionality. 3 for standard mode,
            4 for cut-aware 4D lifting. Defaults to 3.
        cutoff_factor (float, optional): Truncation radius in units of node
            radius. Kernels are zeroed beyond ``cutoff_factor * radius``.
            ``None`` disables truncation. Defaults to ``None``.
    """

    def __init__(
        self,
        num_nodes: int,
        kernel_type: str = "gaussian",
        polynomial_degree: int = 1,
        num_dims: int = 3,
        cutoff_factor: Optional[float] = None,
    ):
        super(RKPM, self).__init__()

        self.num_nodes = num_nodes
        self.kernel_type = kernel_type

        self.num_dims = num_dims
        self.polynomial_degree = polynomial_degree
        self.cutoff_factor = cutoff_factor

        self.initialized = False
        self.register_parameter("nodes", torch.nn.Parameter(torch.zeros(self.num_nodes, self.num_dims)))
        self.nodes.requires_grad = False
        self.register_parameter("radius", torch.nn.Parameter(torch.ones(self.num_nodes)))
        self.radius.requires_grad = False

    def set_kernels(self, nodes, radius):
        r"""Sets the node positions and radii for the RKPM kernels.

        Args:
            nodes (torch.Tensor): Node positions of shape :math:`(N, D)`.
            radius (torch.Tensor): Per-node kernel radii of shape :math:`(N,)`.
        """
        if self.nodes.shape != nodes.shape:
            self.register_parameter("nodes", torch.nn.Parameter(nodes.to(dtype=self.nodes.dtype, device=self.nodes.device)))
            self.nodes.requires_grad = False
            self.register_parameter("radius", torch.nn.Parameter(radius.to(dtype=self.radius.dtype, device=self.radius.device)))
            self.radius.requires_grad = False
            self.num_nodes = nodes.shape[0]
        else:
            self.nodes.data.copy_(nodes)
            self.radius.data.copy_(radius)
        self.initialized = True

    def func_r(self, r):
        r"""Evaluates the uncorrected radial basis kernel as a function of distance.

        Args:
            r (torch.Tensor): Distances from query points to nodes of shape
                :math:`(n, N)`.

        Returns:
            torch.Tensor: Kernel values of shape :math:`(n, N)`.
        """
        # uncorrected RBF kernel, as a function of radius
        if self.kernel_type == "gaussian":
            out = torch.exp(-(r / self.radius) ** 2)
            if self.cutoff_factor is not None:
                out = out * (r <= self.cutoff_factor * self.radius).to(out.dtype)
            return out
        else:
            raise ValueError("Unknown kernel type")

    def func_x(self, x):
        r"""Evaluates the uncorrected radial basis kernel at input locations.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, D)`.

        Returns:
            torch.Tensor: Kernel values of shape :math:`(n, N)`.
        """
        # uncorrected RBF kernel, as a function of input location
        r = torch.linalg.norm(x[:, None, :] - self.nodes[None, :, :], dim=-1)
        return self.func_r(r)

    def dfunc_dx(self, x):
        r"""Computes the spatial gradient of the uncorrected radial basis kernel.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, D)`.

        Returns:
            torch.Tensor: Kernel gradients of shape :math:`(n, N, D)`.
        """
        # derivative of uncorrected RBF kernel, as a function of input location
        displacement = x[:, None, :] - self.nodes[None, :, :]
        func_x = self.func_x(x)
        return func_x[..., None] * (-2 / self.radius[None, :, None] ** 2) * displacement

    def polynomial(self, x):
        r"""Evaluates the polynomial basis at input locations.

        For degree 1, returns :math:`[1, x_1, \ldots, x_D]` for each point.

        Args:
            x (torch.Tensor): Input points of shape :math:`(n, D)`.

        Returns:
            torch.Tensor: Polynomial basis values of shape :math:`(n, P)`.
        """
        if self.polynomial_degree == 1:
            return torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x], dim=-1)
        else:
            raise ValueError("Unknown polynomial degree")

    @property
    def P(self):
        r"""Number of polynomial basis terms.

        For degree 1 in D dimensions, returns D + 1.

        Returns:
            int: Number of polynomial terms.
        """
        # number of polynomial terms
        if self.polynomial_degree == 1:
            # [1, x1, ..., xD] for the first order
            return (1 + self.num_dims)
        else:
            raise ValueError("Unknown polynomial degree")

    def grad_polynomial(self, x):
        r"""Computes spatial gradients of the polynomial basis.

        For degree 1, the gradient of :math:`[1, x_1, \ldots, x_D]` with respect to
        position is :math:`[0, I_{D \times D}]`.

        Args:
            x (torch.Tensor): Input points of shape :math:`(n, D)`.

        Returns:
            torch.Tensor: Polynomial basis gradients of shape :math:`(n, P, D)`.
        """
        if self.polynomial_degree == 1:
            # Px = [1, x], so dPx/dx = [0, I]
            dPx_dx = torch.zeros(x.shape[0], self.P, self.num_dims, device=x.device, dtype=x.dtype)
            dPx_dx[:, 1:, :] = torch.eye(self.num_dims, device=x.device, dtype=x.dtype)[None, :, :]
        else:
            raise ValueError("Unknown polynomial degree")
        return dPx_dx

    def phi(self, x, moment_eps=1e-6):
        r"""Evaluates the corrected RKPM basis functions at query points.

        The corrected basis satisfies polynomial completeness, ensuring that
        the interpolation can exactly reproduce polynomials up to the specified
        degree.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, D)`.
            moment_eps (float): Regularization added to the moment matrix for
                numerical stability. Defaults to 1e-6.

        Returns:
            torch.Tensor: Corrected kernel weights of shape :math:`(n, N)`.
        """
        # corrected RKPM kernel function, weights for each node value
        func_x = self.func_x(x)
        Pn = self.polynomial(self.nodes)  # (N, P)
        Pn_PnT = torch.einsum("Ni,Nj->Nij", Pn, Pn)  # (N, P, P)
        Mx = torch.einsum("nN,Nij->nij", func_x, Pn_PnT)  # (n, P, P)
        Mx = Mx + moment_eps * torch.eye(self.P, device=x.device, dtype=x.dtype)
        Px = self.polynomial(x)  # (n, P)
        Cx = torch.linalg.solve(Mx, Px)  # (n, P)
        phi_x = (Cx @ Pn.T) * func_x  # (n, P) @ (P, N) -> (n, N)
        return phi_x

    def grad_phi(self, x, moment_eps=1e-6):
        r"""Computes spatial gradients of the corrected RKPM basis functions.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, D)`.
            moment_eps (float): Regularization added to the moment matrix for
                numerical stability. Defaults to 1e-6.

        Returns:
            torch.Tensor: Gradients of corrected kernel weights of shape
            :math:`(n, N, D)`.
        """
        dfunc_dx = self.dfunc_dx(x)  # (n, N, D)
        func_x = self.func_x(x)  # (n, N)

        Pn = self.polynomial(self.nodes)  # (N, P)
        Pn_PnT = torch.einsum("Ni,Nj->Nij", Pn, Pn)  # (N, P, P)
        Mx = torch.einsum("nN,Nij->nij", func_x, Pn_PnT)  # (n, P, P)
        Mx = Mx + moment_eps * torch.eye(self.P, device=x.device, dtype=x.dtype)

        Px = self.polynomial(x)  # (n, P)
        Cx = torch.linalg.solve(Mx, Px)  # (n, P)

        # phi = (Cx @ Pn.T) * func_x
        # dW/dx = d((Cx @ Pn.T) * func_x)/dx
        #       = (Cx @ Pn.T) * dfunc_dx + (dCx/dx @ Pn.T) * func_x

        # First term: (Cx @ Pn.T) * dfunc_dx
        CxPnT = Cx @ Pn.T  # (n, N)
        term1 = CxPnT[..., None] * dfunc_dx  # (n, N, D)

        # Second term: (dCx/dx @ Pn.T) * func_x
        dPx_dx = self.grad_polynomial(x)  # (n, P, D)

        # Compute dMx/dx: shape (n, P, P, D)
        dMx_dx = torch.einsum("nNd,Nij->nijd", dfunc_dx, Pn_PnT)  # (n, P, P, D)

        # dCx/dx = Mx^{-1} @ (dPx/dx - dMx/dx @ Cx)
        # Shape: (n, P, D) = (n, P, P) @ ((n, P, D) - (n, P, D))
        dMx_Cx = torch.einsum("nijd,nj->nid", dMx_dx, Cx)  # (n, P, D)
        dCx_dx = torch.linalg.solve(Mx, dPx_dx - dMx_Cx)  # (n, P, D)

        term2 = torch.einsum("npd,Np->nNd", dCx_dx, Pn) * func_x[..., None]  # (n, N, D)

        grad_phi_x = term1 + term2
        return grad_phi_x

    def forward(self, x, c):
        r"""Evaluates the RKPM interpolation of node values at query points.

        Args:
            x (torch.Tensor): Query points of shape :math:`(n, D)`.
            c (torch.Tensor): Node values of shape :math:`(N, C)`.

        Returns:
            torch.Tensor: Interpolated values of shape :math:`(n, C)`.
        """
        if not self.initialized:
            raise ValueError("RKPM not initialized.")
        return self.phi(x) @ c
