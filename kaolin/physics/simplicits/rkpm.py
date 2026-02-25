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
from scipy.spatial import cKDTree

__all__ = [
    'SimplicitsRKPM',
]

class SimplicitsRKPM(nn.Module):
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
    ):
        super(SimplicitsRKPM, self).__init__()
        self.num_points = num_points
        self.num_handles = num_handles
        self.num_nodes = num_nodes

        self.kernel_type = kernel_type
        self.radius_scale = radius_scale
        self.radius_init_kNN = radius_init_kNN
        self.radius_min = radius_min

        self.rkpm = RKPM(num_nodes, kernel_type)

        self.use_double = use_double  # force double precision for all computations involving RKPM
        if self.use_double:
            self.rkpm.double()

        # eigenvectors
        self.evecs = nn.Parameter(torch.zeros(num_nodes, num_handles, dtype=torch.float64 if use_double else torch.float32))
        self.evecs.requires_grad = False


    def init(self, pts, yms, prs, rhos, appx_vol):
        # currently assume all integration samples have equal volume weights = appx_vol / num_points, weights cancelled out
        
        # Use Farthest Point Sampling to determine nodes        
        device = pts.device
        if pts.shape[0] < self.num_nodes:
            print("WARNING: num_nodes is less than the number of points. Using all points as nodes.")
            node_indices = torch.arange(pts.shape[0], device)
        else:
            node_indices = farthest_point_sampling(pts[None], self.num_nodes).squeeze(0)
        
        nodes = pts[node_indices]
        nodes_np = nodes.cpu().numpy()
        nodes_kdtree = cKDTree(nodes_np)

        pts_np = pts.cpu().numpy()
        pts_kdtree = cKDTree(pts_np)

        # Compute node radii
        dists, _ = nodes_kdtree.query(nodes.cpu().numpy(), k=self.radius_init_kNN + 1, workers=-1)
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
        
        self.rkpm.set_kernels(nodes, node_radius)

        # Farthest Point Sampling to determine integration points
        if self.num_points is None:
            sample_indices = torch.arange(pts.shape[0], device=pts.device)
        else:
            sample_indices = farthest_point_sampling(pts[None], self.num_points).squeeze(0)
        x = pts[sample_indices]
        yms_x = yms[sample_indices]
        prs_x = prs[sample_indices]

        if self.use_double:
            x = x.to(dtype=torch.float64)
            yms_x = yms_x.to(dtype=torch.float64)
            prs_x = prs_x.to(dtype=torch.float64)

        # Perform eigenanalysis
        M = self.get_mass_matrix(x)
        H = self.get_hessian_matrix(x, yms_x, prs_x)

        # add one for the zero eigenvalue
        evals, evecs = torch.lobpcg(A=H, B=M, k=(self.num_handles + 1), largest=False, X=None)
        self.evecs.data.copy_(evecs[:, 1:])

    def get_mass_matrix(self, x):
        phi_x = self.rkpm.phi(x)
        M = phi_x.T @ phi_x
        return M
    
    def get_hessian_matrix(self, x, yms, prs, reparameterize_lame=True):
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

    def forward(self, x):
        dtype = x.dtype
        if self.use_double:
            x = x.to(dtype=torch.float64)
        return self.rkpm(x, self.evecs).to(dtype=dtype)
    
    def grad(self, x):
        dtype = x.dtype
        if self.use_double:
            x = x.to(dtype=torch.float64)
        grad_phi = self.rkpm.grad_phi(x)  # (n, N, D)
        grad = torch.einsum("nNd,Nc->ncd", grad_phi, self.evecs)  # (n, D, C)
        return grad.to(dtype=dtype)


class RKPM(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        kernel_type: str = "gaussian",
        polynomial_degree: int = 1,
    ):
        super(RKPM, self).__init__()

        self.num_nodes = num_nodes
        self.kernel_type = kernel_type

        self.num_dims = 3
        self.polynomial_degree = polynomial_degree
        
        self.initialized = False
        self.register_parameter("nodes", torch.nn.Parameter(torch.zeros(self.num_nodes, self.num_dims)))
        self.nodes.requires_grad = False
        self.register_parameter("radius", torch.nn.Parameter(torch.ones(self.num_nodes)))
        self.radius.requires_grad = False

    def set_kernels(self, nodes, radius):
        self.nodes.data.copy_(nodes)
        self.radius.data.copy_(radius)
        self.initialized = True

    def func_r(self, r):
        # uncorrected RBF kernel, as a function of radius
        if self.kernel_type == "gaussian":
            return torch.exp(-(r / self.radius) ** 2)
        else:
            raise ValueError("Unknown kernel type")
    
    def func_x(self, x):
        # uncorrected RBF kernel, as a function of input location
        r = torch.linalg.norm(x[:, None, :] - self.nodes[None, :, :], dim=-1)
        return self.func_r(r)
    
    def dfunc_dx(self, x):
        # derivative of uncorrected RBF kernel, as a function of input location
        displacement = x[:, None, :] - self.nodes[None, :, :]
        func_x = self.func_x(x)
        return func_x[..., None] * (-2 / self.radius[None, :, None] ** 2) * displacement

    def polynomial(self, x):
        if self.polynomial_degree == 1:
            return torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x], dim=-1)
        else:
            raise ValueError("Unknown polynomial degree")

    @property
    def P(self):
        # number of polynomial terms
        if self.polynomial_degree == 1:
            # [1, x, y, z] for the first order
            return (1 + self.num_dims)
        else:
            raise ValueError("Unknown polynomial degree")
    
    def grad_polynomial(self, x):
        if self.polynomial_degree == 1:
            # Px = [1, x], so dPx/dx = [0, I]
            dPx_dx = torch.zeros(x.shape[0], self.P, self.num_dims, device=x.device, dtype=x.dtype)
            dPx_dx[:, 1:, :] = torch.eye(self.num_dims, device=x.device, dtype=x.dtype)[None, :, :]
        else:
            raise ValueError("Unknown polynomial degree")
        return dPx_dx

    def phi(self, x):
        # corrected RKPM kernel function, weights for each node value
        func_x = self.func_x(x)
        Pn = self.polynomial(self.nodes)  # (N, P)
        Pn_PnT = torch.einsum("Ni,Nj->Nij", Pn, Pn)  # (N, P, P)
        Mx = torch.einsum("nN,Nij->nij", func_x, Pn_PnT)  # (n, P, P)
        Px = self.polynomial(x)  # (n, P)
        Cx = torch.linalg.solve(Mx, Px)  # (n, P)
        phi_x = (Cx @ Pn.T) * func_x  # (n, P) @ (P, N) -> (n, N)
        return phi_x

    def grad_phi(self, x):
        dfunc_dx = self.dfunc_dx(x)  # (n, N, D)
        func_x = self.func_x(x)  # (n, N)
        
        Pn = self.polynomial(self.nodes)  # (N, P)
        Pn_PnT = torch.einsum("Ni,Nj->Nij", Pn, Pn)  # (N, P, P)
        Mx = torch.einsum("nN,Nij->nij", func_x, Pn_PnT)  # (n, P, P)
        
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
        if not self.initialized:
            raise ValueError("RKPM not initialized.")
        return self.phi(x) @ c
