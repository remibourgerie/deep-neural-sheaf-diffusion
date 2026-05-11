"""
Deep Neural Sheaf Diffusion (DNSD) — Models

Implements DNSD and NSD (baseline) sheaf diffusion architectures for node
classification, plus standard GNN baselines (GAT, MPNN, MLP).

Reference:
    "Deep Neural Sheaf Diffusion" (ICML 2025 workshop)

Usage:
    from models import create_model, train_model

    model = create_model('dnsd_diag', input_dim=2, hidden_dim=18,
                         output_dim=3, num_stalks=3, num_layers=8,
                         adj=True, odd=True, gate=False)
    model, preds, history = train_model(model, data, epochs=500)
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import degree, softmax
from typing import Dict, Literal, Optional, Tuple

try:
    from torch_householder import torch_householder_orgqr
    HOUSEHOLDER_AVAILABLE = True
except ImportError:
    HOUSEHOLDER_AVAILABLE = False


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ============================================================
# Orthogonal map parameterization
# ============================================================

class Orthogonal(nn.Module):
    """
    Orthogonal matrix parameterization.

    Methods:
    - 'cayley': Cayley transform (default, best gradients, pure PyTorch)
    - 'matrix_exp': Matrix exponential
    - 'householder': Householder reflectors (requires torch-householder)
    - 'householder_native': Householder reflectors, pure PyTorch
    - 'qr': QR decomposition
    """
    def __init__(self, d, method='cayley'):
        super().__init__()
        self.d = d
        if method == 'householder':
            if HOUSEHOLDER_AVAILABLE:
                self.method = 'householder'
            else:
                import warnings
                warnings.warn(
                    "torch-householder not available, falling back to 'householder_native'. "
                    "Install with: pip install torch-householder",
                    UserWarning
                )
                self.method = 'householder_native'
        elif method in ('cayley', 'matrix_exp', 'householder_native', 'qr'):
            self.method = method
        else:
            raise ValueError(f"Unknown method '{method}'.")

    def forward(self, params):
        E = params.size(0)
        d = self.d
        A = params.view(E, d, d)

        if self.method == 'householder':
            Q = torch_householder_orgqr(A)
        elif self.method == 'householder_native':
            I = torch.eye(d, device=params.device).unsqueeze(0)
            Q = I.expand(E, -1, -1).clone()
            for i in range(d):
                v = A[:, i, :]
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
                H = I - 2 * torch.einsum('bi,bj->bij', v, v)
                Q = Q @ H
        elif self.method == 'matrix_exp':
            S = A - A.transpose(-1, -2)
            Q = torch.matrix_exp(S)
        elif self.method == 'cayley':
            S = A - A.transpose(-1, -2)
            I = torch.eye(d, device=S.device).unsqueeze(0)
            Q = torch.linalg.solve(I + S, I - S)
        elif self.method == 'qr':
            A = A + 1e-6 * torch.eye(d, device=A.device).unsqueeze(0)
            Q, _ = torch.linalg.qr(A)

        return Q


# ============================================================
# Restriction map builders
# ============================================================

class RestrictionBuilder(nn.Module):
    def forward(self, x, edge_index):
        raise NotImplementedError

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stalk_dim = getattr(self, 'd', None)
        if stalk_dim is not None:
            return f'{self.__class__.__name__}(stalk_dim={stalk_dim}, params={params:,})'
        return f'{self.__class__.__name__}(params={params:,})'


class FullRestrictionBuilder(RestrictionBuilder):
    def __init__(self, hidden_dim, num_stalks, init_scale=1.0):
        super().__init__()
        self.d = num_stalks
        self.lin = nn.Linear(2 * hidden_dim, num_stalks * num_stalks)
        if init_scale != 1.0:
            with torch.no_grad():
                self.lin.weight.data *= init_scale
                if self.lin.bias is not None:
                    self.lin.bias.data *= init_scale

    def forward(self, x, edge_index):
        u, v = edge_index
        x_cat = torch.cat([x[u], x[v]], dim=-1)
        F = self.lin(x_cat).view(-1, self.d, self.d)
        return torch.tanh(F)


class DiagonalRestrictionBuilder(RestrictionBuilder):
    def __init__(self, hidden_dim, num_stalks, init_scale=1.0):
        super().__init__()
        self.d = num_stalks
        self.lin = nn.Linear(2 * hidden_dim, num_stalks)
        if init_scale != 1.0:
            with torch.no_grad():
                self.lin.weight.data *= init_scale
                if self.lin.bias is not None:
                    self.lin.bias.data *= init_scale

    def forward(self, x, edge_index):
        u, v = edge_index
        x_cat = torch.cat([x[u], x[v]], dim=-1)
        f = torch.tanh(self.lin(x_cat))
        return torch.diag_embed(f)


class OrthogonalRestrictionBuilder(RestrictionBuilder):
    def __init__(self, hidden_dim, num_stalks, init_scale=1.0, ortho_method='qr'):
        super().__init__()
        self.d = num_stalks
        self.num_params = num_stalks * num_stalks
        self.lin = nn.Linear(2 * hidden_dim, self.num_params)
        self.ortho = Orthogonal(num_stalks, method=ortho_method)
        if init_scale != 1.0:
            with torch.no_grad():
                self.lin.weight.data *= init_scale
                if self.lin.bias is not None:
                    self.lin.bias.data *= init_scale

    def forward(self, x, edge_index):
        u, v = edge_index
        x_cat = torch.cat([x[u], x[v]], dim=-1)
        p = self.lin(x_cat)
        return self.ortho(p)


# ============================================================
# NSD v1 convolution layer (Bodnar et al., 2022)
# ============================================================

class NSDConv(MessagePassing):
    """
    NSD convolution layer. Computes one step of sheaf diffusion:
        x' = (1 + eps) * x - ReLU(W1 @ L_F(x) @ W2)
    where L_F is the normalized sheaf Laplacian.
    """
    def __init__(self, hidden_dim: int, num_stalks: int, use_diff_mlp: bool = False):
        super().__init__(aggr='add', flow='target_to_source')
        self.hidden_dim = hidden_dim
        self.num_stalks = num_stalks
        self.stalk_dim = hidden_dim // num_stalks
        self.use_diff_mlp = use_diff_mlp

        self.W1 = nn.Parameter(torch.eye(num_stalks) + 0.01 * torch.randn(num_stalks, num_stalks))
        self.W2 = nn.Parameter(torch.eye(self.stalk_dim) + 0.01 * torch.randn(self.stalk_dim, self.stalk_dim))
        self.eps = nn.Parameter(torch.zeros(num_stalks))

        if use_diff_mlp:
            self.diff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(hidden_dim, hidden_dim),
            )

    def forward(self, x, edge_index, F_source, F_target):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, F_source=F_source, F_target=F_target, norm=norm)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        s, h = self.num_stalks, self.stalk_dim
        x_stalk = x.view(N, s, h)
        lap_stalk = aggr_out.view(N, s, h)
        transformed = torch.einsum('ij,njk,kl->nil', self.W1, lap_stalk, self.W2)
        transformed = F.relu(transformed)
        eps = (1 + self.eps).view(1, s, 1)
        return (eps * x_stalk - transformed).view(N, -1)

    def message(self, x_i, x_j, F_source, F_target, norm):
        E = x_i.size(0)
        s, h = self.num_stalks, self.stalk_dim
        x_source = x_i.view(E, s, h)
        x_target = x_j.view(E, s, h)
        Fx_source = torch.bmm(F_source, x_source)
        Fx_target = torch.bmm(F_target, x_target)
        diff = Fx_source - Fx_target
        if self.use_diff_mlp:
            diff = self.diff_mlp(diff.view(E, -1)).view(E, s, h)
        msg = torch.bmm(F_source.transpose(1, 2), diff)
        return (msg * norm.view(-1, 1, 1)).view(E, -1)


# ============================================================
# DNSD convolution layer
# ============================================================

class DNSDConv(MessagePassing):
    """
    Deep Neural Sheaf Diffusion convolution layer.

    Flags (Table 4 in paper):
      directional: use adjacency operator (target-only, F_v x_v) instead of Laplacian
      odd:         use tanh activation instead of ReLU
      use_stalk_gate: per-stalk GRU-style gate on diffusion updates
    """
    def __init__(
        self,
        hidden_dim: int,
        num_stalks: int,
        use_diff_mlp: bool = False,
        stalk_dropout: float = 0.0,
        directional: bool = False,
        odd: bool = False,
        use_stalk_gate: bool = False,
    ):
        super().__init__(aggr='add', flow='target_to_source')
        self.hidden_dim = hidden_dim
        self.num_stalks = num_stalks
        self.stalk_dim = hidden_dim // num_stalks
        self.use_diff_mlp = use_diff_mlp
        self.stalk_dropout = stalk_dropout
        self.directional = directional
        self.odd = odd
        self.use_stalk_gate = use_stalk_gate

        self.W1 = nn.Parameter(torch.eye(num_stalks) + 0.01 * torch.randn(num_stalks, num_stalks))
        self.W2 = nn.Parameter(torch.eye(self.stalk_dim) + 0.01 * torch.randn(self.stalk_dim, self.stalk_dim))
        self.eps = nn.Parameter(torch.zeros(num_stalks))

        if use_diff_mlp:
            self.diff_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Linear(hidden_dim, hidden_dim),
            )

        if use_stalk_gate:
            self.gate_lin = nn.Linear(2 * self.stalk_dim, 1, bias=True)
            nn.init.zeros_(self.gate_lin.weight)
            nn.init.zeros_(self.gate_lin.bias)

    def forward(self, x, edge_index, F_source, F_target):
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.clamp(min=1).pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, F_source=F_source, F_target=F_target, norm=norm)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        s, h = self.num_stalks, self.stalk_dim
        x_stalk = x.view(N, s, h)
        lap_stalk = aggr_out.view(N, s, h)

        transformed = torch.einsum('ij,njk,kl->nil', self.W1, lap_stalk, self.W2)

        if self.odd:
            transformed = F.tanh(transformed)
        else:
            transformed = F.relu(transformed)

        if self.use_stalk_gate:
            gate_input = torch.cat([x_stalk, lap_stalk], dim=-1)  # [N, S, 2h]
            gate = torch.sigmoid(
                self.gate_lin(gate_input.view(-1, 2 * h))
            ).view(N, s, 1)
            transformed = gate * transformed
            self._last_gates = gate.detach()

        self._last_transformed = transformed.view(N, -1).detach()

        if self.stalk_dropout > 0.0 and self.training:
            mask = torch.ones(N, s, 1, device=transformed.device)
            mask = F.dropout(mask, p=self.stalk_dropout, training=True)
            transformed = transformed * mask

        eps = (1 + self.eps).view(1, s, 1)
        return (eps * x_stalk - transformed).view(N, -1)

    def message(self, x_i, x_j, F_source, F_target, norm):
        E = x_i.size(0)
        s, h = self.num_stalks, self.stalk_dim
        x_source = x_i.view(E, s, h)
        x_target = x_j.view(E, s, h)

        Fx_target = torch.bmm(F_target, x_target)
        if self.directional:
            diff = Fx_target
        else:
            Fx_source = torch.bmm(F_source, x_source)
            diff = Fx_source - Fx_target

        if self.use_diff_mlp:
            diff = self.diff_mlp(diff.view(E, -1)).view(E, s, h)

        msg = torch.bmm(F_source.transpose(1, 2), diff)
        return (msg * norm.view(-1, 1, 1)).view(E, -1)


# ============================================================
# NSD v1 model (Bodnar et al., 2022) — baseline
# ============================================================

class NeuralSheafDiffusion(nn.Module):
    """
    Neural Sheaf Diffusion (Bodnar et al., NeurIPS 2022).

    Args:
        input_dim: Input feature dimension
        hidden_dim: Total hidden size (stalk_dim = hidden_dim // num_stalks)
        output_dim: Number of classes
        num_layers: Number of sheaf diffusion layers
        num_stalks: Stalk dimension
        map_type: 'full', 'diagonal', or 'orthogonal'
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_stalks: int = 3,
        map_type: Literal["full", "diagonal", "orthogonal"] = "full",
        use_diff_mlp: bool = False,
        freeze_restriction_maps: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stalks = num_stalks
        self.stalk_dim = hidden_dim // num_stalks
        self.num_layers = num_layers

        self.lin_in = nn.Linear(input_dim, hidden_dim)

        builder_map = {
            "full": FullRestrictionBuilder,
            "diagonal": DiagonalRestrictionBuilder,
            "orthogonal": OrthogonalRestrictionBuilder,
        }
        if map_type not in builder_map:
            raise ValueError(f"Unknown map_type: {map_type}")
        builder_cls = builder_map[map_type]

        self.builders = nn.ModuleList([
            builder_cls(hidden_dim, num_stalks) for _ in range(num_layers)
        ])
        if freeze_restriction_maps:
            for builder in self.builders:
                for param in builder.parameters():
                    param.requires_grad = False

        self.convs = nn.ModuleList([
            NSDConv(hidden_dim, num_stalks, use_diff_mlp=use_diff_mlp)
            for _ in range(num_layers)
        ])
        self.lin_out = nn.Linear(hidden_dim, output_dim)
        self.sheaf_logger = None
        self.current_epoch = 0

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x)
        for i in range(self.num_layers):
            F = self.builders[i](x, edge_index)
            x = self.convs[i](x, edge_index, F, F)
        return self.lin_out(x)

    def forward_with_intermediates(self, x: torch.Tensor, edge_index: torch.Tensor):
        intermediates = {}
        N = x.size(0)
        x = self.lin_in(x)
        intermediates['layer_0'] = x.view(N, self.num_stalks, self.stalk_dim).detach().clone()
        for i in range(self.num_layers):
            F = self.builders[i](x, edge_index)
            x = self.convs[i](x, edge_index, F, F)
            intermediates[f'layer_{i+1}'] = x.view(N, self.num_stalks, self.stalk_dim).detach().clone()
        return self.lin_out(x), intermediates


# ============================================================
# DNSD model (this work)
# ============================================================

class DeepNeuralSheafDiffusion(nn.Module):
    """
    Deep Neural Sheaf Diffusion (DNSD).

    Key differences from NSD:
      - Separate source/target restriction map builders per layer
      - adjacency operator (directional=True) instead of Laplacian
      - LayerNorm after each diffusion step
      - Odd (tanh) activation and per-stalk gating

    Args:
        input_dim: Input feature dimension
        hidden_dim: Total hidden size (stalk_dim = hidden_dim // num_stalks)
        output_dim: Number of classes
        num_layers: Number of sheaf diffusion layers
        num_stalks: Stalk dimension
        map_type: 'full', 'diagonal', or 'orthogonal'
        layer_norm_mode: 'embedding' (default), 'stalk', or None
        directional: Use adjacency operator (adj flag in paper, default False)
        odd: Use tanh activation (odd flag in paper, default False)
        use_stalk_gate: Per-stalk sigmoid gate (gate flag in paper, default False)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        num_stalks: int = 3,
        map_type: Literal["full", "diagonal", "orthogonal"] = "full",
        use_diff_mlp: bool = False,
        freeze_restriction_maps: bool = False,
        layer_norm_mode: str = "embedding",
        ortho_method: str = "qr",
        stalk_dropout: float = 0.0,
        directional: bool = False,
        odd: bool = False,
        use_stalk_gate: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_stalks = num_stalks
        self.stalk_dim = hidden_dim // num_stalks
        self.num_layers = num_layers
        self.stalk_dropout = stalk_dropout
        self.directional = directional
        self.odd = odd
        self.use_stalk_gate = use_stalk_gate

        self.lin_in = nn.Linear(input_dim, hidden_dim)

        builder_map = {
            "full": FullRestrictionBuilder,
            "diagonal": DiagonalRestrictionBuilder,
            "orthogonal": OrthogonalRestrictionBuilder,
        }
        if map_type not in builder_map:
            raise ValueError(f"Unknown map_type: {map_type}")
        builder_cls = builder_map[map_type]

        init_scale = 1.0 / math.sqrt(2)
        builder_kwargs = {'init_scale': init_scale}
        if map_type == "orthogonal":
            builder_kwargs['ortho_method'] = ortho_method

        self.source_builders = nn.ModuleList([
            builder_cls(hidden_dim, num_stalks, **builder_kwargs) for _ in range(num_layers)
        ])
        self.target_builders = nn.ModuleList([
            builder_cls(hidden_dim, num_stalks, **builder_kwargs) for _ in range(num_layers)
        ])

        if freeze_restriction_maps:
            for builder in list(self.source_builders) + list(self.target_builders):
                for param in builder.parameters():
                    param.requires_grad = False

        self.convs = nn.ModuleList([
            DNSDConv(hidden_dim, num_stalks, use_diff_mlp=use_diff_mlp,
                     stalk_dropout=stalk_dropout, directional=directional,
                     odd=odd, use_stalk_gate=use_stalk_gate)
            for _ in range(num_layers)
        ])

        self.layer_norm_mode = layer_norm_mode
        if layer_norm_mode == "embedding":
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        elif layer_norm_mode == "stalk":
            self.norms = nn.ModuleList([nn.LayerNorm(self.stalk_dim) for _ in range(num_layers)])
        elif layer_norm_mode in (None, False):
            self.norms = None
        else:
            raise ValueError(f"layer_norm_mode must be 'embedding', 'stalk', or None")

        self.lin_out = nn.Linear(hidden_dim, output_dim)
        self.sheaf_logger = None
        self.current_epoch = 0

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.lin_in(x)
        for i in range(self.num_layers):
            F_source = self.source_builders[i](x, edge_index)
            F_target = self.target_builders[i](x, edge_index)
            x = self.convs[i](x, edge_index, F_source, F_target)
            if self.norms is not None:
                if self.layer_norm_mode == "embedding":
                    x = self.norms[i](x)
                elif self.layer_norm_mode == "stalk":
                    N = x.size(0)
                    x = self.norms[i](x.view(N, self.num_stalks, self.stalk_dim)).view(N, -1)
        return self.lin_out(x)

    def forward_with_intermediates(self, x: torch.Tensor, edge_index: torch.Tensor):
        intermediates = {}
        N = x.size(0)
        x = self.lin_in(x)
        intermediates['layer_0'] = x.view(N, self.num_stalks, self.stalk_dim).detach().clone()
        for i in range(self.num_layers):
            F_source = self.source_builders[i](x, edge_index)
            F_target = self.target_builders[i](x, edge_index)
            x = self.convs[i](x, edge_index, F_source, F_target)
            if self.norms is not None:
                if self.layer_norm_mode == "embedding":
                    x = self.norms[i](x)
                elif self.layer_norm_mode == "stalk":
                    x = self.norms[i](x.view(N, self.num_stalks, self.stalk_dim)).view(N, -1)
            intermediates[f'layer_{i+1}'] = x.view(N, self.num_stalks, self.stalk_dim).detach().clone()
        return self.lin_out(x), intermediates


# ============================================================
# Convenience wrappers
# ============================================================

class NSD_Full(NeuralSheafDiffusion):
    def __init__(self, *args, **kwargs):
        kwargs.pop('map_type', None)
        super().__init__(*args, map_type="full", **kwargs)

class NSD_Diagonal(NeuralSheafDiffusion):
    def __init__(self, *args, **kwargs):
        kwargs.pop('map_type', None)
        super().__init__(*args, map_type="diagonal", **kwargs)

class NSD_Orthogonal(NeuralSheafDiffusion):
    def __init__(self, *args, **kwargs):
        kwargs.pop('map_type', None)
        super().__init__(*args, map_type="orthogonal", **kwargs)

class DNSD_Full(DeepNeuralSheafDiffusion):
    def __init__(self, *args, **kwargs):
        kwargs.pop('map_type', None)
        super().__init__(*args, map_type="full", **kwargs)

class DNSD_Diagonal(DeepNeuralSheafDiffusion):
    def __init__(self, *args, **kwargs):
        kwargs.pop('map_type', None)
        super().__init__(*args, map_type="diagonal", **kwargs)

class DNSD_Orthogonal(DeepNeuralSheafDiffusion):
    def __init__(self, *args, **kwargs):
        kwargs.pop('map_type', None)
        super().__init__(*args, map_type="orthogonal", **kwargs)


# ============================================================
# GNN baselines
# ============================================================

class MPNNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class MPNN(nn.Module):
    """Message Passing Neural Network baseline."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        in_dim = input_dim
        for _ in range(num_layers):
            self.convs.append(MPNNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.lin_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)


class GAT(nn.Module):
    """Graph Attention Network baseline."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, heads=3, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers

        # Snap heads down to largest value ≤ requested that divides hidden_dim
        while heads > 1 and hidden_dim % heads != 0:
            heads -= 1

        if num_layers == 1:
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.output_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.convs.append(GATConv(input_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))
            self.output_proj = None

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        if self.output_proj is not None:
            x = self.output_proj(F.elu(x))
        return x


class MLP(nn.Module):
    """MLP baseline (ignores graph structure)."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        if num_layers > 1:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.layers[0] = nn.Linear(input_dim, output_dim)

    def forward(self, x, edge_index=None):
        for layer in self.layers[:-1]:
            x = F.dropout(F.relu(layer(x)), p=self.dropout, training=self.training)
        return self.layers[-1](x)


# ============================================================
# Model factory
# ============================================================

_MODEL_MAP = {
    'nsd_full':     NSD_Full,
    'nsd_diag':     NSD_Diagonal,
    'nsd_ortho':    NSD_Orthogonal,
    'dnsd_full':    DNSD_Full,
    'dnsd_diag':    DNSD_Diagonal,
    'dnsd_ortho':   DNSD_Orthogonal,
    'gat':          GAT,
    'mpnn':         MPNN,
    'mlp':          MLP,
}


def create_model(name: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    DNSD-specific flags (passed as kwargs):
      directional / adj: bool  — use adjacency operator (default False)
      odd: bool                — use tanh activation (default False)
      use_stalk_gate / gate: bool — per-stalk gate (default False)

    Example:
        model = create_model('dnsd_diag', input_dim=2, hidden_dim=18,
                             output_dim=3, num_stalks=3, num_layers=8,
                             directional=True, odd=True)
    """
    if name not in _MODEL_MAP:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_MAP)}")

    # Accept 'adj' as alias for 'directional', 'gate' as alias for 'use_stalk_gate'
    if 'adj' in kwargs:
        kwargs.setdefault('directional', kwargs.pop('adj'))
    if 'gate' in kwargs:
        kwargs.setdefault('use_stalk_gate', kwargs.pop('gate'))

    # Non-sheaf baselines don't use num_stalks; strip sheaf-only keys
    _SHEAF_ONLY = ('num_stalks', 'directional', 'odd', 'use_stalk_gate')
    _BASELINE_MODELS = ('gat', 'mpnn', 'mlp')
    if name in _BASELINE_MODELS:
        for k in _SHEAF_ONLY:
            kwargs.pop(k, None)

    return _MODEL_MAP[name](**kwargs)


# ============================================================
# Training
# ============================================================

def train_model(
    model: nn.Module,
    data: Data,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    epochs: int = 500,
    seed: int = 42,
    verbose: bool = False,
    task_type: str = "classification",
    test_data: Data = None,
    use_scheduler: bool = True,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 100,
) -> Tuple[nn.Module, torch.Tensor, Dict[str, list]]:
    """
    Train any model for node classification.

    Args:
        model: Any PyTorch model with forward(x, edge_index) -> logits
        data: PyG Data with x, edge_index, y, train_mask, (val_mask), test_mask
        test_data: Optional separate test graph (all nodes used for eval)
        early_stopping_patience: Patience on val_acc; set None to disable

    Returns:
        model, predictions, history
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )

    has_val = hasattr(data, 'val_mask') and data.val_mask is not None and data.val_mask.any()
    history = {'train_loss': [], 'train_acc': [], 'test_acc': [], 'lr': []}
    if has_val:
        history['val_acc'] = []

    use_early_stopping = early_stopping_patience is not None and has_val
    best_val_metric = float('-inf')
    es_patience_counter = 0
    best_model_state = None
    stopped_epoch = None

    training_start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_train = model(data.x.to(device), data.edge_index.to(device))
            y_train = data.y.to(device)

            if test_data is not None:
                out_test = model(test_data.x.to(device), test_data.edge_index.to(device))
                y_test = test_data.y.to(device)
                test_mask = torch.ones(test_data.num_nodes, dtype=torch.bool, device=device)
            else:
                out_test = out_train
                y_test = y_train
                test_mask = data.test_mask

            pred_train = out_train.argmax(dim=1)
            pred_test  = out_test.argmax(dim=1)
            train_metric = (pred_train[data.train_mask] == y_train[data.train_mask]).float().mean().item()
            test_metric  = (pred_test[test_mask] == y_test[test_mask]).float().mean().item()

            val_metric = None
            if has_val:
                val_metric = (pred_train[data.val_mask] == y_train[data.val_mask]).float().mean().item()
                history['val_acc'].append(val_metric)

        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_metric)
        history['test_acc'].append(test_metric)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if scheduler is not None:
            sched_input = (1.0 - val_metric) if (has_val and val_metric is not None) else loss.item()
            scheduler.step(sched_input)

        if use_early_stopping and val_metric is not None:
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                es_patience_counter = 0
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                es_patience_counter += 1
            if es_patience_counter >= early_stopping_patience:
                stopped_epoch = epoch + 1
                print(f"  Early stopping at epoch {stopped_epoch} (best val: {best_val_metric:.4f})")
                break

        if verbose and (epoch + 1) % 20 == 0:
            val_str = f", val={val_metric:.4f}" if val_metric is not None else ""
            print(f"Epoch {epoch+1:4d}: loss={loss.item():.4f}, "
                  f"train={train_metric:.4f}, test={test_metric:.4f}{val_str}")

    history['training_time_s'] = time.time() - training_start
    history['stopped_epoch'] = stopped_epoch
    history['early_stopped'] = stopped_epoch is not None

    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    eval_graph = test_data if test_data is not None else data
    model.eval()
    with torch.no_grad():
        out = model(eval_graph.x.to(device), eval_graph.edge_index.to(device))
        predictions = out.argmax(dim=1).cpu()

    return model, predictions, history
