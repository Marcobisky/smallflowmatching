"""MLP velocity field for 2D flow matching."""

from __future__ import annotations

from itertools import pairwise

import torch
import torch.nn as nn


class MLPVelocity(nn.Module):
    """v_theta(x_t, t) for 2D flow matching.

    Time is embedded as 2D [sin(t), cos(t)], concatenated with x_t (2D),
    giving a 4D input to the MLP.
    """

    def __init__(self, hidden_dims: tuple[int, ...] = (16, 128, 128, 128, 128, 16)) -> None:
        super().__init__()
        input_dim = 4  # 2 (x_t) + 2 (time embed)
        layers: list[nn.Module] = []
        for d_in, d_out in pairwise((input_dim,) + hidden_dims):
            layers.extend([nn.Linear(d_in, d_out), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1) if t.dim() == 1 else t
        t_emb = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
        return self.net(torch.cat([x_t, t_emb], dim=1))


class CondEmbedderLabel(nn.Module):
    """Embed discrete class labels into continuous vectors.

    Labels in [0, num_classes) are valid class labels.
    Label = num_classes is the unconditional (null) label.
    During training, labels are randomly replaced with the null label
    at rate ``dropout_prob`` to enable classifier-free guidance.
    """

    def __init__(self, cond_dim: int, num_classes: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(num_classes + 1, cond_dim)
        self.null_cond = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        if self.training:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_ids, self.null_cond, labels)
        return self.embeddings(labels)


class ConditionalMLPVelocity(nn.Module):
    """Conditional v_theta(x_t, t, cond) for 2D flow matching.

    Extends MLPVelocity by concatenating condition embeddings to the input.
    Use ``from_pretrained`` to initialise from a pre-trained unconditional
    backbone: the first linear layer is expanded by ``cond_dim`` columns
    with zero weights so the model initially behaves identically to the
    unconditional version.
    """

    def __init__(
        self,
        cond_dim: int = 4,
        num_classes: int = 7,
        dropout_prob: float = 0.1,
        hidden_dims: tuple[int, ...] = (16, 128, 128, 128, 128, 16),
    ) -> None:
        super().__init__()
        input_dim = 4 + cond_dim  # 2 (x_t) + 2 (time embed) + cond_dim
        layers: list[nn.Module] = []
        for d_in, d_out in pairwise((input_dim,) + hidden_dims):
            layers.extend([nn.Linear(d_in, d_out), nn.GELU()])
        layers.append(nn.Linear(hidden_dims[-1], 2))
        self.net = nn.Sequential(*layers)
        self.cond_embed = CondEmbedderLabel(cond_dim, num_classes, dropout_prob)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
    ) -> torch.Tensor:
        t = t.view(-1, 1) if t.dim() == 1 else t
        t_emb = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
        cond_emb = self.cond_embed(cond)
        return self.net(torch.cat([x_t, t_emb, cond_emb], dim=1))

    @classmethod
    def from_pretrained(
        cls,
        backbone: MLPVelocity,
        cond_dim: int = 4,
        num_classes: int = 7,
        dropout_prob: float = 0.1,
    ) -> ConditionalMLPVelocity:
        """Build from a pre-trained unconditional backbone with zero-init expansion."""
        linears = [m for m in backbone.net if isinstance(m, nn.Linear)]
        hidden_dims = tuple(layer.out_features for layer in linears[:-1])

        model = cls(
            cond_dim=cond_dim, num_classes=num_classes,
            dropout_prob=dropout_prob, hidden_dims=hidden_dims,
        )

        src_sd = backbone.state_dict()
        dst_sd = model.state_dict()

        for key in dst_sd:
            if key.startswith("cond_embed.") or key not in src_sd:
                continue
            src, dst = src_sd[key], dst_sd[key]
            if src.shape == dst.shape:
                dst_sd[key] = src.clone()
            elif src.dim() == 2 and dst.dim() == 2:
                dst_sd[key] = torch.zeros_like(dst)
                dst_sd[key][:, : src.shape[1]] = src

        model.load_state_dict(dst_sd)
        return model



class Reward2D(nn.Module):
    """Differentiable scalar reward r(x) that depends only on position.

    Built from a weighted sum of anisotropic Gaussian bumps, a smooth
    ``(1-x^2)_+(1-y^2)_+`` boundary envelope, and a rational map
    ``z/(z+\\alpha)`` to keep values in a smooth [0, 1)-like range—no plain
    sin/cos ridges. This yields several local maxima in ``[-1,1]^2``.

    The second argument ``y`` is kept for backward compatibility with training
    code that passes class indices; it is ignored.
    """

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        # Unused but kept so call sites can still pass num_classes=...
        self.num_classes = num_classes

        # Nine bumps: distinct centers, anisotropic scales, positive weights.
        centers = torch.tensor(
            [
                [-0.64, 0.58],
                [0.58, 0.46],
                [-0.28, -0.64],
                [0.27, -0.43],
                [-0.74, -0.06],
                [0.20, 0.70],
                [-0.40, 0.34],
                [0.61, -0.42],
                [-0.10, -0.15],
            ],
            dtype=torch.float32,
        )
        # Narrower Gaussians so peaks overlap less in the interior.
        sigma_x = torch.tensor(
            [0.10, 0.09, 0.10, 0.05, 0.09, 0.12, 0.08, 0.09, 0.12], dtype=torch.float32
        )
        sigma_y = torch.tensor(
            [0.10, 0.10, 0.09, 0.04, 0.10, 0.11, 0.10, 0.09, 0.10], dtype=torch.float32
        )
        weights = torch.tensor(
            [1.00, 0.80, 0.86, 0.74, 0.70, 0.92, 0.66, 1.00, 0.78], dtype=torch.float32
        )
        self.register_buffer("centers", centers)
        self.register_buffer("sigma_x", sigma_x)
        self.register_buffer("sigma_y", sigma_y)
        self.register_buffer("weights", weights)
        self.alpha = 0.28

    @property
    def device(self) -> torch.device:
        return self.centers.device

    def _envelope(self, x: torch.Tensor) -> torch.Tensor:
        """Smoothly vanish toward the boundary of [-1, 1]^2."""
        ex = torch.clamp(1.0 - x[:, 0] ** 2, 0.0, 1.0)
        ey = torch.clamp(1.0 - x[:, 1] ** 2, 0.0, 1.0)
        return ex * ey

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, 2] -> [B] in roughly [0, 1]."""
        xc = x[:, 0:1]
        yc = x[:, 1:2]
        cx = self.centers[:, 0].unsqueeze(0)
        cy = self.centers[:, 1].unsqueeze(0)
        sx = self.sigma_x.unsqueeze(0)
        sy = self.sigma_y.unsqueeze(0)
        d2 = (xc - cx) ** 2 / (2.0 * sx**2) + (yc - cy) ** 2 / (2.0 * sy**2)
        z = (self.weights.unsqueeze(0) * torch.exp(-d2)).sum(dim=1)
        env = self._envelope(x)
        # Interior: rational saturation; edges: small floor so ∂r still exists.
        r = (z / (z + self.alpha)) * env + 0.02 * (1.0 - env)
        return r


class PINN(nn.Module):
    """Dual-head PINN for the HJB value function.

    Random Fourier features lift the 2-D spatial input into a
    high-dimensional representation that breaks the spectral bias of
    plain MLPs, enabling the boundary regression w(x,1,c) ≈ ∇r(x)
    for a smooth position-only reward r(x) (maximize reward at t = 1).

    Two linear heads read a shared SiLU backbone:
      * **V-head** (scalar) — used for ∂V/∂t (finite differences).
      * **w-head** (2-D)   — direct approximation of ∇_x V.

    ``forward`` returns w (the residual vector field for alignment).
    """

    def __init__(
        self,
        cond_dim: int = 4,
        num_classes: int = 7,
        hidden_dims: tuple[int, ...] = (128, 256, 128),
        n_fourier: int = 64,
        sigma_fourier: float = 5.0,
    ) -> None:
        super().__init__()
        B = torch.randn(n_fourier, 2) * sigma_fourier
        self.register_buffer("B", B)
        input_dim = 2 * n_fourier + 2 + cond_dim
        layers: list[nn.Module] = []
        for d_in, d_out in pairwise((input_dim,) + hidden_dims):
            layers.extend([nn.Linear(d_in, d_out), nn.SiLU()])
        self.backbone = nn.Sequential(*layers)
        self.head_V = nn.Linear(hidden_dims[-1], 1)
        self.head_w = nn.Linear(hidden_dims[-1], 2)
        # Indices 0 .. num_classes-1: class labels; index num_classes: unconditional (no label).
        self.cond_embed = nn.Embedding(num_classes + 1, cond_dim)

    def _encode(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        xB = x @ self.B.T
        x_ff = torch.cat([torch.sin(xB), torch.cos(xB)], dim=1)
        t = t.view(-1, 1) if t.dim() == 1 else t
        t_emb = torch.cat([torch.sin(t), torch.cos(t)], dim=1)
        c_emb = self.cond_embed(c)
        return self.backbone(torch.cat([x_ff, t_emb, c_emb], dim=1))

    def forward_all(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (V [B,1], w [B,2])."""
        h = self._encode(x, t, c)
        return self.head_V(h), self.head_w(h)

    def potential(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Scalar V(x, t | c) -> [B, 1]."""
        return self.head_V(self._encode(x, t, c))

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """w ≈ ∇_x V -> [B, 2].  The residual vector field for alignment."""
        return self.head_w(self._encode(x, t, c))
