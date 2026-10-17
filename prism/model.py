"""
Anchor-based pure-attention transformer for pairwise geodesic distance on point clouds.

    points (B,N,3[+3 normals])
        │  Fourier pos-enc + MLP
        ▼
    point tokens (B,N,C) ──────────────┐
        │  FPS -> K anchors             │
        ▼                               │
    anchor queries (B,K,C)              │
        │  cross-attn (anchors <- points) x n_enc_cross
        ▼                               │
        │  self-attn among anchors     x n_self
        ▼                               │
    anchor tokens (B,K,C)               │
        │  cross-attn (points <- anchors) x n_dec_cross  (decode per-point features)
        ▼
    point features f (B,N,C)
        │  pair (i,j): symmetric combine -> 3-layer MLP
        ▼
    d(i,j) = ||p_i - p_j|| + softplus(MLP)        (geodesic >= euclidean)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------------
class FourierEmbed(nn.Module):
    def __init__(self, in_dim, n_freq, dim, max_freq=64.0):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0, math.log2(max_freq), n_freq) * math.pi
        self.register_buffer("freqs", freqs, persistent=False)
        feat = in_dim * (2 * n_freq + 1)
        self.mlp = nn.Sequential(nn.Linear(feat, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, x):
        xf = x[..., None] * self.freqs  # (..., in, F)
        h = torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], -1)
        return self.mlp(h)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x, ctx):
        B, Nq, C = x.shape
        q = self.q(x).view(B, Nq, self.h, -1).transpose(1, 2)
        k, v = self.kv(ctx).view(B, ctx.shape[1], 2, self.h, -1).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).reshape(B, Nq, C))


class Block(nn.Module):
    """Pre-norm transformer block. ctx=None -> self-attention, otherwise cross-attention."""

    def __init__(self, dim, heads, mlp_ratio=4.0, cross=False):
        super().__init__()
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim) if cross else None
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        hid = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(nn.Linear(dim, hid), nn.GELU(), nn.Linear(hid, dim))

    def forward(self, x, ctx=None):
        q = self.nq(x)
        kv = q if ctx is None else self.nkv(ctx)
        x = x + self.attn(q, kv)
        return x + self.ffn(self.n2(x))


@torch.no_grad()
def farthest_point_sample(xyz, k, random_start=True):
    """xyz (B,N,3) -> idx (B,k)"""
    B, N, _ = xyz.shape
    idx = torch.zeros(B, k, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device)
    far = (torch.randint(0, N, (B,), device=xyz.device) if random_start
           else torch.zeros(B, dtype=torch.long, device=xyz.device))
    bidx = torch.arange(B, device=xyz.device)
    for i in range(k):
        idx[:, i] = far
        c = xyz[bidx, far][:, None]  # (B,1,3)
        dist = torch.minimum(dist, ((xyz - c) ** 2).sum(-1))
        far = dist.argmax(-1)
    return idx


def gather(x, idx):
    """x (B,N,C), idx (B,...) -> (B,...,C)"""
    B = x.shape[0]
    flat = idx.reshape(B, -1)
    out = torch.gather(x, 1, flat[..., None].expand(-1, -1, x.shape[-1]))
    return out.view(*idx.shape, x.shape[-1])


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
# Sizes aligned to ~half of PRISM (PTv3 enc+dec) S/B/L = 31M / 124M / 494M, same feature dim k.
PRESETS = {
    "S": dict(dim=256, heads=4, n_anchors=256, n_enc_cross=1, n_self=16, n_dec_cross=2),   # 15.3M
    "B": dict(dim=512, heads=8, n_anchors=512, n_enc_cross=1, n_self=16, n_dec_cross=2),   # 61.2M
    "L": dict(dim=768, heads=12, n_anchors=1024, n_enc_cross=2, n_self=30, n_dec_cross=3),  # 250.9M
}


class GeodesicTransformer(nn.Module):
    def __init__(
        self,
        dim=384,
        heads=8,
        n_anchors=512,
        n_enc_cross=1,
        n_self=8,
        n_dec_cross=2,
        n_freq=10,
        use_normals=True,
        euclid_prior=True,
    ):
        super().__init__()
        self.n_anchors = n_anchors
        self.use_normals = use_normals
        self.euclid_prior = euclid_prior
        in_dim = 6 if use_normals else 3

        self.embed = FourierEmbed(in_dim, n_freq, dim)
        self.anchor_tag = nn.Parameter(torch.zeros(1, 1, dim))  # distinguishes anchor queries
        self.enc_cross = nn.ModuleList([Block(dim, heads, cross=True) for _ in range(n_enc_cross)])
        self.self_blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_self)])
        self.dec_cross = nn.ModuleList([Block(dim, heads, cross=True) for _ in range(n_dec_cross)])
        self.out_norm = nn.LayerNorm(dim)

        # 3-layer MLP on a symmetric pair representation
        self.pair_mlp = nn.Sequential(
            nn.Linear(3 * dim + 1, dim), nn.GELU(),
            nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        nn.init.trunc_normal_(self.anchor_tag, std=0.02)

    # -- encoding -----------------------------------------------------------------------------
    def encode(self, points, normals=None, anchor_idx=None):
        """
        points (B,N,3), normals (B,N,3) -> per-point features (B,N,C), anchor tokens (B,K,C)
        """
        x = torch.cat([points, normals], -1) if self.use_normals else points
        tok = self.embed(x)  # (B,N,C)

        if anchor_idx is None:
            k = min(self.n_anchors, points.shape[1])
            anchor_idx = farthest_point_sample(points.float(), k, random_start=self.training)
        a = gather(tok, anchor_idx) + self.anchor_tag  # anchor queries

        for blk in self.enc_cross:  # anchors gather information from all points
            a = blk(a, tok)
        for blk in self.self_blocks:  # global reasoning among anchors
            a = blk(a)
        f = tok
        for blk in self.dec_cross:  # every point reads from the anchors
            f = blk(f, a)
        return self.out_norm(f), a

    # -- pair head ----------------------------------------------------------------------------
    def pair_distance(self, feats, points, pairs):
        """feats (B,N,C), points (B,N,3), pairs (B,Q,2) -> dist (B,Q)"""
        fi, fj = gather(feats, pairs[..., 0]), gather(feats, pairs[..., 1])
        pi, pj = gather(points, pairs[..., 0]), gather(points, pairs[..., 1])
        eu = (pi - pj).norm(dim=-1, keepdim=True)
        h = torch.cat([fi + fj, fi * fj, (fi - fj).abs(), eu], -1)  # symmetric in (i,j)
        out = self.pair_mlp(h).squeeze(-1)
        if self.euclid_prior:
            return eu.squeeze(-1) + F.softplus(out)
        return F.softplus(out)

    def forward(self, points, normals, pairs, anchor_idx=None):
        feats, _ = self.encode(points, normals, anchor_idx)
        return self.pair_distance(feats, points, pairs)

    # -- inference helper ---------------------------------------------------------------------
    @torch.no_grad()
    def distance_matrix(self, points, normals=None, rows=None, chunk=65536):
        """
        Dense geodesic matrix for a single cloud. points (N,3) -> (len(rows) or N, N).
        Points should be normalized the same way as training (bbox center 0, diag 1).
        """
        P = points[None]
        Nr = normals[None] if normals is not None else None
        feats, _ = self.encode(P, Nr)
        N = points.shape[0]
        rows = torch.arange(N, device=points.device) if rows is None else rows
        ii, jj = torch.meshgrid(rows, torch.arange(N, device=points.device), indexing="ij")
        pairs = torch.stack([ii.reshape(-1), jj.reshape(-1)], -1)
        out = torch.cat([self.pair_distance(feats, P, pairs[None, s:s + chunk])[0]
                         for s in range(0, len(pairs), chunk)])
        D = out.view(len(rows), N)
        D[torch.arange(len(rows)), rows] = 0.0
        return D
