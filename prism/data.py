"""
Dataset over shapes produced by prism.preprocess:
    <root>/data/<name>.npz        points (V,3), normals (V,3), faces, src (S,)
    <root>/data/<name>_dist.npy   (S,V) float16 exact geodesics from each source vertex
    <root>/{train,test}.txt

Each item: subsample n points (always containing n_src sources), random rotation,
random (source, target) pairs with GT geodesic distance.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q.astype(np.float32)


class GeodesicDataset(Dataset):
    """
    Returns
        points  (n, 3)   normals (n, 3)
        pairs   (Q, 2)   indices into the n points
        dist    (Q,)     GT geodesic distance
    """

    def __init__(self, root, split="train", n_points=4096, n_src=32, n_pairs=4096, augment=None):
        names = open(os.path.join(root, f"{split}.txt")).read().split()
        assert names, f"empty split {split} in {root}"
        self.files = [os.path.join(root, "data", n) for n in names]
        self.n_points, self.n_src, self.n_pairs = n_points, n_src, n_pairs
        self.augment = split == "train" if augment is None else augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # eval uses a fixed per-shape seed so the metric is comparable across epochs
        rng = np.random.default_rng() if self.augment else np.random.default_rng(idx)
        d = np.load(self.files[idx] + ".npz")
        P, Nrm, src = d["points"], d["normals"], d["src"]
        Dmm = np.load(self.files[idx] + "_dist.npy", mmap_mode="r")  # (S, V)
        N, S = len(P), len(src)

        # choose sources, then fill the rest of the subset with non-source points
        s_sel = np.sort(rng.choice(S, min(self.n_src, S), replace=False))
        n_pts = min(self.n_points, N)
        others = np.setdiff1d(np.arange(N), src[s_sel])
        rest = rng.choice(others, n_pts - len(s_sel), replace=False)
        sub = np.concatenate([src[s_sel], rest])  # first len(s_sel) entries are sources
        D = np.asarray(Dmm[s_sel], dtype=np.float32)[:, sub]  # (n_src, n_pts), only needed rows read

        # pairs: (source k, random target j in subset)
        k = rng.integers(0, len(s_sel), self.n_pairs)
        j = rng.integers(0, n_pts, self.n_pairs)
        dist = D[k, j]
        pairs = np.stack([k, j], 1)
        if self.augment:  # model must be symmetric; randomly swap order
            swap = rng.random(self.n_pairs) < 0.5
            pairs[swap] = pairs[swap][:, ::-1]

        pts, nrm = P[sub], Nrm[sub]
        if self.augment:
            R = random_rotation(rng)
            pts, nrm = pts @ R.T, nrm @ R.T

        return {
            "points": torch.from_numpy(np.ascontiguousarray(pts)),
            "normals": torch.from_numpy(np.ascontiguousarray(nrm)),
            "pairs": torch.from_numpy(np.ascontiguousarray(pairs)).long(),
            "dist": torch.from_numpy(dist.astype(np.float32)),
        }
