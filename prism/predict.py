"""
Predict geodesic distances on a new shape.

    # mesh input: sample points (+ normals) on the surface
    python -m prism.predict --ckpt best.pt --input bunny.obj --n_points 4096 --out dist.npz
    # point cloud input (.npy / .xyz / .ply), normals used if present, otherwise estimated
    python -m prism.predict --ckpt best.pt --input cloud.npy --sources 0 10 20 --out dist.npz

Output npz: points (N,3) in input units, dist (len(sources) or N, N) in input units.
"""
import argparse

import numpy as np
import torch

from .evaluate import load_model


def load_input(path, n_points, seed=0):
    """Returns points (N,3), normals (N,3) or None."""
    if path.endswith(".npy"):
        x = np.load(path)
        return x[:, :3], (x[:, 3:6] if x.shape[1] >= 6 else None)
    if path.endswith(".xyz") or path.endswith(".txt"):
        x = np.loadtxt(path)
        return x[:, :3], (x[:, 3:6] if x.shape[1] >= 6 else None)
    import trimesh

    g = trimesh.load(path, process=False)
    if isinstance(g, trimesh.PointCloud):  # normals are estimated for point-cloud files
        return np.asarray(g.vertices), None
    m = g if isinstance(g, trimesh.Trimesh) else g.to_geometry()
    pts, fid = trimesh.sample.sample_surface(m, n_points, seed=seed)  # area-uniform, exact count
    return np.asarray(pts), np.asarray(m.face_normals[fid])


def estimate_normals(p, k=16):
    """PCA normals, oriented away from the centroid (training normals point outward)."""
    from scipy.spatial import cKDTree

    _, idx = cKDTree(p).query(p, k)
    nb = p[idx] - p[idx].mean(1, keepdims=True)
    n = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))[1][:, :, 0]
    flip = np.einsum("ni,ni->n", n, p - p.mean(0)) < 0
    n[flip] *= -1
    return n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", required=True, help="mesh (.obj/.ply/...) or point cloud (.npy/.xyz/.ply)")
    ap.add_argument("--n_points", type=int, default=4096, help="points sampled from a mesh input")
    ap.add_argument("--sources", type=int, nargs="*", default=None, help="source indices; default: all (N x N)")
    ap.add_argument("--out", default="geodesic.npz")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model(a.ckpt, dev)
    pts, nrm = load_input(a.input, a.n_points)
    if model.use_normals and nrm is None:
        print("no normals in input; estimating with PCA (outward orientation is heuristic)")
        nrm = estimate_normals(pts)
    nrm = None if nrm is None else nrm / np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-12)

    # same normalization as training: bbox center at origin, max radius 1
    center = (pts.max(0) + pts.min(0)) / 2
    scale = np.linalg.norm(pts - center, axis=1).max()
    P = torch.tensor((pts - center) / scale, dtype=torch.float32, device=dev)
    Nr = None if nrm is None else torch.tensor(nrm, dtype=torch.float32, device=dev)
    rows = None if a.sources is None else torch.tensor(a.sources, device=dev)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        D = model.distance_matrix(P, Nr, rows=rows).float().cpu().numpy() * scale
    np.savez(a.out, points=pts.astype(np.float32), dist=D.astype(np.float32),
             sources=np.arange(len(pts)) if a.sources is None else np.array(a.sources))
    print(f"saved {a.out}: points {pts.shape}, dist {D.shape}")


if __name__ == "__main__":
    main()
