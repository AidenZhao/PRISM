"""
Evaluate a checkpoint on the test split: MRE and L1, overall and per ShapeNet category.

    python -m prism.evaluate --ckpt runs/prism_B/best.pt --data DATA_ROOT

MRE = mean |pred - gt| / (gt + eps) over pairs with gt > min_gt; L1 = mean |pred - gt|
(distances in unit-sphere units). 2000 query pairs per shape by default, fixed per-shape seed.
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import GeodesicDataset
from .model import GeodesicTransformer


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GeodesicTransformer(**ck["args"]["model_cfg"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--n_pairs", type=int, default=2000, help="query pairs per shape")
    ap.add_argument("--eps", type=float, default=0.0, help="MRE denominator is gt + eps")
    ap.add_argument("--min_gt", type=float, default=1e-4, help="pairs with gt below this are skipped in MRE")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="optional json with the results")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(a.ckpt, dev)
    ds = GeodesicDataset(a.data, a.split, n_points=a.n_points, n_pairs=a.n_pairs, augment=False)
    names = [os.path.basename(f) for f in ds.files]
    dl = DataLoader(ds, a.bs, num_workers=a.workers)
    amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda")

    rows = []  # per shape: rel_sum, n_rel, l1_sum, n_all
    t0 = time.time()
    for b in dl:
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast(**amp):
            pred = model(b["points"], b["normals"], b["pairs"]).float()
        gt = b["dist"]
        err = (pred - gt).abs()
        mask = gt > a.min_gt
        rel = torch.where(mask, err / (gt + a.eps).clamp_min(1e-12), torch.zeros_like(err))
        rows += torch.stack([rel.sum(1), mask.sum(1), err.sum(1),
                             torch.full_like(err[:, 0], err.shape[1])], 1).double().cpu().tolist()
    dt = time.time() - t0

    rows = np.array(rows)
    cats = defaultdict(list)
    for n, r in zip(names, rows):
        cats[n.split("_")[0]].append(r)

    def summarize(rs):
        s = np.sum(rs, 0)
        return {"mre": s[0] / max(s[1], 1), "l1": s[2] / max(s[3], 1), "n_shapes": len(rs)}

    res = {"overall": summarize(rows), "per_category": {c: summarize(r) for c, r in sorted(cats.items())},
           "ckpt": a.ckpt, "epoch": ck.get("epoch"), "n_pairs": a.n_pairs, "eps": a.eps,
           "seconds_per_shape": dt / len(ds)}
    o = res["overall"]
    print(f"{a.split}: {o['n_shapes']} shapes | MRE {100 * o['mre']:.2f}% | L1 {o['l1']:.4f} | "
          f"{1000 * res['seconds_per_shape']:.1f} ms/shape")
    for c, r in res["per_category"].items():
        print(f"  {c}: MRE {100 * r['mre']:.2f}%  L1 {r['l1']:.4f}  ({r['n_shapes']} shapes)")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
