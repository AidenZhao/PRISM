"""
ShapeNetCore.v2 -> geodesic training data (following PRISM's recipe).

Per shape:
    model_normalized.obj
      -> watertight manifold (mesh2sdf, UDF level set + sign fix)
      -> keep largest connected component
      -> uniform isotropic remeshing (gpytoolbox / Botsch-Kobbelt) to ~--n_verts vertices
      -> normalize to unit sphere
      -> pick --n_sources random source vertices, exact MMP geodesics (pygeodesic) to all vertices

Output (<out>/data):
    <name>.npz       points (V,3) f32, normals (V,3) f32, faces (F,3) i32, src (S,) i64
    <name>_dist.npy  (S,V) float16 geodesic distances, read with mmap by the dataset
and <out>/train.txt, <out>/test.txt, <out>/failed_shard*.txt

Stages (all idempotent, rerun to resume):
    python -m prism.preprocess select  --zip ShapeNetCore.v2.zip --out DATA_ROOT
    python -m prism.preprocess process --out DATA_ROOT --workers 64   # [--shard i --num_shards k]
    python -m prism.preprocess split   --out DATA_ROOT
"""
import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")  # one core per worker process

import argparse
import glob
import json
import multiprocessing as mp
import time
import traceback
import zipfile
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np


# ----------------------------------------------------------------------------------------------
# select + extract
# ----------------------------------------------------------------------------------------------
def stage_select(a):
    assert a.zip, "--zip is required for the select stage"
    os.makedirs(os.path.join(a.out, "raw"), exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    objs = [n for n in z.namelist() if n.endswith("models/model_normalized.obj")]
    by_cat = defaultdict(list)
    for n in objs:
        by_cat[n.split("/")[-4]].append(n)

    # category-balanced: per-category cap chosen so the total is ~n_select
    sizes = sorted(len(v) for v in by_cat.values())
    cap = max(sizes)
    for c in range(1, max(sizes) + 1):
        if sum(min(s, c) for s in sizes) >= a.n_select:
            cap = c
            break
    rng = np.random.default_rng(a.seed)
    chosen = []
    for cat in sorted(by_cat):
        names = sorted(by_cat[cat])
        k = min(len(names), cap)
        chosen += [names[i] for i in sorted(rng.choice(len(names), k, replace=False))]
    print(f"{len(objs)} objs in {len(by_cat)} categories, per-category cap {cap}, selected {len(chosen)}")

    with open(os.path.join(a.out, "selected.txt"), "w") as f:
        for n in chosen:
            syn, mid = n.split("/")[-4], n.split("/")[-3]
            name = f"{syn}_{mid}"
            f.write(name + "\n")
            dst = os.path.join(a.out, "raw", name + ".obj")
            if not os.path.exists(dst):
                with open(dst, "wb") as g:
                    g.write(z.read(n))


# ----------------------------------------------------------------------------------------------
# per-shape processing
# ----------------------------------------------------------------------------------------------
def largest_component(V, F):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    _, lab = connected_components(adj, directed=False)
    flab = lab[F[:, 0]]
    F = F[flab == np.bincount(flab).argmax()]
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F]


def watertight(V, F, res):
    import mesh2sdf

    # mesh2sdf expects [-1,1]; keep a margin so the offset surface fits in the grid
    c = (V.max(0) + V.min(0)) / 2
    s = 1.8 / (V.max(0) - V.min(0)).max()
    Vn = (V - c) * s
    _, m = mesh2sdf.compute(Vn, F, res, fix=True, level=2.0 / res, return_mesh=True)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)


def remesh(V, F, n_verts, iters=10):
    import gpytoolbox

    def run(h):
        Vr, Fr = gpytoolbox.remesh_botsch(V.astype(np.float64), F.astype(np.int32), iters, h, True)
        return largest_component(np.asarray(Vr), np.asarray(Fr, np.int64))

    # uniform triangulation with edge h has ~ (sqrt(3)/2) h^2 area per vertex
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    h = np.sqrt(2 * area / (np.sqrt(3) * n_verts))
    Vr, Fr = run(h)
    if not 0.85 < len(Vr) / n_verts < 1.2:  # one correction step
        Vr, Fr = run(h * np.sqrt(len(Vr) / n_verts))
    return Vr, Fr


def vertex_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # area weighted
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], n)
    return vn / np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12)


def process_one(name, a):
    import trimesh
    from pygeodesic import geodesic

    rng = np.random.default_rng(zlib.crc32(name.encode()))
    t0 = time.time()
    m = trimesh.load(os.path.join(a.out, "raw", name + ".obj"), file_type="obj", force="mesh",
                     process=False, skip_materials=True)
    V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    if len(F) == 0:
        raise ValueError("empty mesh")

    V, F = watertight(V, F, a.sdf_res)
    V, F = largest_component(V, F)
    V, F = remesh(V, F, a.n_verts)
    if len(V) < a.min_verts:
        raise ValueError(f"too few vertices after remesh: {len(V)}")
    tm = trimesh.Trimesh(V, F, process=False)
    if not tm.is_watertight:
        raise ValueError("not watertight after remesh")

    # unit sphere: bbox center at origin, max radius 1
    V = V - (V.max(0) + V.min(0)) / 2
    V = V / np.linalg.norm(V, axis=1).max()
    t1 = time.time()

    src = rng.choice(len(V), a.n_sources, replace=False)
    alg = geodesic.PyGeodesicAlgorithmExact(V, F.astype(np.int32))
    D = np.empty((a.n_sources, len(V)), np.float32)
    for i, s in enumerate(src):
        d, _ = alg.geodesicDistances(np.array([s], np.int32), None)
        D[i] = d
    if not np.isfinite(D).all() or (D > 1e10).any():
        raise ValueError("non-finite geodesic distances")
    eu = np.linalg.norm(V[src][:, None] - V[None], axis=-1)
    if (D < eu - 1e-4).mean() > 1e-3:
        raise ValueError("geodesic shorter than euclidean; bad mesh")
    t2 = time.time()

    out = os.path.join(a.out, "data", name)
    np.save(out + "_dist.npy.tmp.npy", D.astype(np.float16))
    np.savez(out + ".npz", points=V.astype(np.float32), normals=vertex_normals(V, F).astype(np.float32),
             faces=F.astype(np.int32), src=src.astype(np.int64))
    os.replace(out + "_dist.npy.tmp.npy", out + "_dist.npy")  # dist file marks completion
    return {"name": name, "V": len(V), "F": len(F), "t_mesh": t1 - t0, "t_geo": t2 - t1}


def _child(name, a, q):
    try:
        q.put(("ok", process_one(name, a)))
    except Exception as ex:
        q.put(("err", f"{type(ex).__name__}: {ex}\n{traceback.format_exc(limit=2)}"))


def run_isolated(name, a):
    """Run in a fresh process: native libs (MMP, remesher) may segfault or hang on bad meshes."""
    # spawn, not fork: children forked concurrently from many threads inherit each other's
    # sentinel pipes, which delays join() and can make an exiting child look alive
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(name, a, q))
    p.start()
    p.join(a.timeout)
    if p.is_alive():
        p.join(5)  # child may be mid-exit (fds closed, not yet reaped)
    if p.is_alive():
        p.kill()
        p.join()
        return "err", f"timeout {a.timeout}s"
    try:
        return q.get(timeout=5)
    except Exception:
        return "err", f"crashed (exitcode {p.exitcode})"


def stage_process(a):
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    names = open(os.path.join(a.out, "selected.txt")).read().split()
    if a.limit:
        names = names[: a.limit]
    names = names[a.shard:: a.num_shards]
    # per-shard log files: shards run as separate jobs on a shared filesystem
    failed = set()
    for p in glob.glob(os.path.join(a.out, "failed*.txt")):
        failed |= set(l.split("\t")[0] for l in open(p))
    todo = [n for n in names if n not in failed
            and not os.path.exists(os.path.join(a.out, "data", n + "_dist.npy"))]
    print(f"shard {a.shard}/{a.num_shards}: {len(names)} shapes, {len(todo)} to process, "
          f"{len(failed)} previously failed (all shards), {a.workers} workers", flush=True)

    t0, done, nfail = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, \
            open(os.path.join(a.out, f"failed_shard{a.shard}.txt"), "a") as ff, \
            open(os.path.join(a.out, f"stats_shard{a.shard}.jsonl"), "a") as fs:
        futs = {ex.submit(run_isolated, n, a): n for n in todo}
        for fut in as_completed(futs):
            n = futs[fut]
            status, info = fut.result()
            done += 1
            if status == "ok":
                fs.write(json.dumps(info) + "\n")
                fs.flush()
            else:
                nfail += 1
                ff.write(f"{n}\t{info.splitlines()[0]}\n")
                ff.flush()
            if done % 20 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"[{done}/{len(todo)}] fail {nfail} | {el / 60:.1f} min, "
                      f"eta {el / done * (len(todo) - done) / 60:.1f} min", flush=True)


def stage_split(a):
    names = open(os.path.join(a.out, "selected.txt")).read().split()
    ok = [n for n in names if os.path.exists(os.path.join(a.out, "data", n + "_dist.npy"))]
    rng = np.random.default_rng(a.seed)
    ok = [ok[i] for i in rng.permutation(len(ok))]
    test = sorted(ok[: a.n_test])
    train = sorted(ok[a.n_test: a.n_test + a.n_train])
    for split, lst in (("train", train), ("test", test)):
        with open(os.path.join(a.out, f"{split}.txt"), "w") as f:
            f.write("\n".join(lst) + "\n")
    print(f"{len(ok)} processed -> train {len(train)}, test {len(test)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "process", "split"])
    ap.add_argument("--zip", help="path to ShapeNetCore.v2.zip (select stage)")
    ap.add_argument("--out", required=True, help="dataset root")
    ap.add_argument("--seed", type=int, default=0)
    # select
    ap.add_argument("--n_select", type=int, default=18500, help="over-provisioned for failures")
    # process
    ap.add_argument("--sdf_res", type=int, default=128)
    ap.add_argument("--n_verts", type=int, default=8192)
    ap.add_argument("--min_verts", type=int, default=4096)
    ap.add_argument("--n_sources", type=int, default=500)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 8))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (pilot)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    # split
    ap.add_argument("--n_train", type=int, default=15000)
    ap.add_argument("--n_test", type=int, default=2000)
    a = ap.parse_args()
    {"select": stage_select, "process": stage_process, "split": stage_split}[a.stage](a)
