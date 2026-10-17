import argparse
import json
import math
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import GeodesicDataset
from .model import PRESETS, GeodesicTransformer


def geodesic_loss(pred, gt, eps=1e-2):
    """L1 + relative L1 (relative term keeps short distances from being ignored)."""
    l1 = (pred - gt).abs()
    rel = l1 / (gt + eps)
    return l1.mean() + rel.mean(), rel.detach()


class Logger:
    """metrics.jsonl in the run dir always; optionally Comet (resumes the same experiment on restart)."""

    def __init__(self, a):
        self.f = open(os.path.join(a.out, "metrics.jsonl"), "a")
        self.exp = None
        if not a.comet_project:
            return
        if not os.environ.get("COMET_API_KEY"):
            print("comet: disabled (COMET_API_KEY not set)")
            return
        import comet_ml

        key_path = os.path.join(a.out, "comet_key.txt")
        if os.path.exists(key_path):
            self.exp = comet_ml.ExistingExperiment(previous_experiment=open(key_path).read().strip(),
                                                   display_summary_level=0)
        else:
            self.exp = comet_ml.Experiment(project_name=a.comet_project, display_summary_level=0)
            self.exp.set_name(a.run_name or os.path.basename(os.path.normpath(a.out)))
            self.exp.log_parameters({k: v for k, v in vars(a).items() if k != "model_cfg"})
            self.exp.log_parameters(a.model_cfg, prefix="model")
            with open(key_path, "w") as g:
                g.write(self.exp.get_key())
            # replay metrics logged before comet was enabled for this run
            self.f.flush()
            for line in open(os.path.join(a.out, "metrics.jsonl")):
                m = json.loads(line)
                step, epoch = m.pop("step"), m.pop("epoch")
                self.exp.log_metrics(m, step=step, epoch=epoch)
        print(f"comet: {self.exp.url}", flush=True)

    def log(self, metrics, step, epoch):
        self.f.write(json.dumps({"step": step, "epoch": epoch, **metrics}) + "\n")
        self.f.flush()
        if self.exp is not None:
            self.exp.log_metrics(metrics, step=step, epoch=epoch)


def to_dev(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


@torch.no_grad()
def evaluate(model, loader, dev, amp_dtype, ddp):
    """MRE and L1 over all val pairs (summed across ranks)."""
    model.eval()
    tot = torch.zeros(4, dtype=torch.float64, device=dev)  # rel_sum, n_rel, l1_sum, n_all
    for b in loader:
        b = to_dev(b, dev)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(b["points"], b["normals"], b["pairs"]).float()
        err = (pred - b["dist"]).abs()
        mask = b["dist"] > 1e-4
        tot += torch.stack([(err / b["dist"].clamp_min(1e-4))[mask].sum().double(), mask.sum().double(),
                            err.sum().double(), torch.tensor(err.numel(), device=dev, dtype=torch.float64)])
    if ddp:
        dist.all_reduce(tot)
    model.train()
    return (tot[0] / tot[1].clamp_min(1)).item(), (tot[2] / tot[3].clamp_min(1)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root from prism.preprocess (data/, train.txt, test.txt)")
    ap.add_argument("--out", default="runs/geo")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=16, help="per-GPU batch size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n_points", type=int, default=4096)
    ap.add_argument("--n_pairs", type=int, default=4096)
    ap.add_argument("--size", choices=list(PRESETS), default=None,
                    help="preset S/B/L; explicitly given model args below override it")
    ap.add_argument("--dim", type=int)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--n_anchors", type=int)
    ap.add_argument("--n_enc_cross", type=int)
    ap.add_argument("--n_self", type=int)
    ap.add_argument("--n_dec_cross", type=int)
    ap.add_argument("--no_normals", action="store_true")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers per GPU")
    ap.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    ap.add_argument("--comet_project", default=None, help="log to this Comet project (needs COMET_API_KEY)")
    ap.add_argument("--run_name", default=None, help="comet experiment name (default: out dir name)")
    a = ap.parse_args()

    # single process, or torchrun --nproc_per_node N (DDP)
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dev = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        else:  # CPU/gloo, only for testing
            dev = torch.device("cpu")
    else:
        rank, world = 0, 1
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_proc = rank == 0
    a.world_size, a.global_bs = world, a.bs * world
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[a.amp]
    if dev.type == "cpu":
        amp_dtype = None
    os.makedirs(a.out, exist_ok=True)

    tr = GeodesicDataset(a.data, "train", n_points=a.n_points, n_pairs=a.n_pairs)
    va = GeodesicDataset(a.data, "test", n_points=a.n_points, n_pairs=a.n_pairs)
    tr_sampler = DistributedSampler(tr, shuffle=True, drop_last=True) if ddp else None
    va_sampler = DistributedSampler(va, shuffle=False) if ddp else None
    dl = DataLoader(tr, a.bs, shuffle=tr_sampler is None, sampler=tr_sampler, num_workers=a.workers,
                    drop_last=len(tr) >= a.global_bs, pin_memory=True, persistent_workers=a.workers > 0)
    vl = DataLoader(va, a.bs, sampler=va_sampler, num_workers=a.workers)

    model_cfg = dict(PRESETS[a.size]) if a.size else {}
    for k in ("dim", "heads", "n_anchors", "n_enc_cross", "n_self", "n_dec_cross"):
        if getattr(a, k) is not None:
            model_cfg[k] = getattr(a, k)
    model_cfg["use_normals"] = not a.no_normals
    a.model_cfg = model_cfg  # saved in ckpt["args"] to rebuild the model at inference
    raw = GeodesicTransformer(**model_cfg).to(dev)
    if main_proc:
        print(f"params: {sum(p.numel() for p in raw.parameters()) / 1e6:.2f}M | world {world} | "
              f"global bs {a.global_bs} | {len(dl)} steps/epoch", flush=True)

    decay = [p for n, p in raw.named_parameters() if p.ndim >= 2 and "anchor_tag" not in n]
    no_decay = [p for n, p in raw.named_parameters() if p.ndim < 2 or "anchor_tag" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95))
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1.0))))
    scaler = torch.amp.GradScaler(enabled=amp_dtype == torch.float16)

    step, best, start_ep = 0, float("inf"), 0
    last_path = os.path.join(a.out, "last.pt")
    if os.path.exists(last_path):  # resume after preemption / restart
        ck = torch.load(last_path, map_location=dev, weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        step, best, start_ep = ck["step"], ck["best"], ck["epoch"] + 1
        if main_proc:
            print(f"resumed from {last_path}: epoch {start_ep}, step {step}, best val rel {best:.4f}")
    model = (DDP(raw, device_ids=[dev.index]) if dev.type == "cuda" else DDP(raw)) if ddp else raw
    logger = Logger(a) if main_proc else None

    for ep in range(start_ep, a.epochs):
        if tr_sampler is not None:
            tr_sampler.set_epoch(ep)
        t0 = time.time()
        for b in dl:
            b = to_dev(b, dev)
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(b["points"], b["normals"], b["pairs"])
            loss, rel = geodesic_loss(pred.float(), b["dist"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0 and main_proc:  # rank-0 batch only
                logger.log({"train/loss": loss.item(), "train/rel": rel.mean().item(),
                            "lr": sched.get_last_lr()[0]}, step, ep)
                print(f"ep {ep} step {step} loss {loss.item():.4f} rel {rel.mean().item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        t_train = time.time() - t0
        vrel, vl1 = evaluate(model, vl, dev, amp_dtype, ddp)
        is_best = vrel < best
        best = min(best, vrel)
        if main_proc:
            logger.log({"val/rel": vrel, "val/l1": vl1, "epoch_time": time.time() - t0}, step, ep)
            ckpt = {"model": raw.state_dict(), "args": vars(a), "epoch": ep, "step": step, "best": best,
                    "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict()}
            if is_best:
                torch.save({k: ckpt[k] for k in ("model", "args", "epoch", "step", "best")},
                           os.path.join(a.out, "best.pt"))
            torch.save(ckpt, last_path + ".tmp")
            os.replace(last_path + ".tmp", last_path)  # never leave a truncated last.pt
            print(f"epoch {ep} done in {t_train:.1f}s | val rel {vrel:.4f} l1 {vl1:.4f}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
