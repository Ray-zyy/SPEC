"""
models/dl/train.py —— CAMELS-AUS 区域式深度模型训练 (单卡为主, 可选 DDP)

单卡 (推荐, 由 scripts/gpu_queue.sh 批量调度, 四卡吞吐最优):
  CUDA_VISIBLE_DEVICES=0 python models/dl/train.py --arch lstm --loss nse --seed 0 \
      --regime temporal --basins_file data/processed/basins_aus.txt --out runs/aus_lstm_nse_s0

四卡 DDP (只在调试单个大模型时用):
  torchrun --nproc_per_node=4 models/dl/train.py --arch transformer ... --out runs/dbg

产出 runs/<name>/: config.json, best.pt, preds_test.parquet(列: basin,date,obs,sim), history.csv

与原版的差别 (针对本地 CAMELS-AUS):
  * 静态属性改为 data/processed/attributes_climate_aus.csv (只由 forcing 导出, PUB 制度下也合法);
  * PUB 折改为 data/processed/pub_folds_aus.csv (按 AWRC drainage division 分组的空间折), 不再用 hash;
  * 测试期预测由 O(n_basins × n_samples) 改为一次索引, 222 站不再卡住;
  * 加入早停 (--patience)、每轮窗口子采样 (--subsample) 以控制墙钟时间;
  * 数据从 data/processed/camels_aus_all.pkl 缓存读取.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, SubsetRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from spec.data import SPLITS, load_basin_fast, PROC                      # noqa
from spec.io import save_preds                                           # noqa
from models.dl.nets import ARCHS, LOSSES                                 # noqa

DYN = ["prcp", "tmax", "tmin", "srad", "vp", "pet"]
DATASET = "camels_aus"


# ------------------------------------------------------------------ 数据集
class BasinWindows(Dataset):
    """所有站点的滑动窗口拼接; 目标 = 窗口最后一天的流量 (全局标准化)."""

    def __init__(self, basins, split, seq_len=365, scalers=None, attrs=None):
        self.seq_len = seq_len
        self.samples, self.arrays, self.static, self.qstd = [], {}, {}, {}
        a0, a1 = SPLITS[DATASET][split]
        warm = pd.Timedelta(days=seq_len)
        for b in basins:
            df = load_basin_fast(b).loc[pd.Timestamp(a0) - warm: pd.Timestamp(a1)]
            for c in DYN:
                if c not in df:
                    df[c] = 0.0
            x = df[DYN].astype("float32").ffill().bfill().values
            q = df["q"].astype("float32").values
            self.arrays[b] = (x, q, df.index)
            s = (attrs.loc[b].values.astype("float32")
                 if attrs is not None and b in attrs.index else np.zeros(0, "float32"))
            self.static[b] = s
            self.qstd[b] = float(np.nanstd(q) + 1e-6)
            valid = np.where(np.isfinite(q))[0]
            valid = valid[valid >= seq_len - 1]
            valid = valid[df.index[valid] >= pd.Timestamp(a0)]
            self.samples += [(b, int(i)) for i in valid]
        self.by_basin = defaultdict(list)
        for k, (b, _) in enumerate(self.samples):
            self.by_basin[b].append(k)
        if scalers is None:
            allx = np.concatenate([v[0] for v in self.arrays.values()])
            allq = np.concatenate([v[1] for v in self.arrays.values()])
            scalers = dict(xm=np.nanmean(allx, 0), xs=np.nanstd(allx, 0) + 1e-6,
                           qm=float(np.nanmean(allq)), qs=float(np.nanstd(allq) + 1e-6))
        self.sc = scalers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, k):
        b, i = self.samples[k]
        x, q, _ = self.arrays[b]
        xw = (x[i - self.seq_len + 1: i + 1] - self.sc["xm"]) / self.sc["xs"]
        y = (q[i] - self.sc["qm"]) / self.sc["qs"]
        return (torch.from_numpy(np.nan_to_num(xw)).float(),
                torch.from_numpy(self.static[b]).float(),
                torch.tensor(y).float(),
                torch.tensor(q[i]).float(),
                torch.tensor(self.qstd[b]).float())


def load_static(basins, stat_set="climate"):
    """静态属性: climate = 只由 forcing 导出 (PUB 合法); none = 不入模."""
    if stat_set == "none":
        return pd.DataFrame(index=pd.Index(basins, name="basin"))
    f = PROC / "attributes_climate_aus.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} 不存在, 先运行 python scripts/prepare_data.py")
    att = pd.read_csv(f, index_col=0)
    att.index = att.index.astype(str)
    att = att.loc[att.index.intersection([str(b) for b in basins])].apply(pd.to_numeric, errors="coerce")
    att = att.fillna(att.median(numeric_only=True))
    return (att - att.mean()) / (att.std() + 1e-6)


def build_model(arch, d_dyn, d_stat, args, dev):
    kw = {"hidden": args.hidden, "dropout": args.dropout} if arch in ("lstm", "gru") else {}
    net = ARCHS[arch](d_dyn, d_stat, **kw).to(dev)
    if arch == "patchtst":                       # LazyLinear 需一次前向初始化
        with torch.no_grad():
            net(torch.zeros(2, args.seq_len, d_dyn, device=dev), torch.zeros(2, d_stat, device=dev))
    return net


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="camels_aus")          # 保留参数以兼容旧脚本
    ap.add_argument("--arch", default="lstm", choices=list(ARCHS))
    ap.add_argument("--loss", default="nse", choices=list(LOSSES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regime", default="temporal", choices=["temporal", "pub"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--seq_len", type=int, default=365)
    ap.add_argument("--subsample", type=float, default=0.35,
                    help="每轮随机取用的训练窗口比例 (1.0 = 全量). 222 站全量约 160 万窗口/轮")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--basins_file", default="data/processed/basins_aus.txt")
    ap.add_argument("--folds_file", default="data/processed/pub_folds_aus.csv")
    ap.add_argument("--stat_set", default="climate", choices=["climate", "none"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_amp", action="store_true")
    args = ap.parse_args()
    amp = not args.no_amp

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    basins = [l.strip() for l in open(args.basins_file) if l.strip()]
    if args.regime == "pub":
        fo = pd.read_csv(args.folds_file, dtype={"basin": str})
        fo = fo[fo.basin.isin(basins)]
        train_b = fo[fo.fold != args.fold].basin.tolist()
        test_b = fo[fo.fold == args.fold].basin.tolist()
    else:
        train_b = test_b = basins
    if rank == 0:
        print(f"[data] regime={args.regime} train_basins={len(train_b)} test_basins={len(test_b)}", flush=True)

    attrs = load_static(basins, args.stat_set)
    t0 = time.time()
    tr = BasinWindows(train_b, "train", args.seq_len, None, attrs)
    va = BasinWindows(train_b, "val", args.seq_len, tr.sc, attrs)
    if rank == 0:
        print(f"[data] train windows={len(tr)}  val windows={len(va)}  ({time.time()-t0:.0f}s)", flush=True)

    d_dyn, d_stat = len(DYN), attrs.shape[1]
    model = build_model(args.arch, d_dyn, d_stat, args, dev)
    if ddp:
        model = DDP(model, device_ids=[dev.index], find_unused_parameters=False)

    n_sub = int(len(tr) * min(max(args.subsample, 0.01), 1.0))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    lossfn = LOSSES[args.loss]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vdl = DataLoader(va, batch_size=1024, num_workers=max(1, args.num_workers // 2), pin_memory=True)
    best, bad, hist = np.inf, 0, []

    for ep in range(args.epochs):
        idx = np.random.default_rng(args.seed * 1000 + ep).choice(len(tr), n_sub, replace=False)
        if ddp:
            sub = torch.utils.data.Subset(tr, idx.tolist())
            sampler = DistributedSampler(sub, shuffle=True)
            sampler.set_epoch(ep)
            dl = DataLoader(sub, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers,
                            pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0)
        else:
            dl = DataLoader(tr, batch_size=args.batch, sampler=SubsetRandomSampler(idx.tolist()),
                            num_workers=args.num_workers, pin_memory=True, drop_last=True,
                            persistent_workers=args.num_workers > 0)
        model.train()
        t1, run = time.time(), 0.0
        for x, s, y, yphys, qstd in dl:
            x, s, y, yphys, qstd = [t.to(dev, non_blocking=True) for t in (x, s, y, yphys, qstd)]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                pred = model(x, s)
                if args.loss in ("logmse", "delta"):
                    # log-MSE 与 δ-loss 必须在物理流量空间计算；预测先反标准化并截断为非负。
                    pred_phys = pred * tr.sc["qs"] + tr.sc["qm"]
                    loss = lossfn(pred_phys, yphys, std=qstd)
                else:
                    loss = lossfn(pred, y, std=qstd / tr.sc["qs"])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run += float(loss.detach())
        sched.step()
        if rank == 0:
            model.eval()
            tot = k = 0.0
            with torch.no_grad():
                for x, s, y, yphys, qstd in vdl:
                    x, s, y = x.to(dev), s.to(dev), y.to(dev)
                    tot += float(torch.mean((model(x, s) - y) ** 2)) * len(y)
                    k += len(y)
            vmse = tot / max(k, 1)
            hist.append(dict(epoch=ep, train_loss=run / max(len(dl), 1), val_mse=vmse,
                             sec=round(time.time() - t1, 1)))
            print(f"[ep {ep:02d}] train={run/max(len(dl),1):.4f} val_mse={vmse:.4f} "
                  f"{time.time()-t1:.0f}s", flush=True)
            if vmse < best - 1e-5:
                best, bad = vmse, 0
                torch.save((model.module if ddp else model).state_dict(), out / "best.pt")
            else:
                bad += 1
        if ddp:
            stop = torch.tensor([1 if (rank == 0 and bad >= args.patience) else 0], device=dev)
            dist.broadcast(stop, 0)
            if int(stop.item()):
                break
        elif bad >= args.patience:
            print(f"[early stop] no improvement for {bad} epochs", flush=True)
            break

    # ---------------- 测试期逐站预测并存档
    if rank == 0:
        net = build_model(args.arch, d_dyn, d_stat, args, dev)
        net.load_state_dict(torch.load(out / "best.pt", map_location=dev))
        net.eval()
        te = BasinWindows(test_b, "test", args.seq_len, tr.sc, attrs)
        recs = []
        with torch.no_grad():
            for b in test_b:
                idx = te.by_basin.get(b, [])
                if not idx:
                    continue
                pos = [te.samples[k][1] for k in idx]
                _, q, dates = te.arrays[b]
                preds = []
                for j in range(0, len(idx), 512):
                    chunk = idx[j:j + 512]
                    X = torch.stack([te[k][0] for k in chunk]).to(dev)
                    S = torch.stack([te[k][1] for k in chunk]).to(dev)
                    with torch.amp.autocast("cuda", enabled=amp):
                        preds.append(net(X, S).float().cpu().numpy())
                yh = np.concatenate(preds) * te.sc["qs"] + te.sc["qm"]
                recs.append(pd.DataFrame(dict(basin=b, date=dates[pos], obs=q[pos],
                                              sim=np.clip(yh, 0, None))))
        pred = pd.concat(recs)
        save_preds(pred, out)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        json.dump(vars(args) | {"n_train_basins": len(train_b), "n_test_basins": len(test_b),
                                "best_val_mse": float(best), "epochs_run": len(hist)},
                  open(out / "config.json", "w"), indent=2)
        print(f"[done] {out}  rows={len(pred)}  best_val_mse={best:.4f}", flush=True)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
