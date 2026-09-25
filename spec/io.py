"""统一预测序列 I/O: 优先 parquet (需 pyarrow), 缺失时自动退回 pickle, 读取端自动识别."""
from pathlib import Path
import pandas as pd

NAME = "preds_test"


def save_preds(df: pd.DataFrame, outdir) -> Path:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    try:
        p = outdir / f"{NAME}.parquet"; df.to_parquet(p, index=False); return p
    except Exception:
        p = outdir / f"{NAME}.pkl"; df.to_pickle(p); return p


def preds_path(rundir):
    rundir = Path(rundir)
    for ext in (".parquet", ".pkl", ".csv"):
        p = rundir / f"{NAME}{ext}"
        if p.exists():
            return p
    return None


def read_preds(rundir) -> pd.DataFrame:
    p = preds_path(rundir)
    if p is None:
        raise FileNotFoundError(f"no predictions in {rundir}")
    df = {".parquet": pd.read_parquet, ".pkl": pd.read_pickle, ".csv": pd.read_csv}[p.suffix](p)
    df["date"] = pd.to_datetime(df["date"])
    return df
