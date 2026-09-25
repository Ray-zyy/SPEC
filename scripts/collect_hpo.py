"""汇总超参搜索结果, 打印每个架构验证集最优的一组 (拿去跑正式的 8 个种子)."""
import json, sys
from pathlib import Path
import pandas as pd
rows = []
for p in sorted(Path("runs/hpo").glob("*/config.json")):
    c = json.load(open(p))
    rows.append(dict(run=p.parent.name, arch=c["arch"], lr=c["lr"], hidden=c["hidden"],
                     dropout=c["dropout"], batch=c["batch"], val_mse=c.get("best_val_mse")))
df = pd.DataFrame(rows).sort_values(["arch", "val_mse"])
df.to_csv("results/hpo_summary.csv", index=False)
best = df.groupby("arch").head(1)
print(best.to_string(index=False))
print("\n把上面的 --lr/--hidden/--dropout/--batch 填回 scripts/make_joblist_e3.sh 再跑正式训练。")
