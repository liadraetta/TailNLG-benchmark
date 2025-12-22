import pandas as pd
import numpy as np
import glob
import os


def pareto_cutoff(file_path, threshold=0.8):
    df = pd.read_csv(file_path)

    df = df.sort_values("claims", ascending=False)

    total_claims = df["claims"].sum()

    df["cum_share"] = df["claims"].cumsum() / total_claims

    cutoff_idx = np.argmax(df["cum_share"].values >= threshold)

    cutoff_value = df.iloc[cutoff_idx]["claims"]

    # Head/tail counts
    head_count = cutoff_idx + 1
    tail_count = len(df) - head_count

    category = os.path.splitext(os.path.basename(file_path))[0]

    return {
        "category": category,
        "cutoff_claims": int(cutoff_value),
        "head_entities": head_count,
        "tail_entities": tail_count,
        "total_entities": len(df)
    }


results = []
for f in glob.glob("..."):
    results.append(pareto_cutoff(f))

summary = pd.DataFrame(results)
print(summary)

summary.to_csv("...", index=False)

