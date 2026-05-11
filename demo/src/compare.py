"""Per-image runtime table — written by stage_report() into the showcase dir."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def runtime_table(hrnet_results: dict, hmr2_results: dict) -> pd.DataFrame:
    rows = [{
        "image":        name,
        "HRNet_2D_ms": hrnet_results[name]["runtime_s"] * 1000,
        "HMR2_3D_ms":  hmr2_results[name]["runtime_s"] * 1000,
    } for name in hrnet_results]
    df = pd.DataFrame(rows)
    df.loc["mean"] = df.mean(numeric_only=True)
    return df


def save_tables(df_runtime: pd.DataFrame, df_agreement=None, out_dir: str | Path = "results") -> None:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df_runtime.to_csv(out_dir / "runtime_table.csv")
    df_runtime.to_latex(out_dir / "runtime_table.tex", float_format="%.1f", index=True)
