"""
Exploratory analysis on raw_logs.csv and features.csv → plots/eda_*.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FEATURES_CSV, PLOTS, RAW_CSV
from features import FEAT_NAMES

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def eda_raw(path: Path, out: Path) -> None:
    if not path.exists():
        print(f"Skip raw EDA (missing {path})")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(8, 4))
    if "level" in df.columns:
        df["level"].fillna("").value_counts().plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Log level counts (raw)")
        _save(fig, out / "eda_raw_level_counts.png")

    if "status" in df.columns and len(df):
        fig, ax = plt.subplots(figsize=(10, 4))
        df["status"].astype(str).value_counts().iloc[:15].plot(kind="bar", ax=ax, color="coral")
        ax.set_title("HTTP status (top 15, raw)")
        _save(fig, out / "eda_raw_status_top.png")

    if "timestamp_ns" in df.columns and len(df) > 10:
        df = df.sort_values("timestamp_ns")
        sec = (df["timestamp_ns"].astype(np.int64) - df["timestamp_ns"].iloc[0]) / 1e9 / 60
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sec, np.arange(len(df)), color="darkgreen")
        ax.set_xlabel("Minutes from first log")
        ax.set_ylabel("Cumulative count")
        ax.set_title("Log volume over time")
        _save(fig, out / "eda_raw_cumulative_time.png")


def eda_features(path: Path, out: Path) -> None:
    if not path.exists():
        print(f"Skip feature EDA (missing {path})")
        return
    df = pd.read_csv(path)
    cols = [c for c in FEAT_NAMES if c in df.columns]
    if not cols:
        print("No feature columns found")
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    axes = axes.ravel()
    for i, c in enumerate(cols):
        if i >= len(axes):
            break
        df[c].astype(float).hist(ax=axes[i], bins=40, color="slateblue", edgecolor="white")
        axes[i].set_title(c)
    _save(fig, out / "eda_feature_histograms.png")

    if len(cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 7))
        corr = df[cols].astype(float).corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
        ax.set_title("Feature correlation")
        _save(fig, out / "eda_feature_correlation.png")

    if "level" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="level", y=cols[0], ax=ax)
        ax.set_title(f"{cols[0]} by level")
        _save(fig, out / "eda_feature_by_level.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_CSV))
    ap.add_argument("--features", default=str(FEATURES_CSV))
    ap.add_argument("--out", default=str(PLOTS))
    args = ap.parse_args()
    out = Path(args.out)
    eda_raw(Path(args.raw), out)
    eda_features(Path(args.features), out)
    print(f"EDA plots written under {out}")


if __name__ == "__main__":
    main()
