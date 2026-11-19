#!/usr/bin/env python3
"""Fusion plot on the 133-organoid cohort: concentration vs flux as the metabolite source.

Inputs (all share organoids, seeds and fold assignments):
  analysis_output/images/combined_results_kfold_series_idor_133_{day}.json
      per-organoid OOF probabilities for met_nan (concentration), morph, img
  analysis_output/metabolites/met_flux_133.json   (met_flux_kfold --folds-from ...)
      OOF probabilities for met_conc / met_flux, LGBM and LogReg

Fusion is recomputed here from those saved OOF probabilities: mean-probability over
the chosen sources, balanced accuracy per repeat (all 133 organoids), then mean and
SD across the 10 repeats.

Outputs (analysis_output/figures/): fusion_flux_133.png, fusion_flux_133.csv

Usage:
    make run ARGS="-m analysis.paper_2026_04.fusion_flux_plot"
"""

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from pipeline.data_loader import DAY_ORDER

IMG_DIR = Path("analysis_output/images")
FIG_DIR = Path("analysis_output/figures")
EXPECTED_N = 133
SINGLE_STYLE = {  # label -> (color, linestyle)
    "Metabolite (concentration)": ("#1F6F78", "-"),
    "Metabolite (flux)": ("#E07A3F", "-"),
    "Morphology": ("#6B5B95", "-"),
    "Image": ("#7A7A7A", "-"),
}


def load_oof(
    day: str, flux_json: dict, met_model: str
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Per-repeat {source: oof probs} (aligned to one organoid order) and the labels."""
    comb = json.loads((IMG_DIR / f"combined_results_kfold_series_idor_133_{day}.json").read_text())[
        day
    ]["repeat_details"]
    flux = flux_json[day]["repeat_details"]
    assert len(comb) == len(flux), f"{day}: repeat count differs"
    reps = []
    for c, f in zip(comb, flux, strict=True):
        assert c["org_ids"] == f["org_ids"], f"{day}: organoid order differs"
        assert len(c["org_ids"]) == EXPECTED_N, (
            f"{day}: n={len(c['org_ids'])}, expected {EXPECTED_N}"
        )
        assert c["seed"] == f["seed"], f"{day}: seed differs"
        assert c["fold_assignments"] == f["fold_assignments"], f"{day}: folds differ"
        assert c["true_labels"] == f["true_labels"], f"{day}: labels differ"
        probs = {
            "conc": np.array(f["oof_probs"][f"met_conc_{met_model}"], dtype=float),
            "flux": np.array(f["oof_probs"][f"met_flux_{met_model}"], dtype=float),
            "morph": np.array([np.nan if v is None else v for v in c["oof_probs"]["morph"]]),
            "img": np.array([np.nan if v is None else v for v in c["oof_probs"]["img"]]),
        }
        reps.append(probs)
    y = np.array(comb[0]["true_labels"], dtype=int)
    # Organoids lacking a prediction from any source on this day (e.g. no morphology
    # row at Dy17) are dropped from every series so all curves share one organoid set.
    keep = np.ones(len(y), dtype=bool)
    for probs in reps:
        for v in probs.values():
            keep &= ~np.isnan(v)
    n_drop = int((~keep).sum())
    if n_drop:
        dropped = [o for o, k in zip(comb[0]["org_ids"], keep, strict=True) if not k]
        warnings.warn(f"{day}: dropped {n_drop} organoid(s) without all sources: {dropped}")
    assert keep.sum() >= EXPECTED_N - 3, f"{day}: too many dropped ({n_drop})"
    reps = [{k: v[keep] for k, v in probs.items()} for probs in reps]
    return reps, y[keep]


def ba_over_repeats(
    reps: list[dict[str, np.ndarray]], y: np.ndarray, sources: tuple[str, ...]
) -> list[float]:
    out = []
    for probs in reps:
        p = np.mean([probs[s] for s in sources], axis=0)
        out.append(float(balanced_accuracy_score(y, (p >= 0.5).astype(int))))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flux-json", type=Path, default=Path("analysis_output/metabolites/met_flux_133.json")
    )
    parser.add_argument("--met-model", choices=["lgbm", "logreg"], default="lgbm")
    parser.add_argument("--out-stem", default="fusion_flux_133")
    args = parser.parse_args()

    flux_json = json.loads(args.flux_json.read_text())
    days = [
        d
        for d in DAY_ORDER
        if d in flux_json
        and (IMG_DIR / f"combined_results_kfold_series_idor_133_{d}.json").exists()
    ]
    assert days, "no days with both flux and combined results"

    rows = []
    for day in days:
        reps, y = load_oof(day, flux_json, args.met_model)
        singles = {
            "conc": "Metabolite (concentration)",
            "flux": "Metabolite (flux)",
            "morph": "Morphology",
            "img": "Image",
        }
        for src, label in singles.items():
            bas = ba_over_repeats(reps, y, (src,))
            rows.append((day, "single", label, np.mean(bas), np.std(bas)))
        for met, mlabel in (("conc", "concentration"), ("flux", "flux")):
            for combo in ([met, "morph"], [met, "img"], ["morph", "img"], [met, "morph", "img"]):
                if met == "flux" and combo == ["morph", "img"]:
                    continue  # identical to the concentration row
                names = {"conc": "Met", "flux": "Met", "morph": "Morph", "img": "Img"}
                label = " + ".join(names[c] for c in combo) + (
                    f" ({mlabel})" if met in combo else ""
                )
                bas = ba_over_repeats(reps, y, tuple(combo))
                rows.append((day, "fusion", label, np.mean(bas), np.std(bas)))
    df = pd.DataFrame(rows, columns=["day", "kind", "series", "ba_mean", "ba_sd"])
    assert not df.duplicated(["day", "series"]).any(), "duplicate series rows"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG_DIR / f"{args.out_stem}.csv", index=False)

    fusion_series = [s for s in df[df.kind == "fusion"].series.unique()]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    ax = axes[0]
    for label, (color, ls) in SINGLE_STYLE.items():
        d = df[(df.kind == "single") & (df.series == label)]
        ax.errorbar(
            d.day,
            d.ba_mean,
            yerr=d.ba_sd,
            color=color,
            ls=ls,
            marker="o",
            ms=5,
            lw=2,
            capsize=2,
            label=label,
        )
    ax.set_title("Single modalities")
    ax = axes[1]
    for i, s in enumerate(fusion_series):
        d = df[(df.kind == "fusion") & (df.series == s)]
        is_flux = "(flux)" in s
        ax.errorbar(
            d.day,
            d.ba_mean,
            yerr=d.ba_sd,
            color=cmap(i),
            ls="--" if is_flux else "-",
            marker="s" if is_flux else "o",
            ms=5,
            lw=2 if "Met + Morph + Img" in s else 1.4,
            capsize=2,
            label=s,
        )
    ax.set_title("Mean-probability fusion (dashed = flux)")
    for ax in axes:
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Day")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", ncol=1)
    axes[0].set_ylabel("Balanced accuracy (mean ± SD over repeats)")
    fig.suptitle(
        f"series_idor, n={EXPECTED_N}, 10x4-fold CV, identical folds ({args.met_model.upper()} metabolite model)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{args.out_stem}.png", dpi=150)
    print(f"Saved {FIG_DIR / (args.out_stem + '.png')}")
    print(df[df.day == days[-1]].to_string(index=False))


if __name__ == "__main__":
    main()
