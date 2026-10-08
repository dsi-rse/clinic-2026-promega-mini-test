"""Borderline (3/2) organoids: size vs vote fraction, and size-model predictions on them."""
import json, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]


def pool(d):
    out = {}
    for p in ("train", "val", "test"):
        out.update(json.load(open(f"{d}/series/{p}.json")))
    return out


relaxed = pool("data/cohorts/idor_minvotes3_tmp")   # pooled votes, >=3 on one side
main = pool("data/cohorts/idor_main")
ad = json.load(open("data/all_data.json")); ad = ad.get("records", ad)
off = {}
for k, r in ad.items():
    p = k.split()
    if p[2] == "Dy30":
        lab = r.get("label") or {}
        off["_".join([p[0], p[1], p[3]])] = lab

rows = []
for oid, r in relaxed.items():
    bw = r["base_well"]; lab = off.get(bw, {})
    v = lab.get("votes") or {}; rv = lab.get("regular_votes") or {}
    a = {}
    for tp in r["timepoints"]:
        m = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L"))
        a[tp["mdl_day"]] = float((m > 127).mean())
    rows.append({"organoid_id": oid, "in_main": oid in main, "official": lab.get("value"),
                 "n_acc": v.get("Acceptable", 0), "n_not": v.get("Not Acceptable", 0),
                 "n_total": lab.get("total_evaluations", 0),
                 "reg_acc": rv.get("Acceptable", 0), "reg_not": rv.get("Not Acceptable", 0),
                 **{f"area_{d}": a.get(d, np.nan) for d in DAYS}})
df = pd.DataFrame(rows)
df["frac_acc"] = df.n_acc / (df.n_acc + df.n_not)
df["group"] = np.select(
    [df.in_main & (df.official == "Acceptable"), df.in_main & (df.official == "Not Acceptable"),
     ~df.in_main & (df.frac_acc > .5), ~df.in_main & (df.frac_acc < .5)],
    ["Acceptable (clear)", "Not Acceptable (clear)", "Borderline, leaning Acc", "Borderline, leaning Not"], "other")
df.to_csv("/net/projects2/promega/project_data/amanda_test/model_plots/borderline_votes_sizes.csv", index=False)

print("organoids in relaxed cohort:", len(df), "| in idor_main:", int(df.in_main.sum()), "| extra:", int((~df.in_main).sum()))
print(df[~df.in_main].assign(votes=lambda x: x.n_acc.astype(str) + "/" + (x.n_acc + x.n_not).astype(str))
      .groupby(["votes", "official"], dropna=False).size().to_string())
print("\nmedian area (% of frame) by group:")
print((df.groupby("group")[[f"area_{d}" for d in (6, 17, 24, 30)]].median() * 100).round(2).assign(
    n=df.groupby("group").size()).to_string())
print("\nfive-rater organoids: median day-30 area by # Acceptable votes:")
five = df[(df.n_acc + df.n_not) == 5]
print((five.groupby("n_acc").area_30.agg(["median", "count"]).assign(median=lambda x: (x["median"] * 100).round(2))).to_string())
print("\nSpearman(area, fraction Acceptable votes), all organoids:")
for d in (3, 10, 17, 24, 30):
    print(f"  day {d}: all {spearmanr(df[f'area_{d}'], df.frac_acc, nan_policy='omit')[0]:+.2f} | "
          f"clear only {spearmanr(df[df.in_main][f'area_{d}'], df[df.in_main].frac_acc, nan_policy='omit')[0]:+.2f} | "
          f"borderline only {spearmanr(df[~df.in_main][f'area_{d}'], df[~df.in_main].frac_acc, nan_policy='omit')[0]:+.2f}")
# size-trajectory model trained on the 140 clear organoids, applied to borderline organoids
tr = df[df.in_main]; te = df[~df.in_main]; y = (tr.official == "Acceptable").astype(int).values
for last in (17, 30):
    cols = [f"area_{d}" for d in DAYS if d <= last]
    m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    m.fit(np.log(tr[cols].values), y)
    p = m.predict_proba(np.log(te[cols].values))[:, 1]
    te = te.assign(**{f"p_traj_{last}": p})
    print(f"\nsize-trajectory model (days 3-{last}, trained on clear organoids) on borderline organoids:")
    print(te.groupby("group")[f"p_traj_{last}"].agg(["mean", "count"]).round(3).to_string())
    print(f"  Spearman(P, fraction Acceptable votes) among borderline: {spearmanr(p, te.frac_acc)[0]:+.2f}")
te.to_csv("/net/projects2/promega/project_data/amanda_test/model_plots/borderline_traj_preds.csv", index=False)
