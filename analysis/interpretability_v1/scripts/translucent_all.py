"""Translucent fraction for all organoids/days; does it add to size?"""
import json
import numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.stats import mannwhitneyu
from skimage.io import imread
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

T, ERODE = 100, 4
DAYS = [3, 6, 8, 10, 13, 15, 17, 20.5, 24, 28, 30]
OUT = "/net/projects2/promega/project_data/amanda_test/model_plots/translucent_idor.csv"


def pool(d):
    out = {}
    for p in ("train", "val", "test"):
        out.update(json.load(open(f"{d}/series/{p}.json")))
    return out


relaxed, main = pool("data/cohorts/idor_minvotes3_tmp"), pool("data/cohorts/idor_main")
rows = []
for oid, r in relaxed.items():
    for tp in r["timepoints"]:
        img = imread(tp["img_paths"]["clipped"]); g = img.mean(axis=2) if img.ndim == 3 else img.astype(float)
        m = np.asarray(Image.open(tp["mask_paths"]["clipped"]).convert("L")) > 127
        core = binary_erosion(m, iterations=ERODE)
        rows.append({"organoid_id": oid, "day": tp["mdl_day"], "in_main": oid in main,
                     "label": main[oid]["label"] if oid in main else "borderline",
                     "n_votes_good": r["n_votes_good"], "n_votes_total": r["n_votes_total"],
                     "area_frac": float(m.mean()),
                     "translucent_frac": float((g[core] > T).mean()) if core.sum() > 20 else np.nan})
df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
print("saved", OUT, len(df))

c = df[df.in_main]
print("\nper day (clear organoids): median translucent % Acc vs Not | AUC (more translucent = Not) | Mann-Whitney p | % organoids >5% translucent Acc vs Not")
for d in DAYS:
    g = c[c.day == d]; acc, nac = g[g.label == "Acceptable"].translucent_frac, g[g.label != "Acceptable"].translucent_frac
    auc = roc_auc_score((g.label != "Acceptable").astype(int), g.translucent_frac)
    print(f"{d:>5}: {acc.median()*100:5.2f}% vs {nac.median()*100:5.2f}% | AUC {auc:.3f} | p {mannwhitneyu(acc, nac).pvalue:.2g} | "
          f"{(acc > .05).mean():4.0%} vs {(nac > .05).mean():4.0%}")

five = df[(df.day == 30) & (df.n_votes_total == 5)]
print("\nday 30, five-rater organoids: % with >5% translucent by # Acceptable votes")
print(five.groupby("n_votes_good").translucent_frac.agg(lambda s: f"{(s > .05).mean():.0%} (median {s.median()*100:.1f}%, n={len(s)})").to_string())

# does it add to the size trajectory? same 4x10 folds as the CNNs (StratifiedKFold seed 1+r*1000 on sorted ids)
ids = sorted(c.organoid_id.unique())
y = np.array([1 if main[o]["label"] == "Acceptable" else 0 for o in ids])
A = np.log(c.pivot(index="organoid_id", columns="day", values="area_frac").loc[ids])
L = c.pivot(index="organoid_id", columns="day", values="translucent_frac").loc[ids].fillna(0)


def cv(X):
    b, a = [], []
    for rep in range(10):
        p = np.empty(len(ids))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1 + rep * 1000).split(ids, y):
            p[te] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        b.append(balanced_accuracy_score(y, p > .5)); a.append(roc_auc_score(y, p))
    return np.array(b), np.array(a)


print("\nwindow 3-X | size traj bal/AUC | + translucent(day X) | + max translucent(days<=X) | wins (+max vs size)")
for k, d in enumerate(DAYS):
    if d < 17:
        continue
    win = DAYS[:k + 1]
    Xs = A[win].values
    b0, a0 = cv(Xs)
    b1, a1 = cv(np.c_[Xs, L[d].values])
    b2, a2 = cv(np.c_[Xs, L[win].max(axis=1).values])
    print(f"   3-{d:<5} | {b0.mean():.3f} / {a0.mean():.3f} | {b1.mean():.3f} / {a1.mean():.3f} | "
          f"{b2.mean():.3f} / {a2.mean():.3f} | {(b2 > b0).sum()}/10 bal, {(a2 > a0).sum()}/10 AUC")
