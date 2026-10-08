"""Feature screen: does each image feature add to (plate-normalised) size? 4x10 CV + LRT + FDR."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
df = pd.read_csv(sys.argv[1]); OUT = Path(sys.argv[2])
df["plate"] = df.organoid_id.str.split("_").str[:3].str.join("_")
FEATS = ["circularity", "solidity", "elongation", "n_concavities", "boundary_roughness",
         "mean_intensity", "intensity_sd", "translucent_frac", "dense_frac", "texture_sharpness",
         "glcm_contrast", "glcm_homogeneity", "rim_minus_core", "rim_sd", "edge_sharpness"]
NICE = {"circularity": "Roundness", "solidity": "Compactness (solidity)", "elongation": "Elongation",
        "n_concavities": "Number of lobes/concavities", "boundary_roughness": "Boundary roughness",
        "mean_intensity": "Brightness (inside)", "intensity_sd": "Unevenness (inside)",
        "translucent_frac": "Translucent / cyst area", "dense_frac": "Very dark (dense) area",
        "texture_sharpness": "Texture sharpness", "glcm_contrast": "Texture contrast",
        "glcm_homogeneity": "Texture smoothness", "rim_minus_core": "Rim brighter than core",
        "rim_sd": "Rim unevenness", "edge_sharpness": "Edge crispness"}
# label-free plate normalisation: z-score within plate and day
for c in FEATS + ["log_area"]:
    df[c + "_z"] = df.groupby(["plate", "day"])[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
DAYS = sorted(df.day.unique())
ids = sorted(df.organoid_id.unique())
y = df.drop_duplicates("organoid_id").set_index("organoid_id").loc[ids, "label"].eq("Acceptable").astype(int).values
splits = [list(StratifiedKFold(4, shuffle=True, random_state=1 + r * 1000).split(ids, y)) for r in range(10)]


def cv_auc(X):
    out = []
    for sp in splits:
        p = np.empty(len(ids))
        for tr, te in sp:
            p[te] = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        out.append(roc_auc_score(y, p))
    return np.array(out)


def ll(X):
    return sm.Logit(y, sm.add_constant(X)).fit(disp=0)


rows = []
for d in DAYS:
    g = df[df.day == d].set_index("organoid_id").loc[ids]
    s = g[["log_area_z"]].values
    base = cv_auc(s); m0 = ll(s)
    for f in FEATS:
        x = g[[f + "_z"]].values
        if np.isnan(x).any():
            x = np.nan_to_num(x)
        both = cv_auc(np.c_[s, x]); m1 = ll(np.c_[s, x])
        p = chi2.sf(2 * (m1.llf - m0.llf), 1)
        rows.append({"day": d, "feature": f, "size_auc": base.mean(), "size_plus_auc": both.mean(),
                     "delta_auc": both.mean() - base.mean(), "wins": int((both > base).sum()),
                     "lrt_p": p, "direction": "higher = Acceptable" if m1.params[2] > 0 else "higher = Not Acc.",
                     "alone_auc": roc_auc_score(y, x.ravel()), "corr_with_size": np.corrcoef(s.ravel(), x.ravel())[0, 1]})
R = pd.DataFrame(rows)
R["q"] = multipletests(R.lrt_p, method="fdr_bh")[1]
R.to_csv(OUT / "feature_screen_results.csv", index=False)
hits = R[(R.q < .05) & (R.delta_auc > .01)].sort_values(["day", "delta_auc"], ascending=[True, False])
print(f"{len(R)} tests; {int((R.q < .05).sum())} significant after FDR; {len(hits)} also improve CV AUC by >0.01\n")
print(hits[["day", "feature", "size_auc", "size_plus_auc", "delta_auc", "wins", "q", "direction"]].round(3).to_string(index=False))

# heatmap: delta AUC, star = FDR q < .05
H = R.pivot(index="feature", columns="day", values="delta_auc").loc[FEATS]
Q = R.pivot(index="feature", columns="day", values="q").loc[FEATS]
sns.set_theme(context="paper", style="white")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "savefig.dpi": 300, "savefig.bbox": "tight"})
fig, ax = plt.subplots(figsize=(12, 7.5))
lim = max(.06, np.nanmax(np.abs(H.values)))
sns.heatmap(H, cmap="PuOr", center=0, vmin=-lim, vmax=lim, ax=ax, linewidths=.5, linecolor="white",
            cbar_kws={"label": "AUC gain over plate-adjusted size\n(4x10 CV)"})
for i, f in enumerate(FEATS):
    for j, d in enumerate(H.columns):
        if Q.loc[f, d] < .05 and H.loc[f, d] > 0:
            ax.text(j + .5, i + .5, "*", ha="center", va="center", fontsize=18, fontweight="bold")
ax.set_yticklabels([NICE[f] for f in FEATS], rotation=0, fontsize=13)
ax.set_xticklabels([f"{d:g}" for d in H.columns], rotation=0, fontsize=13); ax.set_xlabel("Day", fontsize=15); ax.set_ylabel("")
ax.set_title("Which image features predict quality beyond size?  (* = significant after FDR correction)", fontsize=14)
for e in ("png", "pdf"):
    fig.savefig(OUT / f"20_feature_screen_beyond_size.{e}")
print("\nsaved 20")
