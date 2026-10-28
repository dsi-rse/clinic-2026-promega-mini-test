"""Grad-CAM on the current per-day CNNs (V2: true aspect ratio, cropped, brightened), leakage-free:
each organoid goes through the CV-repeat-1 fold model that never trained on it (rep_1/fold_k/test_ids.json).

Writes to <out_dir>:
  gradcam_focus.csv    every organoid x day x model: share of heat on the organoid (focus), organoid share
                       of the input (mask_frac), P(Acceptable)  -> Fig 2 b
  gradcam_examples.npz input + heat map for 3 typical example organoids (accepted / stalled / overgrown,
                       picked here by a fixed rule) at days 10 and 30 for all 4 CNNs  -> Fig 2 a, Fig 1 a

The organoid mask in input coordinates comes from the same dataset code with mask_only=True, so it goes
through the identical crop / letterbox / resize as the image.

Run from the repo root on the cluster (core_env, GPU):
    python analysis/figures_paper/gradcam_v2.py <out_dir>
"""
import gc, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn.functional as F
from torchvision import transforms as T

sys.path.insert(0, ".")
from analysis.images.cnn_lstm.train_base_model import BaselineEfficientNet, SingleDayOrganoidDataset, input_resize

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("/net/projects2/promega/project_data")
RUNS, MP = DATA / "model_tests/lstm_runs/idor_main_perday", DATA / "amanda_test/model_plots"
VARIANTS = {"std": (False, False), "strong": (False, False), "bbox": (True, False), "brightbbox": (True, True)}   # (bbox crop, brighten)
DAYS = ["3", "6", "8", "10", "13", "15", "17", "20.5", "24", "28", "30"]
EX_DAYS = ("10", "30")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(f"data/cohorts/idor_main/full/{p}.json")))
y = pd.Series({o: int(v["label"] == "Acceptable") for o, v in meta.items()})
plate = pd.Series({o: "_".join(o.split("_")[:3]) for o in meta})
LA = pd.read_csv(MP / "v2/perday_mask_area.csv").pivot(index="organoid_id", columns="day", values="log_area")

# ---- example organoids: typical, not edge cases (fixed rule) ----
P30 = pd.DataFrame({v: pd.read_csv(RUNS / f"base_effnet_kfold4x10_{v}/day_30/oof_predictions.csv").groupby("organoid_id").prob_acceptable.mean()
                    for v in VARIANTS})
has = [o for o in P30.dropna().index if pd.notna(LA.loc[o].get(10.0)) and pd.notna(LA.loc[o].get(30.0))]
acc_med = {pl: LA.loc[[o for o in LA.index if y[o] == 1 and plate[o] == pl], 30.0].median() for pl in plate.unique()}
kind = {o: "accepted" if y[o] else ("overgrown" if LA.loc[o, 30.0] > acc_med[plate[o]] else "stalled") for o in has}
votes = {o: meta[o]["n_votes_good"] for o in has}


def typical(k, ok):
    c = [o for o in has if kind[o] == k and ok(o)]
    med = np.median([LA.loc[o, 30.0] for o in c])
    return min(c, key=lambda o: abs(LA.loc[o, 30.0] - med))


# typical size for its group, clear votes; the CNNs call overgrown rejects ~0.5-0.6, so no P rule there.
# E12 (translucent cyst-like, the old example) is excluded as atypical.
EXAMPLES = [typical("accepted", lambda o: votes[o] == 5 and (P30.loc[o] > .6).all()),
            typical("stalled", lambda o: votes[o] == 0 and P30.loc[o].mean() < .45),
            typical("overgrown", lambda o: votes[o] <= 1 and o != "BA2_96_2_E12_nosplit")]
print("examples:", EXAMPLES, [kind[o] for o in EXAMPLES], [votes[o] for o in EXAMPLES])


def gradcam(model, xb):
    """Grad-CAM for a batch. In eval mode the samples are independent, so backpropagating the sum of the
    per-sample scores gives each sample its own gradient. Each map explains the predicted class
    (as make_gradcam_base_effnet.py)."""
    layer = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)][-1]
    store = {}
    h = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))   # setitem returns None: output untouched
    xb = xb.to(DEV).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logit = model(xb).view(-1); p = torch.sigmoid(logit).detach()
    a = store.pop("a"); h.remove()
    g, = torch.autograd.grad(torch.where(p >= .5, logit, -logit).sum(), a)
    with torch.no_grad():
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=xb.shape[-2:], mode="bilinear", align_corners=False)[:, 0].cpu().numpy()
    del a, g, logit, xb
    return cam, p.cpu().numpy()


rows, ex = [], {}
for v, (bbox, bright) in VARIANTS.items():
    for d in DAYS:
        for fold in sorted((RUNS / f"base_effnet_kfold4x10_{v}/day_{d}/rep_1").glob("fold_*")):
            ids = json.load(open(fold / "test_ids.json"))
            model = BaselineEfficientNet().to(DEV)
            ck = torch.load(next(fold.glob("model_day_*.pth")), map_location=DEV)
            model.load_state_dict(ck["state_dict"]); model.eval()
            for prm in model.parameters():
                prm.requires_grad_(True)
            kw = dict(transform=T.Compose(input_resize("clipped", bbox)), image_type="clipped", bbox_crop=bbox, exact_day=True)
            di = SingleDayOrganoidDataset(ids, meta, float(d), brighten=bright, **kw)
            dm = SingleDayOrganoidDataset(ids, meta, float(d), mask_only=True, **kw)
            assert [s["org_id"] for s in di.samples] == [s["org_id"] for s in dm.samples]
            for b0 in range(0, len(di), 16):
                idx = range(b0, min(b0 + 16, len(di)))
                xs = torch.stack([di[i][0] for i in idx])
                cams, ps = gradcam(model, xs)
                for i, cam, p, x in zip(idx, cams, ps, xs):
                    s = di.samples[i]
                    m = ((dm[i][0] * STD + MEAN).mean(0) < .08).numpy()   # organoid is exactly black in the mask input
                    rows.append(dict(model=v, day=float(d), organoid_id=s["org_id"], label=y[s["org_id"]], prob=float(p),
                                     focus=float(cam[m].sum() / max(cam.sum(), 1e-12)), mask_frac=float(m.mean())))
                    if s["org_id"] in EXAMPLES and d in EX_DAYS:
                        ex[f"{s['org_id']}|{d}|{v}|img"] = np.clip((x * STD + MEAN).permute(1, 2, 0).numpy(), 0, 1).astype(np.float16)
                        ex[f"{s['org_id']}|{d}|{v}|cam"] = (cam / max(cam.max(), 1e-12)).astype(np.float16)
                        ex[f"{s['org_id']}|{d}|{v}|p"] = np.array(float(p))
            del model, ck, di, dm
            gc.collect(); torch.cuda.empty_cache()
        print(v, d, len(rows), torch.cuda.is_available(), flush=True)

pd.DataFrame(rows).to_csv(OUT / "gradcam_focus.csv", index=False)
np.savez_compressed(OUT / "gradcam_examples.npz", examples=np.array(EXAMPLES), **ex)
print("wrote", OUT / "gradcam_focus.csv", len(rows), "rows;", len(ex) // 3, "example maps")
