"""Which days does the LSTM rely on? Per-day dropout / blanking on the V2 CV LSTMs.

For each window (days 3..W), readout and fold, loads the fold's checkpoint (trained with
train_temporal_lstm_kfold.py --keep-models --repeats 1 on data/cohorts/idor_main/full)
and, for every held-out organoid and every day in its sequence:
  blank: replace that day's frame with zeros (= ImageNet mean after normalisation),
         as in analysis/images/cnn_lstm/compute_occlusion_lstm.py
  drop:  remove the frame (and its day tag) from the sequence, as the V2 LSTM saw for
         edge-filtered days during training
and records P(Acceptable) before and after. Writes lstm_day_dropout.csv to <out_dir>.

    python analysis/interpretability_v2/scripts/lstm_day_dropout.py <runs_dir> <out_dir>
"""
import csv, json, sys
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.images.cnn_lstm.train_temporal_ablation_lstm import OrganoidCNN_LSTM
from analysis.images.cnn_lstm.organoid_dataset import OrganoidTimeSeriesDataset

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
meta = {}
for p in ("train", "val", "test"):
    meta.update(json.load(open(ROOT / f"data/cohorts/idor_main/full/{p}.json")))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([transforms.Resize((384, 384), interpolation=InterpolationMode.BILINEAR)])

rows = []
for ro in ("last", "mean"):
    for W in (13, 17, 24, 30):
        for fd in sorted((RUNS / f"temporal_lstm_kfold4x10_{ro}" / f"days_3-{W}" / "rep_1").glob("fold_*")):
            ck = fd / f"model_days_3-{W}.pth"
            model = OrganoidCNN_LSTM(readout=ro).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev)["state_dict"], strict=True); model.eval()
            ids = json.load(open(fd / "test_ids.json"))
            ds = OrganoidTimeSeriesDataset(ids, meta, max_day=W, transform=tf, image_type="clipped")
            with torch.no_grad():
                for i, oid in enumerate(ids):
                    seq, dn, lab, _, _ = ds[i]
                    days = [tp["mdl_day"] for tp in meta[oid]["timepoints"] if tp["mdl_day"] <= W]
                    assert len(days) == seq.shape[0]
                    prob = lambda s, d: float(torch.sigmoid(model(s[None].to(dev), d[None].to(dev).float()))[0])
                    p_full = prob(seq, dn)
                    for t, day in enumerate(days):
                        sb = seq.clone(); sb[t] = 0
                        keep = [k for k in range(len(days)) if k != t]
                        rows.append(dict(readout=ro, window=f"3-{W}", fold=fd.name, organoid_id=oid, label=int(lab),
                                         day=day, position_from_end=len(days) - 1 - t, p_full=round(p_full, 5),
                                         p_blank=round(prob(sb, dn), 5), p_drop=round(prob(seq[keep], dn[keep]), 5)))
            print(f"{ro} 3-{W} {fd.name}: {len(ids)} organoids", flush=True)
            del model; torch.cuda.empty_cache()

with open(OUT / "lstm_day_dropout.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows")
