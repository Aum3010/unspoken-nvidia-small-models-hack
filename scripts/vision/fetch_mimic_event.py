"""Vision preview data: one charted low-glucose event from the MIMIC-IV Clinical Database Demo v2.2.

Downloads only the tables it needs (open access, ODbL 1.0) into data/vision/raw/ (gitignored), finds the glucose,
heart-rate and SpO2 items by name (no hard-coded itemids), picks one ICU stay with a charted glucose below
70 mg/dL, and writes data/vision/mimic_event.json:
  - the dataset's own de-identified ids (subject_id, hadm_id, stay_id) and its shifted dates, as published;
  - the glucose value, unit and charttime;
  - heart rate and SpO2 (numeric values only) for the 6 hours before it;
plus data/vision/ATTRIBUTION.txt and, if missing, data/vision/profile.yaml from profile.example.yaml (SYNTHETIC).
No free text is copied. Never try to re-identify anyone (PhysioNet data use terms).

Usage: python scripts/vision/fetch_mimic_event.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "vision"
RAW = OUT / "raw"
BASE = "https://physionet.org/files/mimic-iv-demo/2.2"
TABLES = ["icu/d_items.csv.gz", "icu/icustays.csv.gz", "icu/chartevents.csv.gz"]
ATTRIBUTION = ("MIMIC-IV Clinical Database Demo v2.2 (Johnson A, Bulgarelli L, Pollard T, Horng S, Celi LA, Mark R), "
               "PhysioNet, https://doi.org/10.13026/dp1f-ex47. Open Data Commons Open Database License v1.0 (ODbL): "
               "https://opendatacommons.org/licenses/odbl/1-0/. De-identified; dates are shifted by the dataset.")
LOW_MG_DL = 70
WINDOW_H = 6


def fetch(table: str) -> Path:
    dst = RAW / table.replace("/", "_")
    if not dst.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"downloading {table}")
        urllib.request.urlretrieve(f"{BASE}/{table}", dst)
    return dst


def items(d_items: pd.DataFrame, pattern: str) -> pd.DataFrame:
    hit = d_items[d_items["label"].str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False)]
    return hit[hit["linksto"] == "chartevents"]


def main():
    d_items = pd.read_csv(fetch("icu/d_items.csv.gz"))
    # measured glucose: charted lab items (not APACHE scores or notes); the unit is on each charted row
    glucose = items(d_items, r"glucose")
    glucose = glucose[glucose["category"].eq("Labs")]
    hr = items(d_items, r"^heart rate$")
    spo2 = items(d_items, r"^o2 saturation pulse ?oxymetry$")
    for name, df in (("glucose", glucose), ("heart rate", hr), ("SpO2", spo2)):
        print(f"{name}: " + ", ".join(f"{r.itemid} {r.label!r} ({r.unitname})" for r in df.itertuples()))
        if df.empty:
            raise SystemExit(f"no chartevents item named like {name}")

    wanted = set(glucose.itemid) | set(hr.itemid) | set(spo2.itemid)
    ce = pd.read_csv(fetch("icu/chartevents.csv.gz"), usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom"])
    ce = ce[ce["itemid"].isin(wanted) & ce["valuenum"].notna()].copy()
    ce["charttime"] = pd.to_datetime(ce["charttime"])

    lows = ce[ce["itemid"].isin(glucose.itemid) & ce["valueuom"].fillna("").str.lower().eq("mg/dl")
              & (ce["valuenum"] < LOW_MG_DL) & (ce["valuenum"] > 0)]
    if lows.empty:
        raise SystemExit("no charted glucose below 70 mg/dL in the demo")
    # deterministic choice: the stay whose low reading has the most heart-rate/SpO2 points in the 6 h before it
    best, best_n = None, -1
    for ev in lows.sort_values(["stay_id", "charttime"]).itertuples():
        w = ce[(ce["stay_id"] == ev.stay_id) & (ce["charttime"] >= ev.charttime - pd.Timedelta(hours=WINDOW_H))
               & (ce["charttime"] <= ev.charttime)]
        n = int(w["itemid"].isin(hr.itemid).sum() + w["itemid"].isin(spo2.itemid).sum())
        if n > best_n:
            best, best_n = ev, n
    ev = best
    w = ce[(ce["stay_id"] == ev.stay_id) & (ce["charttime"] >= ev.charttime - pd.Timedelta(hours=WINDOW_H))
           & (ce["charttime"] <= ev.charttime)].sort_values("charttime")

    def series(ids) -> list[dict]:
        rows = w[w["itemid"].isin(ids)]
        return [{"t_min": round((r.charttime - ev.charttime).total_seconds() / 60, 1), "value": float(r.valuenum)} for r in rows.itertuples()]

    label = glucose.set_index("itemid").loc[ev.itemid, "label"]
    out = {
        "source": {"dataset": "MIMIC-IV Clinical Database Demo", "version": "2.2", "url": "https://physionet.org/content/mimic-iv-demo/2.2/",
                   "license": "ODbL-1.0", "attribution": ATTRIBUTION,
                   "note": "Recorded, de-identified ICU data, used only for a labelled vision preview. No free text. "
                           "Do not attempt re-identification."},
        "ids": {"subject_id": int(ev.subject_id), "hadm_id": int(ev.hadm_id), "stay_id": int(ev.stay_id)},
        "glucose": {"itemid": int(ev.itemid), "label": label, "value": float(ev.valuenum), "unit": ev.valueuom or "mg/dL",
                    "charttime": ev.charttime.isoformat(), "below_mg_dl": LOW_MG_DL},
        "window_hours": WINDOW_H,
        "heart_rate": {"itemids": [int(i) for i in hr.itemid], "unit": "bpm", "points": series(hr.itemid)},
        "spo2": {"itemids": [int(i) for i in spo2.itemid], "unit": "%", "points": series(spo2.itemid)},
        "candidates": {"low_glucose_events": int(len(lows)), "stays": int(lows["stay_id"].nunique())},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mimic_event.json").write_text(json.dumps(out, indent=2))
    (OUT / "ATTRIBUTION.txt").write_text(ATTRIBUTION + chr(10))
    profile = OUT / "profile.yaml"   # the SYNTHETIC profile shown next to it (a template; never real)
    if not profile.exists():
        profile.write_text((Path(__file__).with_name("profile.example.yaml")).read_text())
    print(f"stay {ev.stay_id}: {label} {ev.valuenum} {ev.valueuom} at {ev.charttime}; "
          f"{len(out['heart_rate']['points'])} HR and {len(out['spo2']['points'])} SpO2 points in the {WINDOW_H} h before; "
          f"{len(lows)} low readings in {lows['stay_id'].nunique()} stays -> {OUT / 'mimic_event.json'}")


if __name__ == "__main__":
    main()
