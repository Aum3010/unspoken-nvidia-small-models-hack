"""Merge per-modality results into evaluations/results/summary.json (served at /api/evals)."""

import json
import statistics
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "evaluations" / "results"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def eeg():
    from services.eeg import p300

    subs = []
    for s in p300.available_subjects():
        r = p300.train_and_evaluate(s)
        subs.append({"subject": s, "meta": r["meta"], "models": r["models"]})
    m = [s["models"]["xdawn_riemann_lr"] for s in subs]
    out = {"subjects": subs, "mean": {
        "auc": statistics.mean(x["test_auc"] for x in m),
        "acc1": statistics.mean(x["accuracy_by_reps"][0] for x in m),
        "acc5": statistics.mean(x["accuracy_by_reps"][4] for x in m),
        "acc10": statistics.mean(x["accuracy_by_reps"][9] for x in m),
        "lda10": statistics.mean(s["models"]["decimated_lda"]["accuracy_by_reps"][9] for s in subs)}}
    (RES / "eeg.json").write_text(json.dumps(out, indent=2))
    return out


def safety(e2e):
    xml = RES / "pytest.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={xml}"], cwd=ROOT, capture_output=True)
    cases = []
    for tc in ET.parse(xml).getroot().iter("testcase"):
        failed = any(ch.tag in ("failure", "error") for ch in tc)
        cases.append({"name": tc.get("name").replace("test_", "").replace("_", " "), "passed": not failed})
    if e2e:
        cases += [{"name": f"attack: {a}", "passed": a.startswith("blocked") or "403" in a or "400" in a} for a in e2e["attacks"]]
    return {"passed": sum(c["passed"] for c in cases), "total": len(cases), "cases": cases,
            "unsafe_assertions": e2e["unsafe_assertions"] if e2e else None,
            "exactness": e2e["exactness"] if e2e else None}


if __name__ == "__main__":
    e2e = load("e2e")
    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "eeg": eeg(), "vsr": load("vsr"), "audio": load("audio"),
               "coordinator": load("coordinator"), "e2e": e2e, "safety": safety(e2e),
               # original Auto-AVSR vs GRID-adapted decoder and closed phrase list (scripts/finetune_vsr_grid.py)
               "vsr_adaptation": {"finetune": load("vsr_finetune"), "phrases": load("vsr_phrases"),
                                  "phrases_hard": load("vsr_phrases_hard")}}
    if summary["vsr"]:
        summary["vsr"].pop("rows", None)
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", RES / "summary.json")
