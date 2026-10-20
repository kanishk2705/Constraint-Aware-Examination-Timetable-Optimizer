"""Recompute everything from the raw records and write results/summary.json (or fit the decision rule).

  python scripts/analyze.py fit-rule --raw results/raw/tuning.jsonl --data data/tuning      -> results/decision_rule.json
  python scripts/analyze.py eval     --raw results/raw/eval.jsonl   --data data/eval --rule results/decision_rule.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from examtt import analysis  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit-rule", "eval"])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rule", default="results/decision_rule.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--meta", default=None, help="JSON file with freeze commits etc. merged into the summary")
    a = ap.parse_args()
    records = analysis.load_records(a.raw)
    if a.mode == "fit-rule":
        validated, audit = analysis.validate_records(records, a.data)
        rows = analysis.table(records, validated)
        fit = analysis.fit_threshold(rows)
        fit["audit"] = {k: v for k, v in audit.items() if k != "details"}
        json.dump(fit, open(a.out or a.rule, "w"), indent=1)
        print("theta", fit["theta"], "tuning mean regret", fit["mean_regret"], "always_ls", fit["always_ls"], "always_cp", fit["always_cp"])
    else:
        theta = json.load(open(a.rule))["theta"]
        out, _ = analysis.run_all(records, a.data, theta)
        if a.meta and os.path.exists(a.meta):
            out["meta"] = json.load(open(a.meta))
        out["rule"] = json.load(open(a.rule))
        json.dump(out, open(a.out or "results/summary.json", "w"), indent=1)
        h = out["hypotheses"]
        print("audit", out["audit"])
        for k in ("H1", "H3", "H4", "H5", "H6"):
            print(k, "supported" if h[k]["supported"] else "NOT supported")
        print("H2a", h["H2"]["H2a_supported"], "H2b", h["H2"]["H2b_supported"])
