"""Show how each S5 domain score (G, A, Au, T, R, SAI) is computed from a report's episodes.

Usage:
    python metric_calculations.py LABEL=REPORT.json [LABEL=REPORT.json ...] > METRIC_CALCULATIONS.md

Each report is recomputed from its per-episode data with the formulas in
stronger_ai_metric_spec_v1_1.json, as implemented in run_s5_benchmark.py, and checked against
the scores stored in the report. Read-only: it changes no results.
"""
import json
import sys
from pathlib import Path


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def solved(ep):
    return bool(ep["summary"]["solved"])


def generalization(d):
    eps = d["details"]["generalization"]["episodes"]
    flags = [solved(e) for e in eps]
    value = sum(flags) / len(flags)
    rows = [f"| G{i+1:02d} | seed {e['episode']['seed']} | {'solved' if s else 'not solved'} in {e['summary']['actions']} actions |"
            for i, (e, s) in enumerate(zip(eps, flags))]
    return value, f"G = solved / episodes = {sum(flags)}/{len(flags)} = {fmt(value)}", rows


def adaptation(d):
    det = d["details"]["adaptation"]
    budget = det["recovery_budget"]
    terms, rows = [], []
    for i, e in enumerate(det["episodes"]):
        lat = e.get("recovery_latency")
        L = budget if lat is None else int(lat)
        t = max(0.0, 1 - L / budget)
        terms.append(t)
        rows.append(f"| A{i+1:02d} | change at turn {e.get('change_turn')} | latency {lat if lat is not None else 'none (counted as ' + str(budget) + ')'} → 1 − {L}/{budget} = {fmt(t)}; episode {'solved' if solved(e) else 'not solved'} |")
    value = sum(terms) / len(terms)
    return value, f"A = mean(1 − L/{budget}) = ({' + '.join(fmt(t) for t in terms)}) / {len(terms)} = {fmt(value)}", rows


def autonomy(d):
    eps = d["details"]["generalization"]["episodes"]
    terms, rows = [], []
    for i, e in enumerate(eps):
        m = e["measurements"]
        ner, rar, ncap = m["novel_experiment_rate"], m["redundant_action_rate"], m["no_change_action_persistence"]
        t = (ner + (1 - rar) + (1 - ncap)) / 3
        terms.append(t)
        rows.append(f"| G{i+1:02d} | NER {ner:.3f}, RAR {rar:.3f}, NCAP {ncap:.3f} | ({ner:.3f} + {1-rar:.3f} + {1-ncap:.3f}) / 3 = {fmt(t)} |")
    value = sum(terms) / len(terms)
    return value, f"Au = mean over generalization episodes = ({' + '.join(fmt(t) for t in terms)}) / {len(terms)} = {fmt(value)}", rows


def transfer(d):
    pairs = d["details"]["transfer"]["pairs"]
    terms, rows = [], []
    for p in pairs:
        c, w = p["cold"], p["warm"]
        cs, ws = solved(c), solved(w)
        if ws and not cs:
            why = "warm-only solve → 1"
        elif cs and not ws:
            why = "cold-only solve → 0"
        elif not cs and not ws:
            why = "both failed → 0.5"
        else:
            ca, wa = c["summary"]["actions"], w["summary"]["actions"]
            why = f"both solved → 0.5 + 0.5·({ca} − {wa})/{ca}"
        terms.append(float(p["score"]))
        rows.append(f"| {p['id']} | cold {'solved' if cs else 'failed'}, warm {'solved' if ws else 'failed'} | {why} = {fmt(float(p['score']))} |")
    value = sum(terms) / len(terms)
    src = d["details"]["transfer"]["source"]
    note = f"source episode {'solved' if solved(src) else 'not solved'}"
    return value, f"T = mean pair score = ({' + '.join(fmt(t) for t in terms)}) / {len(terms)} = {fmt(value)} ({note})", rows


def robustness(d):
    det = d["details"]["robustness"]
    clean, pert = det["clean"], det["perturbed"]
    cs = sum(solved(e) for e in clean) / len(clean)
    ps = sum(solved(e) for e in pert) / len(pert)
    if cs > 0:
        value = min(1.0, ps / cs)
        calc = f"R = min(1, perturbed rate / clean rate) = min(1, {ps:.3f} / {cs:.3f}) = {fmt(value)}"
    else:
        value = 0.0
        calc = f"R = 0 because the clean solve rate is 0 ({sum(solved(e) for e in clean)}/{len(clean)}); spec v1.1 gives no robustness credit without clean capability"
    rows = [f"| R{i+1:02d} | clean {'solved' if solved(c) else 'failed'}, perturbed ({det['perturbation_rate']:.0%} noise) {'solved' if solved(p) else 'failed'} |  |"
            for i, (c, p) in enumerate(zip(clean, pert))]
    return value, calc, rows


DOMAINS = [("G", "Generalization", generalization), ("A", "Adaptation", adaptation),
           ("Au", "Autonomy", autonomy), ("T", "Transfer", transfer), ("R", "Robustness", robustness)]


def section(label, path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = [f"## {label}", "", f"Report `{path}` — model `{d['model']}`, condition `{d['condition']}`, profile `{d['profile']}`.", ""]
    values = {}
    for key, name, fn in DOMAINS:
        value, calc, rows = fn(d)
        values[key] = value
        stored = d["scores"].get(key)
        check = "matches report" if stored is not None and abs(stored - value) < 1e-9 else f"REPORT SAYS {fmt(stored)}"
        out += [f"### {name} ({key}) = {fmt(value)} — {check}", "", calc, "", "| episode | data | calculation |", "|---|---|---|", *rows, ""]
    sai = sum(values.values()) / len(values)
    check = "matches report" if abs(d["SAI"] - sai) < 1e-9 else f"REPORT SAYS {fmt(d['SAI'])}"
    out += [f"### SAI composite = {fmt(sai)} — {check}", "",
            "SAI = (G + A + Au + T + R) / 5 = (" + " + ".join(fmt(values[k]) for k, _, _ in DOMAINS) + f") / 5 = {fmt(sai)}", ""]
    return out, values, sai, d


def main(argv):
    if not argv:
        sys.exit(__doc__)
    sections, summary = [], []
    for arg in argv:
        label, path = arg.split("=", 1)
        lines, values, sai, d = section(label, path)
        sections += lines
        g = d["details"]["generalization"]["episodes"]
        summary.append((label, values, sai, sum(solved(e) for e in g), len(g)))
    head = ["# S5 metric calculations", "",
            "Generated by `metric_calculations.py` from the per-episode data in each report.", "",
            "| metric | " + " | ".join(s[0] for s in summary) + " |", "|---|" + "---|" * len(summary)]
    for key, name, _ in DOMAINS:
        cells = [f"{fmt(s[1][key])} ({s[3]}/{s[4]})" if key == "G" else fmt(s[1][key]) for s in summary]
        head.append(f"| {name} ({key}) | " + " | ".join(cells) + " |")
    head.append("| SAI composite | " + " | ".join(fmt(s[2]) for s in summary) + " |")
    print("\n".join(head + [""] + sections))


if __name__ == "__main__":
    main(sys.argv[1:])
