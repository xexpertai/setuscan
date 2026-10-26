"""Real-photo validation hook: run SetuScan on real photos and compare with gauge readings.

Usage (from the repo root, with the project venv):
    PYTHONPATH=. .venv/bin/python eval/eval_real_photos.py PHOTO_DIR GAUGES.csv [--out eval/results/real_photos]

GAUGES.csv columns (header required): file, gauge_width_mm, gauge_length_mm
  file             photo file name inside PHOTO_DIR
  gauge_width_mm   width read with a crack-width card / feeler gauge / ruler at the widest point you can read reliably
  gauge_length_mm  length of the crack along the visible part (optional, may be empty)
See docs/REAL-PHOTO-PROTOCOL.md for how to shoot.

Writes <out>.json and <out>.md. Nothing is invented: rows where SetuScan refuses to measure (no marker,
blurry marker, ...) are listed with the reason and are NOT counted in the error statistics.
The comparison is like for like: the gauge width is compared with SetuScan's MEDIAN width (and the p95
width is shown next to it, because the severity policy uses p95)."""
import argparse, csv, datetime, json, pathlib, statistics as st, sys
import cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from setuscan import agent  # noqa: E402


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def run_one(path: pathlib.Path) -> dict:
    img = cv2.imread(str(path))
    if img is None:
        return {"status": "unreadable", "reason": "could not decode the image"}
    case = agent.run(agent.new_case(img, labels={"source": "REAL photo (validation)"}))
    cur = case.visits["current"]
    out = {"status": case.status, "steps": [t["action"] for t in case.trace], "cracks": []}
    if case.reshoot:
        out["reason"] = "; ".join(case.reshoot.get("reasons", [])) if isinstance(case.reshoot, dict) else str(case.reshoot)
    for c in cur.cracks or []:
        if c.get("width_median_mm") is not None:
            out["cracks"].append({k: c.get(k) for k in ("id", "length_mm", "width_median_mm", "width_max_mm", "width_reliable", "zoomed")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photo_dir"); ap.add_argument("gauges_csv"); ap.add_argument("--out", default="eval/results/real_photos")
    ap.add_argument("--marker-id0-mm", type=float, default=None,
                    help="override the physical size of marker id 0 (mm) when the printed card is not 40 mm; measure it with a ruler and say so in the report")
    a = ap.parse_args()
    if a.marker_id0_mm:
        from setuscan import marker as _m
        _m.MARKER_SIZES_MM[0] = a.marker_id0_mm
    rows = []
    for r in csv.DictReader(open(a.gauges_csv, newline="")):
        f = (r.get("file") or "").strip()
        if not f:
            continue
        res = run_one(pathlib.Path(a.photo_dir) / f)
        gw, gl = num(r.get("gauge_width_mm")), num(r.get("gauge_length_mm"))
        row = {"file": f, "gauge_width_mm": gw, "gauge_length_mm": gl, "status": res["status"], "reason": res.get("reason", "")}
        if res["cracks"]:
            c = max(res["cracks"], key=lambda c: c["width_max_mm"] or 0)  # the widest crack is the one a gauge reading refers to
            row.update(setuscan_width_median_mm=c["width_median_mm"], setuscan_width_p95_mm=c["width_max_mm"], setuscan_length_mm=c["length_mm"],
                       width_reliable=c["width_reliable"], zoomed=c["zoomed"])
            if gw:
                row["width_err_mm"] = round(c["width_median_mm"] - gw, 3); row["width_err_pct"] = round(100 * (c["width_median_mm"] - gw) / gw, 1)
            if gl and c["length_mm"]:
                row["length_err_pct"] = round(100 * (c["length_mm"] - gl) / gl, 1)
        rows.append(row)
    meas = [r for r in rows if "width_err_mm" in r]
    summ = {"n_photos": len(rows), "n_measured": len(meas), "n_refused_or_failed": len(rows) - len(meas)}
    if meas:
        e = [abs(r["width_err_mm"]) for r in meas]
        summ.update(width_mae_mm=round(st.mean(e), 3), width_median_abs_err_mm=round(st.median(e), 3),
                    width_mean_signed_bias_mm=round(st.mean(r["width_err_mm"] for r in meas), 3),
                    width_mape_pct=round(st.mean(abs(r["width_err_pct"]) for r in meas), 1))
    out = {"label": "REAL photos with gauge readings taken by the entrant", "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "marker_id0_mm_used": a.marker_id0_mm or 40.0, "comparison": "gauge width vs SetuScan MEDIAN width (p95 shown separately)", "summary": summ, "rows": rows}
    o = pathlib.Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
    o.with_suffix(".json").write_text(json.dumps(out, indent=1))
    f = lambda v, d=2: "-" if v is None else f"{v:.{d}f}"
    md = ["# Real photos: gauge vs SetuScan", "", "REAL photos, gauge readings taken by the entrant (see `docs/REAL-PHOTO-PROTOCOL.md`). "
          "The gauge width is compared with SetuScan's **median** width; the **p95** width (used by the severity policy) is shown next to it.", "",
          "| Photo | Gauge width (mm) | SetuScan median (mm) | SetuScan p95 (mm) | Error median-gauge (mm) | Error (%) | Gauge length (mm) | SetuScan length (mm) | Outcome |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        outc = r["status"] + (f" ({r['reason']})" if r["reason"] else "") + ("" if r.get("width_reliable", True) else " [thin: width unreliable]")
        md.append(f"| {r['file']} | {f(r['gauge_width_mm'])} | {f(r.get('setuscan_width_median_mm'))} | {f(r.get('setuscan_width_p95_mm'))} | "
                  f"{f(r.get('width_err_mm'), 3)} | {f(r.get('width_err_pct'), 1)} | {f(r['gauge_length_mm'], 0)} | {f(r.get('setuscan_length_mm'), 0)} | {outc} |")
    md += ["", f"Marker id 0 size used: {a.marker_id0_mm or 40.0} mm.", f"Photos: {summ['n_photos']}; measured: {summ['n_measured']}; refused or failed (listed above, not in the statistics): {summ['n_refused_or_failed']}."]
    if meas:
        md.append(f"Width error on measured photos (median vs gauge): MAE {summ['width_mae_mm']} mm, median absolute error {summ['width_median_abs_err_mm']} mm, mean signed bias {summ['width_mean_signed_bias_mm']} mm, mean absolute percentage error {summ['width_mape_pct']} %.")
    md.append("Small n: these figures describe these photos only. Gauge readings have their own uncertainty (about half the smallest gauge step).")
    o.with_suffix(".md").write_text("\n".join(md) + "\n")
    print("\n".join(md)); print("wrote", o.with_suffix(".json"), o.with_suffix(".md"))


if __name__ == "__main__":
    main()
