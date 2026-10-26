"""mm-accuracy on SYNTHETIC scenes with exact ground truth.

For every (width, camera resolution, seed) a synthetic wall photo is rendered with
a printed marker and one constant-width crack. The pipeline must find the marker,
find the crack, and measure its width and length in mm. Errors are reported in mm
and in percent, grouped by true width and by camera px/mm, for:
  * the working view (default analysis, bounded image size), and
  * the zoom view (native resolution on the crack's bounding box), i.e. what the
    agent gets when it decides to look closer.

These numbers describe the measurement maths under controlled, synthetic
conditions. They are not a real-world accuracy claim.

Usage: uv run python eval/eval_synthetic_mm.py [--seeds 4]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import provenance, write_json  # noqa: I001
from setuscan import marker as mk
from setuscan.pipeline import analyze, analyze_view, zoom_view
from setuscan.synth import SceneSpec, default_crack, render_scene

WIDTHS = [0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
PPMS = [4.0, 7.0, 12.0]


def one(args):
    width, ppm, seed = args
    rng = np.random.default_rng(seed * 7919 + int(width * 100) + int(ppm * 10))
    crack = default_crack(rng, width_mm=width, length_mm=float(rng.uniform(110, 160)))
    spec = SceneSpec(seed=int(rng.integers(1e9)), cracks=[crack], cam_ppm=ppm, tilt=float(rng.uniform(0.0, 0.12)),
                     rotate_deg=float(rng.uniform(-10, 10)))
    img, gt = render_scene(spec)
    t0 = time.perf_counter()
    a = analyze(img)
    ms = (time.perf_counter() - t0) * 1000
    g = gt["cracks"][0]
    row = {"width_mm": width, "cam_ppm": ppm, "seed": seed, "image_size": gt["image_size"],
           "gt_length_mm": g["visible_length_mm"], "marker_in_frame": gt["marker_in_frame"],
           "marker_found": a.scale.found, "n_cracks": len(a.cracks), "work_mm_per_px": a.mm_per_px,
           "work_blur_sigma_px": a.blur_sigma_px, "edge_blur_px_native": a.scale.edge_blur_px,
           "true_blur_sigma_px": spec.blur_sigma, "analyze_ms": round(ms, 1)}
    if not a.scale.found or not a.cracks:
        return row
    c = max(a.cracks, key=lambda m: m.length_px)
    k = a.mm_per_px
    row.update(work_width_median_mm=c.width_median_px * k, work_width_p95_mm=c.width_max_px * k,
               work_width_px=c.width_median_px, work_raw_width_mm=c.raw_width_median_px * k,
               total_length_mm=sum(m.length_px for m in a.cracks) * k, main_length_mm=c.length_px * k)
    z = zoom_view(a.view, c.bbox)
    az = analyze_view(img, z, a.scale, a.quality)
    if az.cracks:
        cz = max(az.cracks, key=lambda m: m.length_px)
        row.update(zoom_mm_per_px=az.mm_per_px, zoom_width_median_mm=cz.width_median_px * az.mm_per_px,
                   zoom_width_p95_mm=cz.width_max_px * az.mm_per_px, zoom_width_px=cz.width_median_px)
    return row


def summarize(rows: list[dict], key: str) -> dict:
    out = {}
    for val in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == val]
        inframe = [r for r in sub if r["marker_in_frame"]]
        d = {"n": len(sub), "n_marker_in_frame": len(inframe),
             "marker_found_rate_when_in_frame": round(float(np.mean([r["marker_found"] for r in inframe])), 3)
             if inframe else None,
             "marker_false_detections": sum(1 for r in sub if r["marker_found"] and not r["marker_in_frame"]),
             "crack_found_rate_when_marker_found": round(float(np.mean(
                 [r.get("work_width_median_mm") is not None for r in sub if r["marker_found"]])), 3)}
        raw = [r["work_raw_width_mm"] - r["width_mm"] for r in sub if "work_raw_width_mm" in r]
        if raw:
            d["work_raw_width_bias_mm_before_blur_correction"] = round(float(np.mean(raw)), 4)
        for view in ("work", "zoom"):
            errs = [r[f"{view}_width_median_mm"] - r["width_mm"] for r in sub if f"{view}_width_median_mm" in r]
            if errs:
                e = np.array(errs)
                rel = np.array([x / r["width_mm"] for x, r in zip(errs, [r for r in sub if f"{view}_width_median_mm" in r])])
                d[f"{view}_width_mae_mm"] = round(float(np.abs(e).mean()), 4)
                d[f"{view}_width_bias_mm"] = round(float(e.mean()), 4)
                d[f"{view}_width_mape_pct"] = round(float(np.abs(rel).mean() * 100), 1)
                d[f"{view}_n"] = len(errs)
        lerr = [abs(r["total_length_mm"] - r["gt_length_mm"]) / r["gt_length_mm"] for r in sub
                if "total_length_mm" in r and r["gt_length_mm"] > 0]
        if lerr:
            d["length_mape_pct"] = round(float(np.mean(lerr) * 100), 1)
            d["length_median_ape_pct"] = round(float(np.median(lerr) * 100), 1)
        out[str(val)] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    args = ap.parse_args()
    jobs = [(w, p, s) for w in WIDTHS for p in PPMS for s in range(args.seeds)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, jobs, chunksize=2))
    res = {
        "provenance": provenance("eval/eval_synthetic_mm.py"),
        "label": "SYNTHETIC images with exact ground truth; not real photos",
        "design": {"blur_sigma_px_true": 0.7, "length_metric": "sum of detected crack lengths vs visible "
                   "ground-truth centre-line length", "widths_mm": WIDTHS, "cam_ppm": PPMS, "seeds": args.seeds, "n_scenes": len(rows),
                   "marker_sizes_mm": mk.MARKER_SIZES_MM, "width_metric": "median of per-point profile widths "
                   "(constant-width cracks, so median ~ true width)"},
        "overall": summarize([dict(r, all="all") for r in rows], "all")["all"],
        "by_width_mm": summarize(rows, "width_mm"),
        "by_cam_ppm": summarize(rows, "cam_ppm"),
        "rows": rows,
        "wall_time_s": round(time.time() - t0, 1),
    }
    write_json("synthetic_mm_accuracy.json", res)
    o = res["overall"]
    print({k: o[k] for k in o})


if __name__ == "__main__":
    main()
