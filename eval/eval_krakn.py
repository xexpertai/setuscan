"""Crack / no-crack classification on the public KrakN dataset (CC BY 4.0).

Dataset: KrakN associated dataset, Zarski, Wojcik, Miszczak (2021),
Zenodo record 5266356, DOI 10.5281/zenodo.5266356, licence CC BY 4.0.
224 x 224 px patches of concrete bridge surfaces, labelled Crack / Background.
Files ending in "_r" are rotated copies of crack patches; they are EXCLUDED so no
image is counted twice.

Protocol (no learning, one tuned threshold):
* deterministic split by a hash of the file name: 30 % dev, 70 % test;
* the image-level score is the total centre-line length (px) of crack-shaped
  components found by SetuScan's classical segmentation (no marker in these
  patches, so pixels only);
* the decision threshold on that score is chosen on DEV to maximise F1 and then
  applied once to TEST. Only TEST numbers are headline numbers.

Also writes failure-case montages (highest-scoring backgrounds = false positives,
lowest-scoring cracks = false negatives) to eval/results/.

Usage: uv run python eval/eval_krakn.py [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import RESULTS, ROOT, prf, provenance, write_json  # noqa: I001
from setuscan.measure import skeleton_length, thin
from setuscan.segment import SegParams, segment

DATA = ROOT / "data" / "krakn"
TUNING = ROOT / "eval" / "results" / "krakn_dev_tuning.json"


def _patch_params() -> SegParams:
    """Detector settings for 224 px patches: the best DEV-split configuration, if tuned."""
    import json
    best = json.loads(TUNING.read_text())["best"]["params"] if TUNING.exists() else {}
    return SegParams(max_width_px=17, min_len_px=40.0, **best)


PATCH_PARAMS = _patch_params()


def split_of(name: str) -> str:
    return "dev" if int(hashlib.sha1(name.encode()).hexdigest(), 16) % 10 < 3 else "test"


def list_images() -> list[tuple[Path, int]]:
    crack = sorted(p for p in (DATA / "Crack" / "Crack").glob("*.jpg") if not p.stem.endswith("_r"))
    bg = sorted((DATA / "Background" / "Background").glob("*.jpg"))
    return [(p, 1) for p in crack] + [(p, 0) for p in bg]


def score_image(path: Path) -> tuple[float, int, float]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    t0 = time.perf_counter()
    seg = segment(img, None, PATCH_PARAMS)
    length = skeleton_length(thin(seg.mask)) if seg.mask.any() else 0.0
    return length, len(seg.components), (time.perf_counter() - t0) * 1000


def _job(item):
    p, y = item
    s, n, ms = score_image(p)
    return {"file": p.name, "label": y, "split": split_of(p.name), "score": s, "n_components": n, "ms": ms}


def best_threshold(rows: list[dict]) -> float:
    scores = np.array([r["score"] for r in rows])
    labels = np.array([r["label"] for r in rows])
    cands = np.unique(np.concatenate([[0.0], scores]))
    best, best_f1 = 0.0, -1.0
    for t in cands:
        pred = scores > t
        tp = int((pred & (labels == 1)).sum()); fp = int((pred & (labels == 0)).sum())
        fn = int((~pred & (labels == 1)).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def metrics(rows: list[dict], t: float) -> dict:
    tp = sum(1 for r in rows if r["score"] > t and r["label"] == 1)
    fp = sum(1 for r in rows if r["score"] > t and r["label"] == 0)
    fn = sum(1 for r in rows if r["score"] <= t and r["label"] == 1)
    tn = sum(1 for r in rows if r["score"] <= t and r["label"] == 0)
    return prf(tp, fp, fn, tn)


def average_precision(rows: list[dict]) -> float:
    r = sorted(rows, key=lambda x: -x["score"])
    pos = sum(x["label"] for x in r)
    tp, ap = 0, 0.0
    for i, x in enumerate(r, 1):
        if x["label"]:
            tp += 1
            ap += tp / i
    return round(ap / max(1, pos), 4)


def montage(rows: list[dict], title: str, out: Path, n: int = 12) -> None:
    tiles = []
    for r in rows[:n]:
        sub = "Crack" if r["label"] else "Background"
        img = cv2.imread(str(DATA / sub / sub / r["file"]))
        seg = segment(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None, PATCH_PARAMS)
        ov = img.copy()
        ov[seg.mask > 0] = (0, 140, 255)
        tile = np.hstack([img, ov])
        cv2.rectangle(tile, (0, 0), (448, 22), (20, 20, 20), -1)
        cv2.putText(tile, f"{r['file']}  score {r['score']:.0f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows_img = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    grid = np.vstack(rows_img)
    head = np.full((40, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(head, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([head, grid]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out.relative_to(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per class")
    args = ap.parse_args()
    items = list_images()
    if args.limit:
        items = [x for x in items if x[1] == 1][: args.limit] + [x for x in items if x[1] == 0][: args.limit]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(_job, items, chunksize=64))
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    t = best_threshold(dev)
    sweep = []
    for thr in sorted({0.0, 20.0, 40.0, 60.0, 80.0, 120.0, 160.0, t}):
        m = metrics(test, thr)
        sweep.append({"threshold_px": thr, **{k: m[k] for k in ("precision", "recall", "f1")}})
    res = {
        "provenance": provenance("eval/eval_krakn.py"),
        "dataset": {"name": "KrakN associated dataset", "doi": "10.5281/zenodo.5266356",
                    "url": "https://zenodo.org/records/5266356", "license": "CC BY 4.0",
                    "authors": "Mateusz Zarski, Bartosz Wojcik, Jaroslaw Adam Miszczak",
                    "note": "224x224 patches; '_r' rotated duplicates of crack patches excluded"},
        "method": {"detector": "classical OpenCV (black-hat depth x Hessian ridge, hysteresis, shape filter)",
                   "params": PATCH_PARAMS.__dict__, "score": "total skeleton length of kept components (px)",
                   "threshold_chosen_on": "dev split (max F1)", "learned_weights": None,
                   "params_chosen_on": "dev split, see eval/results/krakn_dev_tuning.json"},
        "counts": {"dev": {"crack": sum(r["label"] for r in dev), "background": sum(1 - r["label"] for r in dev)},
                   "test": {"crack": sum(r["label"] for r in test), "background": sum(1 - r["label"] for r in test)}},
        "threshold_px": t,
        "dev": metrics(dev, t),
        "test": {**metrics(test, t), "average_precision": average_precision(test)},
        "test_threshold_sweep": sweep,
        "runtime_ms_per_image": {"mean": round(float(np.mean([r["ms"] for r in rows])), 2),
                                 "p95": round(float(np.percentile([r["ms"] for r in rows], 95)), 2)},
        "wall_time_s": round(time.time() - t0, 1),
    }
    fps = sorted([r for r in test if r["label"] == 0 and r["score"] > t], key=lambda r: -r["score"])
    fns = sorted([r for r in test if r["label"] == 1 and r["score"] <= t], key=lambda r: r["score"])
    res["failure_examples"] = {"false_positives": [r["file"] for r in fps[:12]],
                               "false_negatives": [r["file"] for r in fns[:12]],
                               "montages": ["eval/results/krakn_false_positives.jpg",
                                            "eval/results/krakn_false_negatives.jpg"]}
    write_json("krakn_classification.json", res)
    (RESULTS / "krakn_scores.csv").write_text(
        "file,label,split,score,n_components\n" +
        "\n".join(f"{r['file']},{r['label']},{r['split']},{r['score']:.1f},{r['n_components']}" for r in rows))
    montage(fps, "KrakN test false positives (Background patches flagged as crack); left raw, right mask",
            RESULTS / "krakn_false_positives.jpg")
    montage(fns, "KrakN test false negatives (Crack patches missed); left raw, right mask",
            RESULTS / "krakn_false_negatives.jpg")
    print("threshold", t, "test", res["test"])


if __name__ == "__main__":
    main()
