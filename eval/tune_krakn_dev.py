"""Parameter sweep for the KrakN crack/no-crack detector on the DEV split ONLY.

The test split is never scored here. The best configuration (by dev F1) is written
to eval/results/krakn_dev_tuning.json and then used, unchanged, by eval_krakn.py.

Usage: uv run python eval/tune_krakn_dev.py
"""

from __future__ import annotations

import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import cv2

from common import provenance, write_json  # noqa: I001
from eval_krakn import best_threshold, list_images, metrics, split_of
from setuscan.measure import skeleton_length, thin
from setuscan.segment import SegParams, segment

GRID = {
    "min_contrast": [3.0, 6.0],
    "k_high": [4.0, 5.0, 7.0],
    "max_asymmetry": [None, 0.10, 0.13],
    "min_elongation": [10.0, 14.0],
}


def _score(args):
    path, params = args
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    seg = segment(img, None, SegParams(max_width_px=17, min_len_px=40.0, **params))
    return skeleton_length(thin(seg.mask)) if seg.mask.any() else 0.0


def main() -> None:
    dev = [(p, y) for p, y in list_images() if split_of(p.name) == "dev"]
    keys = list(GRID)
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        for combo in itertools.product(*GRID.values()):
            params = dict(zip(keys, combo))
            scores = list(ex.map(_score, [(p, params) for p, _ in dev], chunksize=64))
            rows = [{"label": y, "score": s} for (_, y), s in zip(dev, scores)]
            t = best_threshold(rows)
            m = metrics(rows, t)
            out.append({"params": params, "threshold_px": t, **m})
            print(params, t, m["precision"], m["recall"], m["f1"], flush=True)
    out.sort(key=lambda r: -r["f1"])
    write_json("krakn_dev_tuning.json", {"provenance": provenance("eval/tune_krakn_dev.py"),
                                         "split": "dev only", "n_dev": len(dev), "grid": GRID,
                                         "best": out[0], "all": out, "wall_time_s": round(time.time() - t0, 1)})


if __name__ == "__main__":
    main()
