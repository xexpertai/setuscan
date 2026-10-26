"""Build the demo samples shown in the web UI (web/samples/).

Every sample is labelled with where it comes from:
* "SYNTHETIC sample": rendered by setuscan.synth (not a real photo);
* "Public dataset: KrakN (CC BY 4.0)": a patch from the KrakN dataset, credited in
  web/samples/ATTRIBUTION.md. Requires `python scripts/download_data.py` first.

Usage: uv run python scripts/make_samples.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from setuscan.synth import CrackSpec, SceneSpec, default_crack, render_scene, two_visits  # noqa: E402

OUT = ROOT / "web" / "samples"
SYN = "SYNTHETIC sample"
KRAKN = "Public dataset: KrakN (CC BY 4.0)"


def save(name: str, img: np.ndarray) -> str:
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return name


def copy_original(src: Path, name: str) -> str:
    shutil.copyfile(src, OUT / name)  # byte-for-byte copy of the dataset file
    return name


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = []

    rng = np.random.default_rng(41)
    img, gt = render_scene(SceneSpec(seed=41, cracks=[default_crack(rng, width_mm=2.4, length_mm=140)]))
    samples.append({"id": "wide", "title": "Wide crack on a wall", "label": SYN,
                    "description": f"Rendered crack, true width {gt['cracks'][0]['width_max_mm']} mm. "
                                   "Expect: measured, sent to an engineer for approval.",
                    "files": {"current": save("wide.jpg", img)}, "truth": {k: v for k, v in gt.items() if k != "mask"}})

    rng = np.random.default_rng(42)
    img, gt = render_scene(SceneSpec(seed=42, cracks=[default_crack(rng, width_mm=0.35, length_mm=150)],
                                     cam_ppm=10.0, wall_mm=(300.0, 220.0)))
    samples.append({"id": "thin", "title": "Thin crack, wide shot", "label": SYN,
                    "description": "True width 0.35 mm (compare with the median width), only ~2 px in the working view. Expect: the agent zooms "
                                   "to native resolution before deciding.",
                    "files": {"current": save("thin.jpg", img)}, "truth": {k: v for k, v in gt.items() if k != "mask"}})

    (a, ga), (b, gb) = two_visits(43, 0.5, 1.1, 85, 135)
    samples.append({"id": "growth", "title": "Two visits: the crack grew", "label": SYN,
                    "description": "Same wall, two photos from different positions. Crack rendered 85 mm long, "
                                   "0.5 mm wide, then 135 mm, 1.1 mm. Expect: aligned, growth flagged, engineer asked.",
                    "files": {"current": save("growth_now.jpg", b), "previous": save("growth_before.jpg", a)},
                    "truth": {"previous": {k: v for k, v in ga.items() if k != "mask"},
                              "current": {k: v for k, v in gb.items() if k != "mask"}}})

    rng = np.random.default_rng(44)
    img, _ = render_scene(SceneSpec(seed=44, cracks=[default_crack(rng, width_mm=1.0)], motion_blur_px=15,
                                    blur_sigma=2.0))
    samples.append({"id": "blurry", "title": "Shaky photo", "label": SYN,
                    "description": "Motion blur. Expect: a re-shoot request instead of a measurement.",
                    "files": {"current": save("blurry.jpg", img)}})

    rng = np.random.default_rng(45)
    img, _ = render_scene(SceneSpec(seed=45, cracks=[default_crack(rng, width_mm=1.2)], marker_id=None))
    samples.append({"id": "nomarker", "title": "Forgot the marker", "label": SYN,
                    "description": "No scale marker in view. Expect: crack found in pixels, re-shoot with marker requested.",
                    "files": {"current": save("nomarker.jpg", img)}})

    kdir = ROOT / "data" / "krakn" / "Crack" / "Crack"
    picks = ["crack6482.jpg"]
    for name in picks:
        p = kdir / name
        if p.exists():
            samples.append({"id": "krakn", "title": "Real bridge-concrete patch", "label": KRAKN,
                            "description": f"KrakN dataset patch {name} (224 x 224 px, unmodified). No marker, so "
                                           "pixels only. Expect: crack found, marker re-shoot requested.",
                            "files": {"current": copy_original(p, "krakn_" + name)}})

    (OUT / "index.json").write_text(json.dumps(samples, indent=1))
    (OUT / "ATTRIBUTION.md").write_text(
        "# Sample attribution\n\n"
        "Files named `krakn_*.jpg` are from the **KrakN associated dataset** by Mateusz Zarski, Bartosz Wojcik and\n"
        "Jaroslaw Adam Miszczak (2021), https://doi.org/10.5281/zenodo.5266356, licensed under\n"
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The patch is unmodified.\n\n"
        "All other images in this folder are SYNTHETIC, rendered by `setuscan/synth.py` (MIT licence). They are\n"
        "not photos of real structures.\n")
    print("wrote", len(samples), "samples to", OUT)


if __name__ == "__main__":
    main()
