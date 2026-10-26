"""Download the public crack datasets used by the evaluation harness into data/.

Only datasets with a clear, verified licence are fetched automatically:

* KrakN associated dataset (Zarski, Wojcik, Miszczak, 2021), Zenodo record 5266356,
  DOI 10.5281/zenodo.5266356, licence CC BY 4.0 (from the Zenodo record metadata).

SDNET2018 (Utah State University, DOI 10.15142/T3TD19, CC BY 4.0 per the dataset page)
is supported by the evaluation script, but its file host serves a browser challenge
to scripted downloads, so it must be downloaded by hand from
https://digitalcommons.usu.edu/all_datasets/48/ and unzipped into data/sdnet2018/.

Usage:  python scripts/download_data.py
Stdlib only, so it runs before the project dependencies are installed.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

KRAKN = {
    "name": "KrakN associated dataset",
    "record": "https://zenodo.org/records/5266356",
    "doi": "10.5281/zenodo.5266356",
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "authors": ["Mateusz Zarski", "Bartosz Wojcik", "Jaroslaw Adam Miszczak"],
    "files": {
        "Crack.zip": "1466df0503574aecb00da4fe5d2699be",
        "Background.zip": "6988bcffef44210ebb3189d2a8d049a6",
    },
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, expected_md5: str) -> None:
    if dest.exists() and md5(dest) == expected_md5:
        print(f"ok (cached) {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("ab" if start else "wb") as out:
        if start and r.status != 206:  # server ignored the range: restart
            out.truncate(0)
        while chunk := r.read(1 << 16):
            out.write(chunk)
    got = md5(tmp)
    if got != expected_md5:
        tmp.unlink()
        raise SystemExit(f"MD5 mismatch for {dest.name}: {got} != {expected_md5}")
    tmp.rename(dest)
    print(f"ok {dest.name} md5={got}")


def main() -> int:
    out = DATA / "krakn"
    out.mkdir(parents=True, exist_ok=True)
    for name, digest in KRAKN["files"].items():
        z = out / name
        fetch(f"https://zenodo.org/api/records/5266356/files/{name}/content", z, digest)
        target = out / z.stem
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                for member in zf.namelist():  # refuse path traversal
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise SystemExit(f"unsafe path in zip: {member}")
                zf.extractall(target)
    (out / "SOURCE.json").write_text(json.dumps(KRAKN, indent=2))
    print("KrakN ready in", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
