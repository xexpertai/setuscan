"""Case storage. Phase 1: local folder. Phase 2: S3 (images) + DynamoDB (case records).

Only re-encoded pixels are stored (no original file bytes), so EXIF metadata such as
GPS location from the phone never reaches storage. Cases older than RETENTION_H are
deleted on the next write (phase 2 uses an S3 lifecycle rule and DynamoDB TTL).
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .agent import Case, case_to_dict

RETENTION_H = 24
SAFE = re.compile(r"^[a-z0-9_]{1,40}$")


class LocalStore:
    def __init__(self, root: Path, keep_in_memory: int = 64):
        self.root = Path(root)
        (self.root / "cases").mkdir(parents=True, exist_ok=True)
        (self.root / "tickets").mkdir(parents=True, exist_ok=True)
        self._mem: OrderedDict[str, Case] = OrderedDict()
        self._keep = keep_in_memory
        self._lock = threading.Lock()

    def case_dir(self, case_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", case_id):
            raise KeyError(case_id)
        return self.root / "cases" / case_id

    def save(self, case: Case) -> dict:
        d = self.case_dir(case.id)
        d.mkdir(parents=True, exist_ok=True)
        for name, img in case.images.items():
            if SAFE.match(name) and not (d / f"{name}.jpg").exists():
                cv2.imwrite(str(d / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rec = case_to_dict(case)
        (d / "case.json").write_text(json.dumps(rec, indent=1, default=str))
        if case.ticket:
            (self.root / "tickets" / f"{case.ticket['ticket_id']}.json").write_text(json.dumps(case.ticket, indent=1))
        with self._lock:
            self._mem[case.id] = case
            self._mem.move_to_end(case.id)
            while len(self._mem) > self._keep:
                self._mem.popitem(last=False)
        self.purge_old()
        return rec

    def get(self, case_id: str) -> Case | None:
        with self._lock:
            return self._mem.get(case_id)

    def record(self, case_id: str) -> dict:
        p = self.case_dir(case_id) / "case.json"
        if not p.exists():
            raise KeyError(case_id)
        return json.loads(p.read_text())

    def image_path(self, case_id: str, name: str) -> Path:
        if not SAFE.match(name):
            raise KeyError(name)
        p = self.case_dir(case_id) / f"{name}.jpg"
        if not p.exists():
            raise KeyError(name)
        return p

    def purge_old(self) -> int:
        cutoff = time.time() - RETENTION_H * 3600
        n = 0
        for d in (self.root / "cases").iterdir():
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n


def encode_jpeg(img: np.ndarray, quality: int = 88) -> bytes:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return enc.tobytes()


class S3Store(LocalStore):
    """Lambda store: LocalStore in /tmp as a cache, S3 as the durable copy (bucket lifecycle: 1 day).

    Lambda instances do not share /tmp, so a case saved by one instance is re-hydrated from S3
    when another instance serves the approve/reject call or the image requests.
    """

    def __init__(self, root: Path, bucket: str, keep_in_memory: int = 64):
        super().__init__(root, keep_in_memory)
        import boto3
        self.bucket = bucket
        self.s3 = boto3.client("s3")

    def _key(self, case_id: str, fname: str) -> str:
        return f"cases/{case_id}/{fname}"

    def save(self, case: Case) -> dict:
        base = getattr(case, "_base", None)
        if base is not None:  # re-hydrated case: keep the stored visits/images, update decision fields
            rec = dict(base)
            rec.update(status=case.status, approval=case.approval, ticket=case.ticket, trace=case.trace)
            d = self.case_dir(case.id)
            d.mkdir(parents=True, exist_ok=True)
            (d / "case.json").write_text(json.dumps(rec, indent=1, default=str))
            self.s3.put_object(Bucket=self.bucket, Key=self._key(case.id, "case.json"),
                               Body=json.dumps(rec, default=str).encode(), ContentType="application/json")
            if case.ticket:
                self.s3.put_object(Bucket=self.bucket, Key=f"tickets/{case.ticket['ticket_id']}.json",
                                   Body=json.dumps(case.ticket).encode(), ContentType="application/json")
            return rec
        rec = super().save(case)
        d = self.case_dir(case.id)
        for f in d.iterdir():
            self.s3.upload_file(str(f), self.bucket, self._key(case.id, f.name),
                                ExtraArgs={"ContentType": "application/json" if f.suffix == ".json" else "image/jpeg"})
        if case.ticket:
            self.s3.put_object(Bucket=self.bucket, Key=f"tickets/{case.ticket['ticket_id']}.json",
                               Body=json.dumps(case.ticket).encode(), ContentType="application/json")
        return rec

    def _pull(self, case_id: str, fname: str) -> bool:
        p = self.case_dir(case_id) / fname
        if p.exists():
            return True
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, self._key(case_id, fname), str(p))
            return True
        except Exception:
            return False

    def record(self, case_id: str) -> dict:
        if not self._pull(case_id, "case.json"):
            raise KeyError(case_id)
        return super().record(case_id)

    def image_path(self, case_id: str, name: str) -> Path:
        if not SAFE.match(name):
            raise KeyError(name)
        self._pull(case_id, f"{name}.jpg")
        return super().image_path(case_id, name)

    def get(self, case_id: str) -> Case | None:
        case = super().get(case_id)
        if case is not None:
            return case
        try:
            rec = self.record(case_id)
        except KeyError:
            return None
        from types import SimpleNamespace
        from .agent import SeverityPolicy
        comp = rec.get("comparison")
        case = Case(id=case_id, visits={}, policy=SeverityPolicy(), status=rec["status"], trace=rec["trace"],
                    comparison=SimpleNamespace(**comp) if comp else None, classification=rec.get("classification"),
                    reshoot=rec.get("reshoot"), approval=rec.get("approval"), ticket=rec.get("ticket"),
                    created=rec["created"], labels=rec.get("labels", {}))
        case._base = rec  # type: ignore[attr-defined]
        return case
