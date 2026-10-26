"""SetuScan local web API (FastAPI). Phase 2 runs the same handlers behind
API Gateway + Lambda, with Step Functions driving the agent loop.

Run:  uv run uvicorn setuscan.api:app --port 8765
"""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import agent
from .deciders import RulesDecider
from .pipeline import decode_image
from .quality import resize_max
from .store import LocalStore

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SAMPLES = WEB / "samples"
if os.environ.get("SETUSCAN_BUCKET"):  # AWS Lambda: S3-backed store
    from .store import S3Store
    STORE = S3Store(Path(os.environ.get("SETUSCAN_RUNS", "/tmp/runs")), os.environ["SETUSCAN_BUCKET"])
else:
    STORE = LocalStore(Path(os.environ.get("SETUSCAN_RUNS", ROOT / "runs")))
MAX_UPLOAD = 12 * 1024 * 1024
RATE = (20, 600)  # at most 20 inspections per 10 minutes per client
DECIDER = RulesDecider()

app = FastAPI(title="SetuScan", docs_url="/api/docs", redoc_url=None)
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
    return resp


def _rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "anon"
    now = time.time()
    with _hits_lock:
        q = _hits[key]
        while q and q[0] < now - RATE[1]:
            q.popleft()
        if len(q) >= RATE[0]:
            raise HTTPException(429, "Too many inspections from this client; try again in a few minutes.")
        q.append(now)


def _samples() -> list[dict]:
    idx = SAMPLES / "index.json"
    return json.loads(idx.read_text()) if idx.exists() else []


def _response(case: agent.Case | None, rec: dict) -> dict:
    rec = dict(rec)
    rec["image_urls"] = {n: f"/api/cases/{rec['id']}/images/{n}.jpg" for n in rec.get("images", [])}
    return rec


def _run(current, previous, labels: dict) -> dict:
    case = agent.new_case(current, previous, labels=labels)
    case.images["current_input"] = resize_max(current, 1600)[0]
    if previous is not None:
        case.images["previous_input"] = resize_max(previous, 1600)[0]
    agent.run(case, DECIDER)
    return _response(case, STORE.save(case))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((WEB / "index.html").read_text())


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "opencv": cv2.__version__, "python": platform.python_version(),
            "decider": DECIDER.name, "decider_kind": "deterministic rules (no LLM)",
            "storage": "S3 (1-day lifecycle)" if os.environ.get("SETUSCAN_BUCKET") else "local folder",
            "runtime": "AWS Lambda" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "local"}


@app.get("/api/samples")
def samples() -> list[dict]:
    return [{k: s[k] for k in ("id", "title", "label", "description", "files")} for s in _samples()]


@app.get("/api/samples/{sample_id}/{fname}")
def sample_file(sample_id: str, fname: str):
    s = next((s for s in _samples() if s["id"] == sample_id), None)
    if s is None or fname not in s["files"].values():
        raise HTTPException(404)
    return FileResponse(SAMPLES / fname, media_type="image/jpeg")


@app.post("/api/inspect_sample/{sample_id}")
def inspect_sample(sample_id: str, request: Request) -> dict:
    _rate_limit(request)
    s = next((s for s in _samples() if s["id"] == sample_id), None)
    if s is None:
        raise HTTPException(404, "unknown sample")
    cur = cv2.imread(str(SAMPLES / s["files"]["current"]))
    prev = cv2.imread(str(SAMPLES / s["files"]["previous"])) if "previous" in s["files"] else None
    return _run(cur, prev, {"source": s["label"], "sample": s["id"], "title": s["title"]})


async def _read(f: UploadFile | None):
    if f is None or not f.filename:
        return None
    data = await f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Photo is larger than 12 MB.")
    try:
        return decode_image(data)
    except ValueError:
        raise HTTPException(415, "Could not read that file as a photo (JPEG, PNG or WebP).")


@app.post("/api/inspect")
async def inspect(request: Request, current: UploadFile = File(...), previous: UploadFile | None = File(None)) -> dict:
    _rate_limit(request)
    cur = await _read(current)
    if cur is None:
        raise HTTPException(400, "current photo is required")
    prev = await _read(previous)
    return _run(cur, prev, {"source": "User upload"})


@app.get("/api/cases/{case_id}")
def get_case(case_id: str) -> dict:
    try:
        return _response(None, STORE.record(case_id))
    except KeyError:
        raise HTTPException(404, "case not found (cases are deleted after 24 h)")


@app.get("/api/cases/{case_id}/images/{name}.jpg")
def case_image(case_id: str, name: str):
    try:
        return FileResponse(STORE.image_path(case_id, name), media_type="image/jpeg")
    except KeyError:
        raise HTTPException(404)


def _decision(case_id: str, approve: bool, reviewer: str, note: str) -> dict:
    case = STORE.get(case_id)
    if case is None:
        raise HTTPException(404, "case not found or expired; run the inspection again")
    try:
        (agent.approve if approve else agent.reject)(case, reviewer, note)
    except ValueError as e:
        raise HTTPException(409, str(e))
    return _response(case, STORE.save(case))


@app.post("/api/cases/{case_id}/approve")
def approve(case_id: str, reviewer: str = Form(...), note: str = Form("")) -> dict:
    return _decision(case_id, True, reviewer, note)


@app.post("/api/cases/{case_id}/reject")
def reject(case_id: str, reviewer: str = Form(...), note: str = Form("")) -> dict:
    return _decision(case_id, False, reviewer, note)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):  # never leak stack traces to the browser
    return JSONResponse({"detail": "internal error"}, status_code=500)
