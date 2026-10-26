"""Live AWS evaluation: calls the public endpoint, records per-sample outcome and latency.
Usage: python3 eval/eval_live_aws.py   (stdlib only; ~50 API calls total, sequential, >=0.4 s apart)"""
import urllib.parse, json, statistics as st, sys, time, urllib.request, urllib.error, subprocess, os, uuid, datetime
URL = os.environ.get("URL", "https://l4734hezca.execute-api.ap-southeast-2.amazonaws.com")
SAMPLES = ["wide", "thin", "blurry", "nomarker", "krakn", "growth"]
REPEATS = {'wide': 3, 'thin': 3, 'blurry': 3, 'nomarker': 2, 'krakn': 2, 'growth': 2}  # app limit: 20 inspections / 10 min / client
calls = 0

def req(method, path, data=None, headers=None):
    global calls
    calls += 1
    r = urllib.request.Request(URL + path, data=data, method=method, headers=headers or {})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=90) as resp:
            body = resp.read(); code = resp.status
    except urllib.error.HTTPError as e:
        body = e.read(); code = e.code
    ms = (time.perf_counter() - t) * 1000
    time.sleep(0.4)
    return code, body, ms

def summ(v):
    v = sorted(v); return {"n": len(v), "min": round(v[0], 1), "median": round(st.median(v), 1), "max": round(v[-1], 1)}

out = {"endpoint": URL, "run_utc": datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).isoformat() + "Z", "region": "ap-southeast-2",
       "client": "WSL laptop (India) -> Sydney, includes network RTT", "samples": {}, "notes": []}
code, body, ms = req("GET", "/api/health")
out["health"] = {"status": code, "latency_ms": round(ms, 1), "body": json.loads(body)}
for sid in SAMPLES:
    runs = []
    for i in range(REPEATS[sid]):
        code, body, ms = req("POST", f"/api/inspect_sample/{sid}")
        d = json.loads(body) if code == 200 else {}
        cl = d.get("classification") or {}
        runs.append({"http": code, "wall_ms": round(ms, 1), "agent_total_ms": d.get("labels", {}).get("agent_total_ms"),
                     "status": d.get("status"), "steps": len(d.get("trace", [])),
                     "actions": [s["action"] for s in d.get("trace", [])], "deciders": sorted({s["decider"] for s in d.get("trace", [])}),
                     "widths_mm": [{k: c.get(k) for k in ("id", "width_median_mm", "width_max_mm", "length_mm")} for c in (d.get("visits", {}).get("current", {}).get("cracks") or [])],
                     "classification": {k: cl.get(k) for k in ("band", "max_width_mm", "length_mm", "label") if k in cl} or cl})
    stat = {r["status"] for r in runs}
    out["samples"][sid] = {"runs": runs, "statuses": sorted(map(str, stat)), "consistent": len(stat) == 1,
                           "wall_ms": summ([r["wall_ms"] for r in runs]),
                           "agent_ms": summ([r["agent_total_ms"] for r in runs if r["agent_total_ms"] is not None]) if all(r["agent_total_ms"] for r in runs) else None}
# upload path: multipart with a real sample file
def multipart(files):
    b = uuid.uuid4().hex; parts = []
    for name, path in files.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{name}"; filename="{os.path.basename(path)}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + open(path, "rb").read() + b"\r\n")
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"
S = "web/samples/"
up = {}
data, ct = multipart({"current": S + "wide.jpg"})
code, body, ms = req("POST", "/api/inspect", data, {"Content-Type": ct}); d = json.loads(body) if code == 200 else {}
up["upload_single_wide"] = {"http": code, "wall_ms": round(ms, 1), "status": d.get("status"),
                         "widths_mm": [{k: c.get(k) for k in ("id", "width_median_mm", "width_max_mm")} for c in (d.get("visits", {}).get("current", {}).get("cracks") or [])]}
data, ct = multipart({"current": S + "growth_now.jpg", "previous": S + "growth_before.jpg"})
code, body, ms = req("POST", "/api/inspect", data, {"Content-Type": ct}); d = json.loads(body) if code == 200 else {}
up["upload_two_visit_growth"] = {"http": code, "wall_ms": round(ms, 1), "status": d.get("status"), "comparison": {k: v for k, v in (d.get("comparison") or {}).items() if isinstance(v, (int, float, str, bool))}}
case_id = d.get("id")
# human gate: attempt approve on a case, check ticket only after approval
if case_id and d.get("status") == "awaiting_approval":
    up["ticket_before_approval"] = d.get("ticket")
    fm = urllib.parse.quote  # noqa
    code, body, ms = req("POST", f"/api/cases/{case_id}/approve", b"reviewer=Harsha+Shinde+%28live+eval%29&note=live+eval+run", {"Content-Type": "application/x-www-form-urlencoded"})
    a = json.loads(body) if code == 200 else {}
    up["approve"] = {"http": code, "wall_ms": round(ms, 1), "status": a.get("status"), "ticket_created": bool(a.get("ticket"))}
    code, body, ms = req("POST", f"/api/cases/{case_id}/approve", b"reviewer=x", {"Content-Type": "application/x-www-form-urlencoded"})
    up["approve_twice"] = {"http": code}
code, body, ms = req("POST", "/api/inspect_sample/nope"); up["unknown_sample"] = {"http": code}
bd = b"--B\r\nContent-Disposition: form-data; name=\"current\"; filename=\"x.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nnot an image\r\n--B--\r\n"
code, body, ms = req("POST", "/api/inspect", bd, {"Content-Type": "multipart/form-data; boundary=B"}); up["non_image"] = {"http": code}
code, body, ms = req("POST", "/api/cases/doesnotexist/approve", b"reviewer=x", {"Content-Type": "application/x-www-form-urlencoded"}); up["approve_missing_case"] = {"http": code}
out["upload_and_gate"] = up
allw = [r["wall_ms"] for s in out["samples"].values() for r in s["runs"]]
out["overall_wall_ms"] = summ(allw); out["first_call_this_session_ms"] = out["samples"][SAMPLES[0]]["runs"][0]["wall_ms"]
out["api_calls_made"] = calls
json.dump(out, open("eval/live-aws.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("overall_wall_ms", "api_calls_made")}))
for sid, s in out["samples"].items(): print(sid, s["statuses"], s["consistent"], s["wall_ms"], s["agent_ms"])
print(json.dumps(up, indent=0)[:1500])
