"""Private smoke test: invoke the deployed Lambda directly (no public endpoint) with API Gateway v2 events."""
import base64, json, os, subprocess, sys, tempfile
AWS = os.path.expanduser("~/.local/bin/aws"); REGION = os.environ.get("AWS_DEFAULT_REGION", "ap-southeast-2")

def call(method, path, body=b"", ctype=None):
    h = {"host": "x"}
    if ctype: h["content-type"] = ctype
    ev = {"version": "2.0", "routeKey": "$default", "rawPath": path, "rawQueryString": "", "headers": h,
          "requestContext": {"http": {"method": method, "path": path, "sourceIp": "127.0.0.1", "userAgent": "smoke"}, "stage": "$default"},
          "body": base64.b64encode(body).decode(), "isBase64Encoded": True}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f: json.dump(ev, f)
    out = f.name + ".out"
    subprocess.run([AWS, "lambda", "invoke", "--region", REGION, "--function-name", "setuscan-api", "--payload", f"fileb://{f.name}",
                    "--cli-binary-format", "raw-in-base64-out", out], check=True, capture_output=True)
    r = json.load(open(out)); return r["statusCode"], r.get("body", "")

fails = []
def check(name, ok, info=""):
    print(("PASS " if ok else "FAIL ") + name, info); ok or fails.append(name)

s, b = call("GET", "/"); check("home page", s == 200 and "SetuScan" in b)
s, b = call("GET", "/api/health"); h = json.loads(b); check("health: OpenCV 5 on Lambda", s == 200 and h["opencv"].startswith("5.") and h["runtime"] == "AWS Lambda", h)
want = {"wide": "awaiting_approval", "thin": "closed", "blurry": "needs_reshoot"}
for sid, st in want.items():
    s, b = call("POST", f"/api/inspect_sample/{sid}")
    c = json.loads(b) if s == 200 else {}
    check(f"sample {sid}", s == 200 and c.get("status") == st, (s, c.get("status"), c.get("labels", {}).get("agent_total_ms")))
s, b = call("POST", "/api/inspect_sample/nope"); check("error: unknown sample -> 404", s == 404)
bd = b"--B\r\nContent-Disposition: form-data; name=\"current\"; filename=\"x.jpg\"\r\nContent-Type: image/jpeg\r\n\r\nnot an image\r\n--B--\r\n"
s, b = call("POST", "/api/inspect", bd, "multipart/form-data; boundary=B"); check("error: non-image upload -> 415", s == 415, s)
print("FAILED:" if fails else "ALL PASSED", fails); sys.exit(1 if fails else 0)
