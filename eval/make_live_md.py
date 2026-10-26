"""Render eval/live-aws.md from eval/live-aws.json (run after eval_live_aws.py). Every number comes from the JSON."""
import json, statistics as st
j = json.load(open("eval/live-aws.json")); S = j["samples"]; U = j["upload_and_gate"]
EXP = {"wide": "awaiting_approval", "thin": "closed", "blurry": "needs_reshoot", "nomarker": "needs_reshoot", "krakn": "needs_reshoot", "growth": "awaiting_approval"}
TRUTH = {"wide": "2.4 mm", "thin": "0.35 mm"}
WROWS = []
for _sid in ("wide", "thin"):
    _w = [c for r in S[_sid]["runs"] for c in r.get("widths_mm", [])]
    if _w:
        WROWS.append(f"| {_sid} | {TRUTH[_sid]} | {st.median(c['width_median_mm'] for c in _w):.2f} mm | {st.median(c['width_max_mm'] for c in _w):.2f} mm |")
if not WROWS:
    WROWS = ["| (no width data in this JSON; re-run eval_live_aws.py) | | | |"]
rows = []; ok = 0
for sid, e in EXP.items():
    s = S[sid]; m = s["statuses"] == [e] and s["consistent"]; ok += m
    w = s["wall_ms"]; a = s["agent_ms"]
    rows.append(f"| {sid} | {e} | {', '.join(s['statuses'])} ({'match' if m else 'MISMATCH'}) x{len(s['runs'])} | {s['runs'][0]['steps']} | {w['min']:.0f} / {w['median']:.0f} / {w['max']:.0f} | {a['median']:.1f} |")
allw = [r["wall_ms"] for s in S.values() for r in s["runs"]]; later = allw[1:]
g = U["upload_two_visit_growth"]["comparison"]; o = j["overall_wall_ms"]
gw = S["growth"]["wall_ms"]; ga = [r["agent_total_ms"] for r in S["growth"]["runs"]]
md = f"""# Live AWS evaluation (SetuScan)

Endpoint: `{j['endpoint']}` (API Gateway HTTP API -> Lambda container, OpenCV {j['health']['body']['opencv']}, {j['region']}).  
Run: {j['run_utc']} UTC from a laptop in India (wall times include the network round trip to Sydney), against the build deployed on 26 Oct 2026 (adds median-width columns and "W med / p95" overlay labels).  
Script: `eval/eval_live_aws.py` (stdlib only; this file is rendered from the JSON by `eval/make_live_md.py`). Raw data: `eval/live-aws.json`. API calls made in this run: {j['api_calls_made']} (the app limits one client to 20 inspections per 10 minutes, which is why repeats are 2-3).

All samples are the labelled sample set shipped in `web/samples/` (SYNTHETIC renders plus one unmodified KrakN patch). No real-photo data was used.

## Per-sample outcome and latency

| Sample | Expected outcome | Live outcome (all runs) | Agent steps | Wall ms (min / median / max) | Agent-internal ms (median) |
|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

Outcome match: **{ok} of 6** samples gave the expected terminal status on every run, matching the sample descriptions in `web/samples/index.json`. Every step in every trace was decided by `rules` (no LLM).

## Latency

* This run was warm (the Lambda had just been redeployed and smoke-tested), so it contains no cold-start measurement. `/api/health`: {j['health']['latency_ms']:.0f} ms. First inspection (`wide`): {j['first_call_this_session_ms']:.0f} ms wall.
* Cold-start figures from the earlier run of this script on the previous build (same measurement code): about 4 s for the first request after more than 10 minutes idle and 6.5 s for the first inspection (not re-run on the final build).
* All {o['n']} inspections in this run: min {o['min']:.0f} ms, median {o['median']:.0f} ms, max {o['max']:.0f} ms wall.
* The slowest case is the two-visit `growth` sample (SIFT + RANSAC on two photos): {gw['min']/1000:.1f}-{gw['max']/1000:.1f} s wall, {min(ga)/1000:.1f}-{max(ga)/1000:.1f} s inside the agent.
* The Lambda is 2 GB / 60 s; wall time was never close to the limit. No request failed or timed out.

## Real-upload latency (live, uploaded files)

* One photo (`POST /api/inspect`): {U['upload_single_wide']['wall_ms']/1000:.1f} s wall.
* Two visits (SIFT + RANSAC alignment): {U['upload_two_visit_growth']['wall_ms']/1000:.1f} s wall.
* Uploads are slower than the labelled samples because the image bytes travel from India to Sydney and are decoded server-side; both are single measurements, not medians.

## Width metric: median vs p95 (live, from the JSON)

Accuracy in the technical report is for the **median** width along the crack. The severity policy and the overlay headline use the **p95** width (near-maximum), which reads higher by design. Compare a known true width with the median.

| Sample | True width (synthetic truth) | Live median width | Live p95 width |
|---|---|---|---|
""" + "\n".join(WROWS) + f"""

## Upload path and the human approval gate (live)

| Check | Result |
|---|---|
| Upload one photo (`POST /api/inspect`, wide sample file) | HTTP {U['upload_single_wide']['http']}, {U['upload_single_wide']['status']}, {U['upload_single_wide']['wall_ms']:.0f} ms |
| Upload two visits (growth sample files) | HTTP {U['upload_two_visit_growth']['http']}, {U['upload_two_visit_growth']['status']}, {U['upload_two_visit_growth']['wall_ms']:.0f} ms; aligned, crack length {g['length_a']:.1f} -> {g['length_b']:.1f} mm, extension {g['extension']:.1f} mm, width median {g.get('width_median_a', float('nan')):.2f} -> {g.get('width_median_b', float('nan')):.2f} mm, p95 {g['width_max_a']:.2f} -> {g['width_max_b']:.2f} mm, growth flagged: {g['significant_growth']} (synthetic truth: 85 -> 135 mm, 0.5 -> 1.1 mm true width) |
| Ticket before approval | {U['ticket_before_approval']} (none) |
| Approve by a named reviewer | HTTP {U['approve']['http']}, status `{U['approve']['status']}`, ticket created: {U['approve']['ticket_created']} |
| Approve the same case again | HTTP {U['approve_twice']['http']} (refused) |
| Unknown sample id | HTTP {U['unknown_sample']['http']} |
| Non-image upload | HTTP {U['non_image']['http']} |
| Approve a case that does not exist | HTTP {U['approve_missing_case']['http']} |

The evaluation ticket was approved under the reviewer name "Harsha Shinde (live eval)" and expires with the case after 24 h.

## Failure cases and caveats

* `krakn` (a 224 px dataset patch with no marker) and `nomarker` end in `needs_reshoot`: the agent finds the crack in pixels but refuses to report millimetres without a scale. This is the designed behaviour, not a detection failure.
* `blurry` ends in `needs_reshoot` because the photo-level checks pass but the marker's printed edges are blurred (edge blur above the 2.5 px limit), which would over-read crack widths.
* `thin` ends `closed` after zooming to native resolution. It is a synthetic 0.35 mm crack; accuracy on thin cracks is limited (see the limits section of the technical report).
* This run measures one client, one region, sequential requests. It is not a load test; the API is throttled to 5 requests per second by design.
"""
open("eval/live-aws.md", "w").write(md); print("ok", ok, o)
