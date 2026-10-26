# Live AWS evaluation (SetuScan)

Endpoint: `https://l4734hezca.execute-api.ap-southeast-2.amazonaws.com` (API Gateway HTTP API -> Lambda container, OpenCV 5.0.0, ap-southeast-2).  
Run: 2026-10-25T23:48:15.191055Z UTC from a laptop in India (wall times include the network round trip to Sydney), against the build deployed on 26 Oct 2026 (adds median-width columns and "W med / p95" overlay labels).  
Script: `eval/eval_live_aws.py` (stdlib only; this file is rendered from the JSON by `eval/make_live_md.py`). Raw data: `eval/live-aws.json`. API calls made in this run: 23 (the app limits one client to 20 inspections per 10 minutes, which is why repeats are 2-3).

All samples are the labelled sample set shipped in `web/samples/` (SYNTHETIC renders plus one unmodified KrakN patch). No real-photo data was used.

## Per-sample outcome and latency

| Sample | Expected outcome | Live outcome (all runs) | Agent steps | Wall ms (min / median / max) | Agent-internal ms (median) |
|---|---|---|---|---|---|
| wide | awaiting_approval | awaiting_approval (match) x3 | 5 | 2518 / 2574 / 2780 | 1145.2 |
| thin | closed | closed (match) x3 | 6 | 3239 / 3254 / 3258 | 1716.6 |
| blurry | needs_reshoot | needs_reshoot (match) x3 | 3 | 1461 / 1468 / 1559 | 103.3 |
| nomarker | needs_reshoot | needs_reshoot (match) x2 | 4 | 2433 / 2471 / 2509 | 1003.5 |
| krakn | needs_reshoot | needs_reshoot (match) x2 | 4 | 1424 / 1444 / 1464 | 109.4 |
| growth | awaiting_approval | awaiting_approval (match) x2 | 10 | 6430 / 6433 / 6436 | 4734.0 |

Outcome match: **6 of 6** samples gave the expected terminal status on every run, matching the sample descriptions in `web/samples/index.json`. Every step in every trace was decided by `rules` (no LLM).

## Latency

* This run was warm (the Lambda had just been redeployed and smoke-tested), so it contains no cold-start measurement. `/api/health`: 2860 ms. First inspection (`wide`): 2780 ms wall.
* Cold-start figures from the earlier run of this script on the previous build (same measurement code): about 4 s for the first request after more than 10 minutes idle and 6.5 s for the first inspection (not re-run on the final build).
* All 15 inspections in this run: min 1424 ms, median 2518 ms, max 6436 ms wall.
* The slowest case is the two-visit `growth` sample (SIFT + RANSAC on two photos): 6.4-6.4 s wall, 4.7-4.8 s inside the agent.
* The Lambda is 2 GB / 60 s; wall time was never close to the limit. No request failed or timed out.

## Real-upload latency (live, uploaded files)

* One photo (`POST /api/inspect`): 4.4 s wall.
* Two visits (SIFT + RANSAC alignment): 8.5 s wall.
* Uploads are slower than the labelled samples because the image bytes travel from India to Sydney and are decoded server-side; both are single measurements, not medians.

## Width metric: median vs p95 (live, from the JSON)

Accuracy in the technical report is for the **median** width along the crack. The severity policy and the overlay headline use the **p95** width (near-maximum), which reads higher by design. Compare a known true width with the median.

| Sample | True width (synthetic truth) | Live median width | Live p95 width |
|---|---|---|---|
| wide | 2.4 mm | 2.41 mm | 2.54 mm |
| thin | 0.35 mm | 0.37 mm | 0.41 mm |

## Upload path and the human approval gate (live)

| Check | Result |
|---|---|
| Upload one photo (`POST /api/inspect`, wide sample file) | HTTP 200, awaiting_approval, 4400 ms |
| Upload two visits (growth sample files) | HTTP 200, awaiting_approval, 8545 ms; aligned, crack length 84.5 -> 135.7 mm, extension 47.1 mm, width median 0.49 -> 1.08 mm, p95 0.58 -> 1.16 mm, growth flagged: True (synthetic truth: 85 -> 135 mm, 0.5 -> 1.1 mm true width) |
| Ticket before approval | None (none) |
| Approve by a named reviewer | HTTP 200, status `ticket_raised`, ticket created: True |
| Approve the same case again | HTTP 409 (refused) |
| Unknown sample id | HTTP 404 |
| Non-image upload | HTTP 415 |
| Approve a case that does not exist | HTTP 404 |

The evaluation ticket was approved under the reviewer name "Harsha Shinde (live eval)" and expires with the case after 24 h.

## Failure cases and caveats

* `krakn` (a 224 px dataset patch with no marker) and `nomarker` end in `needs_reshoot`: the agent finds the crack in pixels but refuses to report millimetres without a scale. This is the designed behaviour, not a detection failure.
* `blurry` ends in `needs_reshoot` because the photo-level checks pass but the marker's printed edges are blurred (edge blur above the 2.5 px limit), which would over-read crack widths.
* `thin` ends `closed` after zooming to native resolution. It is a synthetic 0.35 mm crack; accuracy on thin cracks is limited (see the limits section of the technical report).
* This run measures one client, one region, sequential requests. It is not a load test; the API is throttled to 5 requests per second by design.
