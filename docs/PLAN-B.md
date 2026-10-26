# Plan B and the real-AWS path (updated 25 Oct, evening)

## UPDATE: new verdict (supersedes the "Verdict" below)
The org SCP is **region-locked**: the account's home region is **ap-southeast-2 (Sydney)**, where Lambda, ECR, S3, CloudFormation and API Gateway all work. The earlier "everything denied" probes only covered us-east-1, us-west-2, ap-south-1 and eu-west-1.
- **Deployed (private, phase 1) in ap-southeast-2:** CloudFormation stack `setuscan`: ECR image with OpenCV 5.0.0 (`opencv-contrib-python-headless==5.0.0.93`), Lambda `setuscan-api` (2 GB, 60 s), private S3 buckets (cases with 1-day expiry, UI), 14-day log group, CloudWatch alarm + SNS email topic, CloudWatch dashboard `setuscan`. No API Gateway, no CloudFront, nothing reachable from the internet.
- **Verified by direct Lambda invocation** (`scripts/smoke_lambda.py`, 7/7 pass): home page, `/api/health` (`opencv 5.0.0`, `runtime: AWS Lambda`), samples wide / thin / blurry (same statuses as the local Python agent), unknown sample -> 404, non-image upload -> 415. Wide sample agent time 2.5 s on Lambda.
- **Eligibility verdict once phase 2 is run:** meets the gate. OpenCV 5 does the analysis inside an AWS Lambda function ("run a meaningful component on AWS") and there is a working web endpoint (CloudFront + API Gateway). Before phase 2 it has no endpoint, so it does not yet meet it.
- **Bedrock still fails** in ap-southeast-2 (Converse on amazon.nova-micro, apac.amazon.nova-micro, amazon.nova-lite, apac.amazon.nova-lite: "Operation not allowed"). The agentic decider on the Lambda path is therefore the deterministic rules policy (OpenCV output drives zoom / re-shoot / approval decisions; a human approves). The Bedrock `LLMDecider` exists in code but is not wired in and has no run. Do not claim an LLM-driven agent. The Agentic Vision award case rests on the rules agent plus human approval, which is weaker. COOL award: not pursued (Lambda x86, stock OpenCV).
- **Public phase command:** `PUBLIC=1 scripts/deploy_aws.sh` (adds API Gateway HTTP API throttled to 5 req/s + CloudFront with OAC; change set verified: 11 resources added, none else modified except the dashboard).
- The browser (wasm) build below remains a working offline mode but cannot be hosted from the old us-east-1 attempt; in Sydney `scripts/deploy_planb.sh` would work (set the region), but the Lambda path is the main entry.

---
# (Earlier analysis, us-east-1 only) Plan B: OpenCV 5 in the browser

## Verdict (honest)
**As of 25 Oct, Plan B cannot be hosted on this AWS account, and even if it could, it would only weakly meet the gate.**

The rules say: "Every entry must use OpenCV 5 for substantive image or video analysis and run a meaningful component on AWS."
and the entry needs "A working web endpoint or an arranged live screen-share demonstration."

1. **Hosting is blocked.** The organisation SCP `p-ibcu3apr` also explicitly denies `s3:CreateBucket` (tried in us-east-1, us-west-2, ap-south-1, eu-west-1). `s3 ls` works, but the account has no buckets, so there is nothing to host from. CloudFront has no origin to point at.
2. **Bedrock invocation is not working today.** Converse/InvokeModel on Amazon Nova returned `ValidationException: Operation not allowed` (us-east-1, us-east-2, us-west-2) and `Your account is currently being verified` (eu-west-1, ap-south-1). The model shows `authorizationStatus: NOT_AUTHORIZED`. This contradicts the earlier "nova-micro replied" note; it may be the pending account verification and may clear.
3. **Cognito (identity pools and user pools) and Lambda are denied by the SCP**, so a browser cannot get Bedrock credentials. A "Live" Bedrock mode is impossible without a backend.
4. **Even when deployed, the AWS part would be weak.** The OpenCV 5 analysis runs in the viewer's browser, not on AWS. AWS would only serve static files (S3, CloudFront) and, if Bedrock works, hold pre-recorded decision traces. A strict judge could read "run a meaningful component on AWS" as needing AWS compute. We should not claim more than this. The Best Use of COOL award is out of reach (no Graviton compute at all). The Agentic Vision award is only partly met: the OpenCV result drives the next action in-browser with the rules decider, and a Bedrock decider exists in code but has no recorded run.

## Probe results (read-only calls, plus failed create attempts that changed nothing)
| Service | Result |
|---|---|
| Cognito identity pools / user pools | Explicit SCP deny |
| Lambda, API Gateway, Step Functions, Events, AppSync, Amplify, Athena, Glue, EMR, SSM, CodeBuild, Batch, Cloud9, WorkSpaces, RUM, AgentCore | Denied |
| S3 list | Allowed (no buckets). **CreateBucket: explicit SCP deny, 4 regions** |
| CloudFront list, CloudWatch describe, IAM, ECR Public describe | Allowed (read) |
| Lightsail `get-instances`, `get-bundles` | **Allowed (read). Not tested for create** (it would create a public, paid VM). The earlier handoff only listed Lightsail *containers* as denied. |
| Bedrock Converse / InvokeModel | Not allowed today (see above) |

## Recommended route (needs Harsha's OK): Lightsail instance
Lightsail Ubuntu instances are the one compute option not seen denied. A `nano_3_0` bundle lists at $5/month (a few days is well under $2 of the credits). It would run the existing FastAPI app with the Python OpenCV 5 build on a real AWS VM, which is a more defensible "meaningful component on AWS" than static hosting. Unknowns: whether `CreateInstances` passes the SCP (cannot be tested without creating it), and the account verification state. If it works, the browser build below becomes a bonus offline mode.

## What was built for Plan B (works locally)
- `planb/web/setuscan.js`: a JavaScript port of the SetuScan pipeline on OpenCV 5.0.0 (opencv.js WebAssembly, `@techstark/opencv-js@5.0.0-release.1`, integrity pinned in `scripts/build_planb.sh`). Uses ArUco (objdetect), findHomography/warpPerspective (calib3d), black-hat/Hessian-ridge segmentation, distance transform, contours (imgproc). Zhang-Suen thinning is plain JS because `ximgproc` is not in the JS build.
- Not ported: SIFT growth comparison (SIFT is not in the JS build). The page says so and shows the Python result as "Recorded".
- `planb/web/worker.js` runs it in a Web Worker; `planb/web/index.html` is the UI (samples, upload, trace, human approval gate, cross-check against the Python reference, Bedrock panel that says "Not available" with the reason, since no trace exists).
- `scripts/record_bedrock_traces.py`: runs the agent with Bedrock as the decider and stores labelled **RECORDED** traces. Cap 200 calls; the log is `planb/data/bedrock_calls.log`. Run it when Bedrock works, then `scripts/build_planb.sh`.
- `scripts/deploy_planb.sh` (`bucket`, `publish`, `status`) and `scripts/teardown_planb.sh`: private bucket, CloudFront OAC, alarms (no SNS, which is denied), dashboard. **Untested against AWS** because CreateBucket is denied.

## Test results
`tests/test_planb_browser.py` (Playwright, Chromium): 9 passed. Every bundled sample reaches the same final status as desktop Python. Worst-crack width, browser vs Python: wide 0.0 %, thin 0.49 %, growth 0.26 %, no-marker 0.0 %, KrakN 0.2 % (tolerance in test: 5 %). The human gate test confirms no ticket exists before approval. Speed: about 6 to 25 s per sample (non-SIMD wasm, single thread).
Screenshots: `docs/screenshots/planb-{home,wide,blurry}-{desktop,mobile}.png` (1440 and 390 px; no horizontal overflow asserted).

## AWS usage so far
Bedrock: 5 calls logged in `bedrock_calls.log` (all failed in the fallback test) plus about 17 manual failed probes. No successful invocation, so about $0. No resources created.
