# SetuScan: Devpost field text (draft for Harsha to paste by hand)

Entrant: Harsha Shinde (solo). Submit by hand on Devpost; nothing here has been submitted.

## Project name
SetuScan

## Tagline (one line)
Agentic crack inspection with OpenCV 5: measures cracks in millimetres, re-looks when unsure, tracks growth, and waits for an engineer's approval.

## Inspiration
Cracks in concrete and masonry are usually recorded by eye. A phone photo has no scale, so nobody can say whether a crack is hairline or wide, or whether it is growing between visits. Setu means bridge in Hindi and Sanskrit. SetuScan aims to make routine visual inspection measured, comparable over time and auditable, with a human engineer in charge. It is decision support, not a structural certification.

## What it does
1. Quality gate: too dark, glare or a featureless frame leads to a re-shoot request with advice.
2. Scale and blur from one printed ArUco marker placed beside the crack (px per mm, perspective removed, and the marker's edges reveal photo blur).
3. Crack segmentation and measurement: length from the skeleton, sub-pixel width from perpendicular profiles, blur-corrected.
4. Active perception: if a crack is under 3 px wide or near a policy band edge, the agent re-renders it at native resolution and re-measures, or asks for a closer photo.
5. Two-visit change detection with SIFT and RANSAC: crack extension in mm and width change.
6. Human approval: the agent can only request approval. A ticket exists only after a named engineer approves. Rejections are logged.
Every step is in a trace with the reason, citing measured numbers. The decider is a deterministic rules policy (no LLM). The severity bands are an example policy.

## How we built it
* OpenCV 5.0.0 (opencv-contrib-python-headless 5.0.0.93): imgproc, objdetect (ArUco), calib3d (homography, RANSAC/MAGSAC), features2d (SIFT, FLANN), ximgproc (thinning). No learned weights.
* Python 3.12, FastAPI, a plain HTML/JS front end.
* AWS (ap-southeast-2): Lambda container image (2 GB), API Gateway HTTP API (throttled to 5 req/s), ECR, private S3 with 1-day expiry, CloudWatch logs, dashboard and alarm, SNS email, least-privilege IAM, CloudFormation, AWS Budgets ($20, alerts at $5/$10/$20).
* One-command deploy and teardown scripts, pinned dependencies (uv.lock, requirements-lambda.txt).

## Evaluation (all numbers come from files in the repo)
* KrakN public dataset (CC BY 4.0), held-out test split of 2,819 crack and 5,677 background patches: precision 0.852, recall 0.884, F1 0.868.
* SYNTHETIC scenes with exact ground truth (84 scenes): mean width error 0.054 mm, measured on the median width along the crack; at most 1.5 percent median-width error for cracks 0.8 mm and wider (the app shows median and p95; the severity policy uses p95, which reads higher: wide sample true 2.4 mm, median 2.41 mm, p95 2.54 mm); thin cracks (0.3 mm and below) are over-read by 0.09 to 0.16 mm.
* Agent task success on 14 SYNTHETIC scenarios x 5 seeds: 69 of 70.
* Live AWS endpoint, re-run 26 Oct 2026 on the final build (eval/live-aws.md): 6 of 6 labelled samples gave the expected outcome on every run; sample inspections took 1.4 to 3.3 s from India and the two-visit growth sample 6.4 s; real uploads took 4.4 s (one photo) and 8.5 s (two visits); a cold start after idle took 4 to 6.5 s in an earlier measurement. The approval gate and the 404, 409 and 415 error paths were checked live.

## Challenges
* The AWS account sat behind an organisation policy that blocked compute until we found it was region-locked to Sydney.
* Thin cracks are only a few pixels wide, so we added blur correction from the marker and a zoom step instead of trusting one reading.
* Amazon Bedrock was denied on this account, so the agent runs on rules. We chose to say so rather than fake an LLM.

## Accomplishments we are proud of
A live, deployed OpenCV 5 pipeline on AWS with an honest evaluation including failure cases, an agent that refuses to give a number when the photo cannot support one, and a human approval gate that the code enforces.

## What we learned
Scale and blur have to be measured from the scene itself (the marker) before any millimetre number means anything, and a conservative agent that asks for a better photo is more useful than one that always answers.

## What is not done (stated on the page, in the README and in the video)
* No Amazon Bedrock model and no LLM in the loop (access denied). The LLMDecider code exists but has never run.
* No CloudFront (account verification needed); the Lambda serves the UI.
* No Graviton / COOL build; the COOL award is not claimed.
* Real-photo millimetre validation is preliminary: 2 real cracks, 4 photos, ruler as gauge, marker measured at about 18 mm. Median width within 6-48 % of the ruler's maximum-width reading (mean absolute error 0.24 mm over 3 photos); in 1 of 4 photos no crack was found (a miss); crack length unreliable on rough plaster. Main accuracy is on SYNTHETIC scenes, detection on KrakN.
* Severity bands are an example policy. Not a structural certification.

## Built with
OpenCV 5, Python, FastAPI, AWS Lambda, Amazon API Gateway, Amazon ECR, Amazon S3, Amazon CloudWatch, Amazon SNS, AWS CloudFormation, AWS Budgets.

## Try it
Live: https://l4734hezca.execute-api.ap-southeast-2.amazonaws.com (press "Inspect this sample" on any sample; about 2 to 7 s).
Code: https://github.com/xexpertai/setuscan
Video: https://youtu.be/aEa8UofuG-c

## Special awards
* Agentic Vision Award: the agent loop is OpenCV-driven (quality, scale, segmentation, zoom, registration), has guardrails outside the decider, a full trace, failure handling (re-shoot requests) and a human approval gate. Weakness to state plainly: the decider is rules, not an LLM.
* Best Use of COOL: not claimed (x86_64 Lambda, stock OpenCV wheel).

## AI-assistance disclosure
The code, tests and documentation were written with an AI coding assistant (Claude Code) under the entrant's direction and review. Narration: AI text-to-speech voice (Kokoro). No language model runs inside the product. Synthetic images are generated by the repository's own generator and labelled; KrakN (Zarski, Wojcik, Miszczak, 2021, CC BY 4.0) is used with attribution.

## Responsible use
Decision support only. Uploads are re-encoded so EXIF and GPS are never stored; cases are deleted after 24 h; users are asked not to photograph people; a named engineer must approve any ticket.
