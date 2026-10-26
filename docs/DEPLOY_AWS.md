# Deploying SetuScan to AWS (phase 2: planned, not yet deployed)

Status on 12 Oct 2026: the AWS account does not exist yet. Nothing in this file has been run.
Prices and free-tier figures are planning estimates to verify in the AWS console at deploy time.

## 0. Prerequisites (done by the account owner, by hand)

1. Create the AWS account on the **Free plan** (no charges unless upgraded to the Paid plan).
2. Turn on root MFA. Create an IAM Identity Center user (or an IAM user) for the CLI. Never use root keys.
3. **AWS Budgets: alerts only, no automatic shutdown actions** (judges must never hit an outage):
   * budget `setuscan-monthly`, monthly cost, amount $20;
   * alert thresholds at **$5, $10 and $20** actual spend, plus a forecast alert at $20, emailed to the owner;
   * do not attach any budget action (no IAM deny, no SCP, no stopping resources).
4. Region: `us-east-1`.
5. Store the CLI credentials only in `~/.aws/` or a gitignored `.env.local`, typed via hidden input.
   Never commit or print them.
6. Optional, only if available on the Free plan: Amazon Bedrock model access for one low-cost
   first-party model, used by `LLMDecider`. If it is not available, the deterministic
   `RulesDecider` stays in use, and the UI and trace already say so.

Not used: AWS Marketplace / COOL (needs the Paid plan), and EC2 instances.

## 1. Target architecture

| Concern | AWS resource | Notes |
|---|---|---|
| Static UI | S3 bucket (private) + CloudFront (OAC) | `web/index.html` + samples; HTTPS; no cookies |
| API | API Gateway **HTTP API** | routes `/api/*`; stage throttling 5 req/s, burst 10 |
| Vision tools | **Lambda, arm64 (Graviton), container image** from ECR | base `public.ecr.aws/lambda/python:3.12-arm64`; `opencv-contrib-python-headless==5.0.0.93` (manylinux aarch64 wheel exists on PyPI); 2048 MB; 60 s timeout; reserved concurrency 5 |
| Agent loop | **Step Functions Standard** state machine | one state per tool call; a Choice state reads the decider's output; `request_approval` uses `.waitForTaskToken` |
| Decider | Lambda `decide` | `RulesDecider` by default; `LLMDecider` with Bedrock only if enabled, with a daily call cap held in DynamoDB, falling back to rules |
| Human approval | API route `/api/cases/{id}/approve` → Lambda → `SendTaskSuccess` / `SendTaskFailure` | the ticket is written only after the task token is returned by a human action |
| Images | S3 bucket `setuscan-cases-<acct>` | SSE-S3, Block Public Access on, **lifecycle: expire after 1 day**; presigned PUT for uploads (12 MB cap) |
| Case records and tickets | DynamoDB on-demand tables `cases`, `tickets` | TTL attribute: 24 h on cases; tickets kept for the demo period |
| Observability | CloudWatch Logs (14-day retention), embedded-metric-format metrics, X-Ray tracing on Lambda and Step Functions | the Step Functions execution graph is the run-time agent trace |
| Security | IAM: one least-privilege role per function | no wildcard resources; no secrets in the browser; WAF not needed at this scale |

The local code maps one-to-one: `agent.TOOLS[name]` becomes one Lambda handler (or one handler
dispatching on `name`), `agent.allowed_actions` runs inside the `decide` Lambda before a choice is
returned, and `store.LocalStore` is replaced by an S3/DynamoDB store with the same methods.

## 2. Steps (to run once the account exists)

```bash
# 0) tools: AWS CLI v2, AWS SAM CLI, Docker with buildx for linux/arm64
aws configure sso            # or aws configure, typing keys via the prompt (never on the command line)
aws sts get-caller-identity  # sanity check

# 1) build the arm64 image (Dockerfile.lambda to be added in phase 2)
docker buildx build --platform linux/arm64 -f Dockerfile.lambda -t setuscan:phase2 .

# 2) deploy everything with SAM (template.yaml to be added in phase 2)
sam build --use-container
sam deploy --guided --stack-name setuscan --region us-east-1 \
  --parameter-overrides DeciderMode=rules BedrockDailyCap=0

# 3) upload the static UI
aws s3 sync web/ s3://<ui-bucket>/ --delete
aws cloudfront create-invalidation --distribution-id <id> --paths '/*'

# 4) smoke test (script to be added in phase 2): health, one sample per path, approve flow
python scripts/smoke_test.py --base https://<cloudfront-domain>
```

Repeatability: everything is in `template.yaml` (SAM). `sam delete` removes the stack, and S3
buckets are emptied by a custom resource on delete.

## 3. Cost estimate (build period 13–26 Oct plus judging 27 Oct – 9 Nov)

Assumption: up to 2,000 inspections in total, each about 2 s of Lambda at 2 GB with about
12 Step Functions transitions.

| Item | Expected | Worst case with throttles | Basis (verify at deploy time) |
|---|---|---|---|
| Lambda arm64 | $0 | $2 | ~8,000 GB-s needed; free allowance and credits cover it |
| API Gateway HTTP API | < $0.10 | $1 | ~$1 per million requests |
| Step Functions Standard | < $0.30 | $1 | ~24k transitions |
| S3 + CloudFront | < $0.20 | $1 | tiny objects, 1-day expiry |
| ECR (one ~0.5 GB image) | ~$0.05 | $0.20 | |
| DynamoDB, CloudWatch, X-Ray | ~$0 | $1 | on-demand, 14-day log retention |
| Bedrock (only if enabled) | $0–3 | $6 | capped per day in code, then rules fallback |
| **Total** | **≈ $0–4** | **≈ $12** | inside the Free-plan credits; expected card charge $0 |

Budget alerts at $5/$10/$20 (alerts only) give early warning far below the credit balance.

## 4. Responsible-operation checklist for phase 2

- [ ] S3 lifecycle 1 day and DynamoDB TTL 24 h verified with a test object
- [ ] Upload size cap and content-type checks in the presigned-URL Lambda
- [ ] API throttling and Lambda reserved concurrency set (protects credits)
- [ ] CloudWatch alarm on Lambda errors > 0 over 5 minutes (email)
- [ ] X-Ray traces visible for one full run; Step Functions execution history kept 90 days
- [ ] The UI footer shows the decider actually in use (rules or LLM), read from the API
- [ ] No people or number plates in samples; the upload form says not to photograph people
