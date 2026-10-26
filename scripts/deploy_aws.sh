#!/usr/bin/env bash
# Two-phase deploy of SetuScan to AWS (default region ap-southeast-2; this account's SCP blocks most other regions).
#   scripts/deploy_aws.sh             PHASE 1 (private): ECR image -> CloudFormation (Public=false: buckets, Lambda, log group,
#                                     alarm + SNS topic, dashboard) -> UI files into the private UI bucket -> direct Lambda smoke test.
#                                     Nothing is reachable from the internet.
#   PUBLIC=1 scripts/deploy_aws.sh    PHASE 2: same stack with Public=true -> API Gateway HTTP API + CloudFront (OAC) = the public URL.
# Needs: aws CLI v2, docker. Override the region with AWS_DEFAULT_REGION.
set -euo pipefail
cd "$(dirname "$0")/.."
export AWS_PAGER="" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-2}" PATH="$HOME/.local/bin:$PATH"
AWS="${AWS:-aws}"; STACK=setuscan REPO=setuscan
PUB=false; [ "${PUBLIC:-}" = 1 ] && PUB=true
ACCT=$($AWS sts get-caller-identity --query Account --output text)
REG="$ACCT.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com"
out() { $AWS cloudformation describe-stacks --stack-name $STACK --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }

URI=""
if [ "$PUB" = true ] && [ -z "${REBUILD:-}" ] && $AWS cloudformation describe-stacks --stack-name $STACK >/dev/null 2>&1; then
  URI=$($AWS lambda get-function --function-name setuscan-api --query Code.ImageUri --output text)   # reuse the phase-1 image
fi
if [ -z "$URI" ]; then
  $AWS ecr describe-repositories --repository-names $REPO >/dev/null 2>&1 || \
    $AWS ecr create-repository --repository-name $REPO --image-scanning-configuration scanOnPush=true >/dev/null
  $AWS ecr put-lifecycle-policy --repository-name $REPO --lifecycle-policy-text \
    '{"rules":[{"rulePriority":1,"description":"keep 3","selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":3},"action":{"type":"expire"}}]}' >/dev/null
  $AWS ecr get-login-password | docker login --username AWS --password-stdin "$REG" >/dev/null
  TAG=$(date +%Y%m%d%H%M%S)
  PROV=""; docker buildx version >/dev/null 2>&1 && PROV="--provenance=false"
  docker build $PROV -f Dockerfile.lambda -t "$REG/$REPO:$TAG" .
  docker push "$REG/$REPO:$TAG"
  URI="$REG/$REPO:$TAG"
fi

$AWS cloudformation deploy --stack-name $STACK --template-file infra/template.yaml \
  --capabilities CAPABILITY_IAM --no-fail-on-empty-changeset \
  --parameter-overrides "ImageUri=$URI" "Public=$PUB" "UseCdn=${USE_CDN:-false}" "AlarmEmail=${ALARM_EMAIL:-alarms@example.com}" "ReservedConcurrency=${RESERVED_CONCURRENCY:-0}"

UI=$(out UiBucketName)
$AWS s3 cp web/index.html "s3://$UI/index.html" --content-type "text/html; charset=utf-8" --cache-control "no-cache"

# direct invoke (no public endpoint needed): API Gateway v2 payload for GET /api/health
EV=$(mktemp); cat > "$EV" <<'JSON'
{"version":"2.0","routeKey":"$default","rawPath":"/api/health","rawQueryString":"","headers":{"host":"x"},
 "requestContext":{"http":{"method":"GET","path":"/api/health","sourceIp":"127.0.0.1","userAgent":"smoke"},"stage":"$default"},"isBase64Encoded":false}
JSON
$AWS lambda invoke --function-name setuscan-api --payload "fileb://$EV" --cli-binary-format raw-in-base64-out /tmp/setuscan-smoke.json >/dev/null
echo "Lambda smoke: $(cat /tmp/setuscan-smoke.json | cut -c1-300)"; rm -f "$EV"

if [ "$PUB" = true ]; then
  [ "${USE_CDN:-false}" = true ] && $AWS cloudfront create-invalidation --distribution-id "$(out DistributionId)" --paths '/*' >/dev/null
  echo "PUBLIC: $(out SiteUrl)"
else
  echo "Phase 1 done: private only. For the public URL run:  PUBLIC=1 scripts/deploy_aws.sh"
fi
