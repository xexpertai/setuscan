#!/usr/bin/env bash
# Remove everything deploy_planb.sh created. Needs CONFIRM=1.
set -euo pipefail
[ "${CONFIRM:-}" = "1" ] || { echo "Set CONFIRM=1 to delete the Plan B site, alarms, dashboard and bucket." >&2; exit 2; }
export AWS_PAGER="" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" PATH="$HOME/.local/bin:$PATH"
ACCT=$(aws sts get-caller-identity --query Account --output text); BUCKET="${BUCKET:-setuscan-planb-$ACCT}"; NAME=setuscan-planb
did=$(aws cloudfront list-distributions --query "DistributionList.Items[?Comment=='$NAME'].Id | [0]" --output text | sed 's/^None$//')
if [ -n "$did" ]; then
  etag=$(aws cloudfront get-distribution-config --id "$did" --query ETag --output text)
  aws cloudfront get-distribution-config --id "$did" --query DistributionConfig | python3 -c 'import json,sys;c=json.load(sys.stdin);c["Enabled"]=False;print(json.dumps(c))' > /tmp/planb-dist.json
  aws cloudfront update-distribution --id "$did" --if-match "$etag" --distribution-config file:///tmp/planb-dist.json >/dev/null
  aws cloudfront wait distribution-deployed --id "$did"
  aws cloudfront delete-distribution --id "$did" --if-match "$(aws cloudfront get-distribution-config --id "$did" --query ETag --output text)"
fi
oac=$(aws cloudfront list-origin-access-controls --query "OriginAccessControlList.Items[?Name=='$NAME'].Id | [0]" --output text | sed 's/^None$//')
[ -z "$oac" ] || aws cloudfront delete-origin-access-control --id "$oac" --if-match "$(aws cloudfront get-origin-access-control --id "$oac" --query ETag --output text)"
aws cloudwatch delete-alarms --alarm-names $NAME-5xxErrorRate $NAME-4xxErrorRate $NAME-requests-burst || true
aws cloudwatch delete-dashboards --dashboard-names $NAME || true
aws s3 rm "s3://$BUCKET" --recursive --only-show-errors 2>/dev/null || true
aws s3api delete-bucket --bucket "$BUCKET" 2>/dev/null || true
echo "Plan B resources removed."
