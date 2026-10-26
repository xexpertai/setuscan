#!/usr/bin/env bash
# Plan B deploy (static site on AWS; no compute services). Run from anywhere. Idempotent.
#   scripts/deploy_planb.sh bucket    private S3 bucket (all public access blocked) + upload dist-planb/   [safe, no public URL]
#   CONFIRM_PUBLIC=1 scripts/deploy_planb.sh publish
#                                     CloudFront distribution with Origin Access Control + bucket policy
#                                     + CloudWatch alarms + dashboard. THIS CREATES THE PUBLIC URL.
#   scripts/deploy_planb.sh status    print bucket / distribution / URL
set -euo pipefail
cd "$(dirname "$0")/.."
export AWS_PAGER="" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}" PATH="$HOME/.local/bin:$PATH"
ACCT=$(aws sts get-caller-identity --query Account --output text)
BUCKET="${BUCKET:-setuscan-planb-$ACCT}"; NAME=setuscan-planb
cmd="${1:-bucket}"

dist_id() { aws cloudfront list-distributions --query "DistributionList.Items[?Comment=='$NAME'].Id | [0]" --output text 2>/dev/null | sed 's/^None$//'; }

bucket() {
  scripts/build_planb.sh
  if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    aws s3api create-bucket --bucket "$BUCKET" >/dev/null   # us-east-1: no LocationConstraint
  fi
  aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
  aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
  aws s3api put-bucket-tagging --bucket "$BUCKET" --tagging 'TagSet=[{Key=project,Value=setuscan-planb}]'
  aws s3 sync dist-planb "s3://$BUCKET" --delete --exclude "index.html" --cache-control "public, max-age=3600" --only-show-errors
  aws s3 cp dist-planb/index.html "s3://$BUCKET/index.html" --content-type "text/html; charset=utf-8" --cache-control "no-cache" --only-show-errors
  aws s3 cp dist-planb/vendor/opencv.js "s3://$BUCKET/vendor/opencv.js" --content-type "text/javascript" --cache-control "public, max-age=31536000, immutable" --only-show-errors
  echo "Private bucket ready: s3://$BUCKET (public access blocked, no distribution)."
}

publish() {
  [ "${CONFIRM_PUBLIC:-}" = "1" ] || { echo "Refusing: this makes the site public. Re-run with CONFIRM_PUBLIC=1 after Harsha's OK." >&2; exit 2; }
  bucket
  local oac did
  oac=$(aws cloudfront list-origin-access-controls --query "OriginAccessControlList.Items[?Name=='$NAME'].Id | [0]" --output text | sed 's/^None$//')
  [ -n "$oac" ] || oac=$(aws cloudfront create-origin-access-control --origin-access-control-config \
    "Name=$NAME,Description=$NAME,SigningProtocol=sigv4,SigningBehavior=always,OriginAccessControlOriginType=s3" --query OriginAccessControl.Id --output text)
  did=$(dist_id)
  if [ -z "$did" ]; then
    cfg=$(mktemp)
    cat > "$cfg" <<JSON
{"CallerReference":"$NAME-$(date +%s)","Comment":"$NAME","Enabled":true,"DefaultRootObject":"index.html","PriceClass":"PriceClass_100","HttpVersion":"http2and3",
 "Origins":{"Quantity":1,"Items":[{"Id":"s3","DomainName":"$BUCKET.s3.$AWS_DEFAULT_REGION.amazonaws.com","OriginAccessControlId":"$oac","S3OriginConfig":{"OriginAccessIdentity":""}}]},
 "DefaultCacheBehavior":{"TargetOriginId":"s3","ViewerProtocolPolicy":"redirect-to-https","Compress":true,"AllowedMethods":{"Quantity":2,"Items":["GET","HEAD"]},
   "CachePolicyId":"658327ea-f89d-4fab-a63d-7e88639e58f6","ResponseHeadersPolicyId":"67f7725c-6f97-4210-82d7-5512b698cee1"}}
JSON
    did=$(aws cloudfront create-distribution --distribution-config "file://$cfg" --query Distribution.Id --output text); rm -f "$cfg"
  fi
  aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"CloudFrontOAC\",\"Effect\":\"Allow\",
    \"Principal\":{\"Service\":\"cloudfront.amazonaws.com\"},\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::$BUCKET/*\",
    \"Condition\":{\"StringEquals\":{\"AWS:SourceArn\":\"arn:aws:cloudfront::$ACCT:distribution/$did\"}}}]}"
  # alarms: no SNS (denied by the org policy), so they show on the dashboard and in the console only
  for m in "5xxErrorRate:5" "4xxErrorRate:30"; do
    aws cloudwatch put-metric-alarm --alarm-name "$NAME-${m%%:*}" --namespace AWS/CloudFront --metric-name "${m%%:*}" \
      --dimensions Name=DistributionId,Value="$did" Name=Region,Value=Global --statistic Average --period 300 --evaluation-periods 2 \
      --threshold "${m##*:}" --comparison-operator GreaterThanThreshold --treat-missing-data notBreaching
  done
  aws cloudwatch put-metric-alarm --alarm-name "$NAME-requests-burst" --namespace AWS/CloudFront --metric-name Requests \
    --dimensions Name=DistributionId,Value="$did" Name=Region,Value=Global --statistic Sum --period 300 --evaluation-periods 1 \
    --threshold 20000 --comparison-operator GreaterThanThreshold --treat-missing-data notBreaching
  dash=$(python3 - "$did" <<'PY'
import json, sys
d = sys.argv[1]
def w(x, y, title, metric, stat):
    return {"type": "metric", "x": x, "y": y, "width": 12, "height": 6, "properties": {"title": title, "region": "us-east-1", "stat": stat, "period": 300,
            "metrics": [["AWS/CloudFront", metric, "DistributionId", d, "Region", "Global"]]}}
print(json.dumps({"widgets": [w(0, 0, "Requests", "Requests", "Sum"), w(12, 0, "Bytes downloaded", "BytesDownloaded", "Sum"),
                              w(0, 6, "5xx error rate %", "5xxErrorRate", "Average"), w(12, 6, "4xx error rate %", "4xxErrorRate", "Average")]}))
PY
)
  aws cloudwatch put-dashboard --dashboard-name "$NAME" --dashboard-body "$dash" >/dev/null
  aws cloudfront create-invalidation --distribution-id "$did" --paths '/*' >/dev/null
  echo "Distribution $did created. URL (may take ~5-10 min to deploy): https://$(aws cloudfront get-distribution --id "$did" --query Distribution.DomainName --output text)/"
}

status() {
  echo "bucket: $BUCKET ($(aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null && echo exists || echo missing))"
  did=$(dist_id); [ -n "$did" ] && echo "distribution: $did https://$(aws cloudfront get-distribution --id "$did" --query Distribution.DomainName --output text)/" || echo "distribution: none (site is NOT public)"
}
"$cmd"
