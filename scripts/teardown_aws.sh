#!/usr/bin/env bash
# Removes everything scripts/deploy_aws.sh created (stack, buckets' contents, ECR repo). Asks for CONFIRM=1.
set -euo pipefail
export AWS_PAGER="" AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-ap-southeast-2}"
AWS="${AWS:-aws}"
[ "${CONFIRM:-}" = 1 ] || { echo "Set CONFIRM=1 to delete the setuscan stack, buckets and ECR repo."; exit 1; }
out() { $AWS cloudformation describe-stacks --stack-name setuscan --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text; }
for k in UiBucketName CasesBucketName; do b=$(out $k 2>/dev/null || true); [ -n "$b" ] && [ "$b" != None ] && $AWS s3 rm "s3://$b" --recursive --only-show-errors || true; done
$AWS cloudformation delete-stack --stack-name setuscan
$AWS cloudformation wait stack-delete-complete --stack-name setuscan
$AWS ecr delete-repository --repository-name setuscan --force >/dev/null || true
echo "Deleted. (Budget setuscan-monthly is kept; remove with: aws budgets delete-budget --account-id <id> --budget-name setuscan-monthly)"
