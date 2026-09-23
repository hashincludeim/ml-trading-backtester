#!/usr/bin/env bash
# One-time Google Cloud setup so GitHub Actions can deploy StockML to Cloud Run.
#
# Run it in Cloud Shell (shell.cloud.google.com), which already has gcloud and is signed in:
#   bash gcp-setup.sh <project-id> <github-owner/repo> [region]
#
# It creates two service accounts and a Workload Identity Federation pool. The deployer can
# only be used by workflows from that repository's main branch, and no key file is created.
# It is safe to run again: anything that already exists is left alone.
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: bash gcp-setup.sh <project-id> <github-owner/repo> [region]" >&2
    exit 1
fi
PROJECT_ID=$1
REPO=$2
REGION=${3:-europe-west2}
POOL=github
PROVIDER=github-actions
DEPLOYER=stockml-deployer@$PROJECT_ID.iam.gserviceaccount.com
RUNTIME=stockml-runtime@$PROJECT_ID.iam.gserviceaccount.com

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "Enabling APIs (this can take a minute)…"
gcloud services enable run.googleapis.com iam.googleapis.com iamcredentials.googleapis.com \
    sts.googleapis.com

create_service_account() {  # name, display name
    if ! gcloud iam service-accounts describe "$1@$PROJECT_ID.iam.gserviceaccount.com" \
        >/dev/null 2>&1; then
        gcloud iam service-accounts create "$1" --display-name="$2"
    fi
}
# The app calls no Google APIs, so the runtime identity gets no roles at all.
create_service_account stockml-runtime "StockML Cloud Run runtime"
create_service_account stockml-deployer "StockML GitHub Actions deployer"

# New service accounts can take a few seconds before IAM bindings accept them.
for _ in 1 2 3 4 5 6; do
    gcloud iam service-accounts describe "$DEPLOYER" >/dev/null 2>&1 && break
    sleep 5
done

echo "Granting the deployer permission to deploy and to run the service as the runtime account…"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$DEPLOYER" \
    --role=roles/run.admin --condition=None >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME" \
    --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser >/dev/null

echo "Setting up keyless sign-in from GitHub Actions…"
if ! gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1; then
    gcloud iam workload-identity-pools create "$POOL" --location=global \
        --display-name="GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" >/dev/null 2>&1; then
    gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
        --workload-identity-pool="$POOL" --display-name="GitHub Actions" \
        --issuer-uri="https://token.actions.githubusercontent.com" \
        --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
        --attribute-condition="assertion.repository == '$REPO' && assertion.ref == 'refs/heads/main'"
fi
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER" \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO" \
    >/dev/null

cat <<EOF

Done. In GitHub, open $REPO → Settings → Secrets and variables → Actions → Variables,
and add these repository variables:

  GCP_PROJECT_ID       $PROJECT_ID
  GCP_REGION           $REGION
  GCP_WIF_PROVIDER     projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER
  GCP_SERVICE_ACCOUNT  $DEPLOYER

Then re-run the "CI and deploy" workflow (Actions → CI and deploy → Run workflow).
EOF
