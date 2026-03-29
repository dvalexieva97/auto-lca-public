# SOG Extraction - Quick Start Guide

## TL;DR

```bash
# 1. Deploy (first time only)
./build-sog.sh

# 2. Update after code/requirements changes
./update-sog.sh

# 3. Process your PDFs
python3 upload-and-process.py src/auto_lca/data/input-papers
```

---

## Common Workflows

### First Time Setup
```bash
# Deploy everything
./build-sog.sh

# This takes ~5-10 minutes (builds image, deploys services)
```

### After Updating requirements.txt
```bash
# Quick update (much faster!)
./update-sog.sh

# This takes ~3-5 minutes (only rebuilds image and updates services)
```

### Process a Folder of PDFs
```bash
# Upload local PDFs to GCS and process them
python3 upload-and-process.py src/auto_lca/data/input-papers

# The script will:
# 1. Upload all PDFs to gs://lca-pdfs/pdfs/
# 2. Trigger the extraction service
# 3. Return immediately (processing happens in background)

# Optional: specify custom bucket or project
python3 upload-and-process.py path/to/pdfs \
  --project-id ist-lca \
  --bucket-name lca-pdfs \
  --service-url https://sog-main-service-xxx.run.app
```

### Process PDFs Already in GCS
```bash
SERVICE_URL=$(gcloud run services describe sog-main-service \
  --region=us-central1 \
  --format='value(status.url)')

curl -X POST ${SERVICE_URL}/extract_pdfs_from_gcs \
  -H "Content-Type: application/json" \
  -d '{"gcs_prefix": "pdfs/"}'
```

---

## Key Files

| File | Purpose |
|------|---------|
| `build-sog.sh` | Initial deployment (full setup) |
| `update-sog.sh` | Quick update (rebuild & redeploy only) |
| `upload-and-process.py` | Upload local PDFs to GCS and trigger extraction |
| `app_main_sog.py` | Main service (queues PDFs) |
| `Dockerfile.sog` | Docker image definition |

---

## Monitoring

### Check Queue Status
```bash
gcloud tasks queues describe sog-extraction-queue \
  --location=us-central1
```

### View Logs
```bash
# Main service logs
gcloud run services logs read sog-main-service \
  --region=us-central1 \
  --limit=50

# Worker service logs
gcloud run services logs read sog-worker-service \
  --region=us-central1 \
  --limit=50
```

### Query Results (BigQuery)
```sql
SELECT 
  pid,
  inserted_at,
  exec_time,
  error
FROM `ist-lca.paper_sog_extraction`
ORDER BY inserted_at DESC
LIMIT 10
```

---

## Troubleshooting

### "Folder not found" error
- Make sure you're using the client scripts (`extract-pdfs.sh` or `upload-and-extract.py`)
- The deployed service can't access your local filesystem directly
- PDFs must be uploaded to GCS first

### Build takes forever
- Use `./update-sog.sh` instead of `./build-sog.sh` for updates
- Cloud Build can be slow for large images
- Consider caching layers in `Dockerfile.sog`

### PDFs not processing
- Check queue: `gcloud tasks queues describe sog-extraction-queue --location=us-central1`
- Check worker logs for errors
- Ensure worker service has enough memory (8Gi for Mistral)
- Check if PDFs are actually in GCS: `gsutil ls gs://lca-pdfs/pdfs/`

---

## Architecture

```
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│   Local PC   │──────>│  GCS Bucket  │<──────│    Worker    │
│              │upload │  lca-pdfs    │download│   Service    │
└──────────────┘       └──────────────┘       └──────────────┘
       │                       │                      │
       │                       │                      │
       │ trigger               │                      │ writes
       v                       v                      v
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│     Main     │──────>│ Cloud Tasks  │──────>│   BigQuery   │
│   Service    │create │    Queue     │invoke │    Table     │
└──────────────┘       └──────────────┘       └──────────────┘
```

1. Upload PDFs to GCS
2. Main service creates tasks (one per PDF)
3. Workers pull tasks from queue
4. Workers download PDFs, run extraction
5. Results saved to BigQuery


