# Deploying the comic panel search (Vercel-hosted option)

This branch (`vercel-hosting`) moves the app off local files so it can run as a
public, hosted service. Pinecone (dense, sparse, FTS) is already remote; the two
things that pin the app to a laptop are the **64GB of panel images** and the
**CLIP text encoder** that embeds dense queries. The sound feature is **deferred**
for the initial release (the button is unwired; the backend stays in `src/sounds/`).

## Status on this branch

Done:
- **Image hosting abstraction** — `src/storage/images.py` resolves a panel's
  relative `image_path` to either a base64 data URI (local) or an S3/CDN URL,
  switched by `IMAGE_STORE`. `app.py` now calls `image_src()` instead of reading
  files directly.
- **Upload script** — `scripts/upload_images_to_s3.py` pushes the panel images to
  S3 with the same key layout the app reads back, with optional downscaling.
- **Sound button unwired** — removed from the results UI; `_find_sounds` /
  `_render_sounds` and `src/sounds/*` are intact for a future release.
- **Config** — `.env.example` documents the new vars; `boto3` added to requirements.

Remaining (not yet built — see "Next" below):
- Frontend on Vercel (Next.js) — Streamlit cannot run on Vercel.
- A hosted CLIP text-query embedder — torch does not fit a Vercel function.

## 1. Images → S3

```bash
aws s3 mb s3://YOUR_BUCKET --region us-east-1

# Recommended: downscale to ~900px on the way up (the UI never shows larger).
python scripts/upload_images_to_s3.py \
    --bucket YOUR_BUCKET --prefix panels --region us-east-1 \
    --resize --workers 32
```

Then set, in the app's environment:

```
IMAGE_STORE=s3
S3_BUCKET=YOUR_BUCKET
S3_PREFIX=panels          # must match the --prefix used above
S3_REGION=us-east-1
# IMAGE_CDN_BASE_URL=https://dxxxx.cloudfront.net   # if fronting with CloudFront
```

The bucket must be publicly readable (bucket policy, or a CloudFront
distribution in front of it). For private buckets, swap `s3_url()` in
`src/storage/images.py` for presigned URLs generated server-side.

No Pinecone re-upsert is needed: metadata already stores the relative paths the
S3 keys mirror.

## 2. Frontend on Vercel — NEXT

Streamlit needs a persistent websocket server, which Vercel does not provide, so
the UI gets rebuilt as a Next.js app deployed to Vercel. It calls Pinecone for
sparse/FTS directly, calls the embedder service (below) for dense queries, and
renders images straight from the S3/CloudFront URLs.

> If Vercel isn't a hard requirement, a persistent host (Fly / Railway / Render /
> EC2 / HF Spaces) can run the existing Streamlit app as-is with only step 1's
> env vars — CLIP runs in-process on CPU and step 3 is unnecessary.

## 3. CLIP text-query embedder — NEXT (decision pending)

Dense queries must be embedded with the **exact** model used for the panels —
OpenAI CLIP ViT-B/16 (`open_clip`, `pretrained="openai"`, 512-dim) — or vectors
won't be comparable. Torch can't run in a Vercel function, so this lives in a
small service that wraps `src.embeddings.embed_images.embed_text_query`.

Hosting options (undecided): AWS Lambda **container image** (10GB limit fits
torch; stays in existing AWS billing), SageMaker endpoint, or Modal / Baseten /
Replicate. Verify parity once by embedding a few queries locally vs. through the
endpoint and confirming cosine similarity ≈ 1.0.
