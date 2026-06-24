# Deploying the comic panel search (Hugging Face Spaces)

This branch (`streamlit-hosting`) is the distilled, deployable version of the app. It runs the existing Streamlit UI as-is on a persistent host, with CLIP running in-process on CPU, sparse and full-text search going straight to Pinecone, and panel images served from S3. The sound feature is deferred (the button is unwired; the backend in `src/sounds/` is kept for later).

The heavier Vercel + Next.js + Lambda-embedder architecture lives on the `vercel-hosting` branch.

## Why this is simple

Nothing here needs a separate embedder service. The CLIP text encoder is small and fast on CPU, so `embed_text_query` runs in the same process as the app (cached via `@st.cache_resource`). The only thing that had to change for hosting is where images come from, which is handled by `src/storage/images.py` (`IMAGE_STORE=s3`).

## 1. Images to S3

```bash
aws s3 mb s3://YOUR_BUCKET --region us-east-1

# Downscales to ~900px (the UI never shows larger) and sets immutable cache headers.
python scripts/upload_images_to_s3.py \
    --bucket YOUR_BUCKET --prefix panels --region us-east-1 \
    --resize --workers 32
```

Make the bucket readable either with a public-read bucket policy, or by putting a CloudFront distribution in front of it (recommended for caching). If you use CloudFront, note its domain for `IMAGE_CDN_BASE_URL` below. Object keys mirror the relative `image_path` stored in Pinecone, so no re-upsert is needed.

## 2. Create the Space and push

Create a new Hugging Face Space with the **Streamlit** SDK, then push this branch to it:

```bash
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git push space streamlit-hosting:main
```

The Space reads the YAML front-matter at the top of `README.md` (`sdk: streamlit`, `app_file: app.py`) and builds from `requirements.txt`.

## 3. Space secrets

In the Space, set these under Settings → Variables and secrets:

```
PINECONE_API_KEY      = <your key>
PINECONE_INDEX_NAME   = comic-panels
PINECONE_NAMESPACE    = comics-v1
IMAGE_STORE           = s3
S3_BUCKET             = YOUR_BUCKET
S3_PREFIX             = panels          # must match the --prefix used in step 1
S3_REGION             = us-east-1
IMAGE_CDN_BASE_URL    = https://dxxxx.cloudfront.net   # only if using CloudFront
```

No AWS credentials are needed at runtime. `image_src` only builds public URLs; the browser fetches images directly from S3/CloudFront, never through the app.

## 4. Build and run notes

- The free CPU tier (16GB RAM) comfortably runs torch + CLIP. Streamlit Community Cloud's ~1GB tier would OOM, which is why HF Spaces is the target.
- Free Spaces sleep after inactivity, so the first request after a sleep reloads the CLIP model and takes a few extra seconds.

## Optional: faster, smaller builds with CPU-only torch

The default `torch` wheel pulls ~2GB of CUDA libraries that a CPU Space never uses. To slim and speed up the build, replace the torch line in `requirements.txt` with a CPU build, for example:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.2.2+cpu
```

Pick a `+cpu` version that matches `open-clip-torch`'s supported range. This is an optimization, not required; the default torch works on the free tier.

## Local development

Leave `IMAGE_STORE` unset (defaults to `local`) to read panels off disk and inline them as base64, exactly as before:

```bash
IMAGE_STORE=local streamlit run app.py
```
