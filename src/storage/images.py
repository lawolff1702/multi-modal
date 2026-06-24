"""Resolve a panel's stored relative ``image_path`` to a browser-loadable src.

Pinecone metadata stores each panel's path relative to the project root (e.g.
``data/comics/raw/raw_panel_images/0/0_0.jpg``). Two backends turn that into
something an ``<img src>`` can load, selected by the ``IMAGE_STORE`` env var:

- ``s3``   — return a public S3/CDN URL. The app never touches the bytes; the
             browser fetches directly from S3 (or CloudFront). Used in hosted
             deploys where the 64GB image set lives in a bucket, not on disk.
- ``local`` (default) — read the file from disk, downscale, and return a base64
             data URI. The original behavior, kept for local dev.

The S3 key mirrors the relative path exactly, so ``scripts/upload_images_to_s3.py``
and this module agree on where each panel lives without any extra mapping.
"""

import base64
import os
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]

IMAGE_STORE = os.environ.get("IMAGE_STORE", "local").lower()
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_PREFIX = os.environ.get("S3_PREFIX", "").strip("/")
S3_REGION = os.environ.get("S3_REGION") or os.environ.get("AWS_REGION", "us-east-1")
# Public base URL to serve images from (a CloudFront domain or a public bucket
# website endpoint). When unset, a virtual-hosted-style S3 URL is built from the
# bucket and region.
IMAGE_CDN_BASE_URL = os.environ.get("IMAGE_CDN_BASE_URL", "").rstrip("/")
LOCAL_MAX_WIDTH = int(os.environ.get("IMAGE_LOCAL_MAX_WIDTH", "900"))


def s3_key(image_path: str, prefix: str | None = None) -> str:
    """Map a relative image path to its S3 object key (prefix + normalized path).

    ``prefix`` defaults to the ``S3_PREFIX`` env var resolved at import; the
    upload script passes it explicitly so its keys can't drift from serve time.
    """
    pfx = S3_PREFIX if prefix is None else prefix.strip("/")
    rel = image_path.replace("\\", "/").lstrip("/")
    return f"{pfx}/{rel}" if pfx else rel


def s3_url(image_path: str) -> str:
    """Build the public URL for a panel, via CDN base if configured else S3."""
    key = quote(s3_key(image_path))
    if IMAGE_CDN_BASE_URL:
        return f"{IMAGE_CDN_BASE_URL}/{key}"
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def _local_data_uri(image_path: str) -> str | None:
    """Downscale a local panel and return it as a base64 JPEG data URI."""
    try:
        from PIL import Image

        p = Path(image_path)
        if not p.is_absolute():
            p = ROOT / p
        img = Image.open(p).convert("RGB")
        scale = min(1.0, LOCAL_MAX_WIDTH / img.width)
        if scale < 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, "JPEG", quality=82)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception:
        return None


def image_src(image_path: str) -> str | None:
    """Return an ``<img src>`` for a panel, or None if it can't be resolved."""
    if not image_path:
        return None
    if IMAGE_STORE == "s3":
        if not (S3_BUCKET or IMAGE_CDN_BASE_URL):
            return None
        return s3_url(image_path)
    return _local_data_uri(image_path)
