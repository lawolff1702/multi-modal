#!/usr/bin/env python3
"""Upload comic panel images to S3, preserving the relative path layout.

Each object's key mirrors the file's path relative to the project root, so it
matches what ``src.storage.images.s3_key()`` builds at serve time — no extra
mapping needed between upload and lookup.

The source set is ~64GB across ~1.2M files. ``--resize`` downscales each panel
to ``--max-width`` (the UI only ever displays ~900px) and re-encodes JPEG q82 on
the way up, which shrinks both storage and per-card load time substantially.

Credentials come from the standard AWS chain (env vars, shared config, or an
instance profile). Example:

    python scripts/upload_images_to_s3.py \\
        --bucket my-comics-bucket --prefix panels --region us-east-1 \\
        --resize --workers 32
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_IMAGES_DIR = ROOT / "data" / "comics" / "raw" / "raw_panel_images"


def _resized_jpeg(path: Path, max_width: int) -> bytes:
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    scale = min(1.0, max_width / img.width)
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def _exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _upload_one(s3, bucket: str, path: Path, key: str, *, resize: bool,
                max_width: int, acl: str | None, skip_existing: bool) -> str:
    if skip_existing and _exists(s3, bucket, key):
        return "skipped"
    # Panels are immutable, so cache them effectively forever (ideal for CloudFront).
    extra = {"ContentType": "image/jpeg",
             "CacheControl": "public, max-age=31536000, immutable"}
    if acl:
        extra["ACL"] = acl
    if resize:
        s3.put_object(Bucket=bucket, Key=key, Body=_resized_jpeg(path, max_width), **extra)
    else:
        s3.upload_file(str(path), bucket, key, ExtraArgs=extra)
    return "uploaded"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", required=True, help="Target S3 bucket name")
    ap.add_argument("--prefix", default="", help="Key prefix (must match S3_PREFIX at serve time)")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    ap.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--resize", action="store_true", help="Downscale to --max-width before upload")
    ap.add_argument("--max-width", type=int, default=900)
    ap.add_argument("--acl", default=None,
                    help="e.g. public-read. Omit on modern buckets (Object Ownership = "
                         "Bucket owner enforced rejects ACLs); use a bucket policy or CloudFront OAC instead.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true",
                    help="head_object before each upload (slower; for resuming)")
    ap.add_argument("--limit", type=int, default=0, help="Upload at most N files (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="List what would upload, then exit")
    args = ap.parse_args()

    from src.storage.images import s3_key

    if not args.images_dir.is_dir():
        print(f"Images dir not found: {args.images_dir}", file=sys.stderr)
        return 1

    print(f"Scanning {args.images_dir} …")
    files = [p for p in args.images_dir.rglob("*") if p.is_file()]
    if args.limit:
        files = files[: args.limit]
    print(f"{len(files):,} files to process "
          f"(resize={args.resize}, workers={args.workers}, prefix={args.prefix!r})")

    if args.dry_run:
        for p in files[:10]:
            print(f"  {p.relative_to(ROOT)}  ->  s3://{args.bucket}/{s3_key(str(p.relative_to(ROOT)), prefix=args.prefix)}")
        print("  …" if len(files) > 10 else "")
        return 0

    import boto3
    from tqdm import tqdm

    s3 = boto3.client("s3", region_name=args.region)
    counts = {"uploaded": 0, "skipped": 0}
    failures: list[tuple[str, str]] = []

    def job(path: Path) -> str:
        key = s3_key(str(path.relative_to(ROOT)), prefix=args.prefix)
        return _upload_one(s3, args.bucket, path, key, resize=args.resize,
                           max_width=args.max_width, acl=args.acl,
                           skip_existing=args.skip_existing)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, p): p for p in files}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="img"):
            p = futures[fut]
            try:
                counts[fut.result()] += 1
            except Exception as exc:  # noqa: BLE001 - report and continue
                failures.append((str(p.relative_to(ROOT)), str(exc)))

    print(f"\nuploaded={counts['uploaded']:,}  skipped={counts['skipped']:,}  failed={len(failures):,}")
    if failures:
        out = ROOT / "logs" / "s3_upload_failures.csv"
        out.parent.mkdir(exist_ok=True)
        out.write_text("path,error\n" + "\n".join(f'"{p}","{e}"' for p, e in failures))
        print(f"Wrote {len(failures):,} failures to {out}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
