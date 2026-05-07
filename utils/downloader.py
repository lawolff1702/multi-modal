"""Download comic archives from URLs to data/raw/."""
import hashlib
from pathlib import Path

import requests


def download_comic(url: str, raw_dir: Path, filename: str | None = None) -> Path:
    """Download a CBZ/ZIP from url into raw_dir. Returns the local path."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0] or hashlib.md5(url.encode()).hexdigest() + ".cbz"
    dest = raw_dir / filename
    if dest.exists():
        print(f"  already downloaded: {dest.name}")
        return dest
    print(f"  downloading {filename}...")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    print(f"  saved → {dest}")
    return dest
