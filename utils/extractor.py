"""Extract pages from CBZ/ZIP comic archives into data/pages/<comic_slug>/."""
import zipfile
from pathlib import Path


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def extract_comic(archive_path: Path, pages_dir: Path) -> list[dict]:
    """Extract image pages from a CBZ/ZIP archive. Returns a list of page dicts."""
    slug = slugify(archive_path.stem)
    out_dir = pages_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    pages = []

    with zipfile.ZipFile(archive_path) as zf:
        image_names = sorted(
            n for n in zf.namelist()
            if Path(n).suffix.lower() in image_exts and not Path(n).name.startswith(".")
        )
        for i, name in enumerate(image_names, start=1):
            suffix = Path(name).suffix.lower()
            dest = out_dir / f"page_{i:03d}{suffix}"
            if not dest.exists():
                dest.write_bytes(zf.read(name))
            pages.append({
                "comic_slug": slug,
                "comic_title": archive_path.stem,
                "page_num": i,
                "file_path": str(dest),
            })

    return pages
