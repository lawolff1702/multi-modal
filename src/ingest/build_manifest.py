"""Build a panel manifest from COMICS raw panel images and OCR CSV."""

from pathlib import Path
import hashlib
import re
import pandas as pd

from src.ingest.clean_text import clean_ocr, build_search_text


COMICS_DATA_DIR = Path("data/comics")
RAW_DIR = COMICS_DATA_DIR / "raw"
PROCESSED_DIR = COMICS_DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def parse_panel_path(image_path: Path) -> dict:
    """
    Infer comic_id, page_num, panel_num from an image path.

    COMICS raw_panel_images layout (from dataset inspection):
      raw_panel_images/{book_id}/{page_num:04d}_{panel_num:04d}.jpg

    Returns partial metadata; fields are None if pattern doesn't match.
    """
    parts = image_path.parts
    # Expect: ...raw_panel_images/<book_id>/<filename>
    try:
        panel_dir_idx = next(i for i, p in enumerate(parts) if p == "raw_panel_images")
        book_id = parts[panel_dir_idx + 1] if panel_dir_idx + 1 < len(parts) else None
    except StopIteration:
        book_id = None

    page_num, panel_num = None, None
    m = re.match(r"^(\d+)_(\d+)\.(jpg|jpeg|png)$", image_path.name, re.IGNORECASE)
    if m:
        page_num = int(m.group(1))
        panel_num = int(m.group(2))

    comic_id = book_id
    page_id = f"{book_id}_p{page_num:04d}" if book_id and page_num is not None else None
    panel_id = (
        f"{comic_id}:{page_num}:{panel_num}"
        if comic_id and page_num is not None and panel_num is not None
        else f"comics:{stable_hash(str(image_path))}"
    )

    return {
        "panel_id": panel_id,
        "comic_id": comic_id,
        "book_id": book_id,
        "page_id": page_id,
        "page_num": page_num,
        "panel_num": panel_num,
    }


def load_ad_pages(ad_pages_path: Path) -> set[str]:
    """Load advertisement page identifiers from predadpages.txt."""
    if not ad_pages_path.exists():
        print(f"  Warning: {ad_pages_path} not found; no ad-page flags will be set")
        return set()
    lines = ad_pages_path.read_text().splitlines()
    return {line.strip() for line in lines if line.strip()}


def build_ocr_lookup(ocr_df: pd.DataFrame) -> dict[str, str]:
    """
    Build a mapping from panel_id keys to OCR text.

    After inspecting the OCR CSV columns, this function maps panel image
    paths to their OCR text. The COMICS OCR CSV format uses columns:
      comic_no, page_no, panel_no, ocr  (or similar)

    Prints column info on first run so you can verify the mapping is correct.
    """
    print("\nOCR CSV columns:", list(ocr_df.columns))
    print("OCR CSV shape:", ocr_df.shape)
    print("OCR CSV sample rows:")
    print(ocr_df.head(3).to_string())
    print()

    # Best-effort mapping based on common COMICS dataset column names.
    # Adjust the column names below after inspecting the output above.
    lookup: dict[str, str] = {}

    # Try to find comic/book, page, panel, and text columns
    col_map = {c.lower(): c for c in ocr_df.columns}

    book_col = next((col_map[k] for k in ["comic_no", "book_id", "comic_id", "book_no"] if k in col_map), None)
    page_col = next((col_map[k] for k in ["page_no", "page_num", "page_id", "page"] if k in col_map), None)
    panel_col = next((col_map[k] for k in ["panel_no", "panel_num", "panel_id", "panel"] if k in col_map), None)
    text_col = next((col_map[k] for k in ["ocr", "text", "ocr_text", "dialogue"] if k in col_map), None)

    if not all([book_col, page_col, panel_col, text_col]):
        print(f"  Warning: Could not auto-map all OCR columns.")
        print(f"  book_col={book_col}, page_col={page_col}, panel_col={panel_col}, text_col={text_col}")
        print("  Update build_ocr_lookup() with the correct column names after inspection.")
        return lookup

    print(f"  Using columns: book={book_col}, page={page_col}, panel={panel_col}, text={text_col}")

    for row in ocr_df.itertuples(index=False):
        book = str(getattr(row, book_col)).strip()
        try:
            page = int(getattr(row, page_col))
            panel = int(getattr(row, panel_col))
        except (ValueError, TypeError):
            continue
        text = str(getattr(row, text_col)) if getattr(row, text_col, None) is not None else ""
        key = f"{book}:{page}:{panel}"
        lookup[key] = text

    print(f"  OCR lookup entries: {len(lookup)}")
    return lookup


def main():
    panel_dir = RAW_DIR / "raw_panel_images"
    ocr_path = RAW_DIR / "COMICS_ocr_file.csv"
    ad_pages_path = RAW_DIR / "predadpages.txt"

    if not panel_dir.exists():
        raise FileNotFoundError(
            f"Panel image directory not found: {panel_dir}\n"
            "Run the download step first:\n"
            "  mkdir -p data/comics/raw && cd data/comics/raw\n"
            "  wget -c https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz\n"
            "  tar -xzf raw_panel_images.tar.gz"
        )

    panel_paths = sorted(
        p for p in panel_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    print(f"Found {len(panel_paths)} panel images")

    ocr_lookup: dict[str, str] = {}
    if ocr_path.exists():
        ocr_df = pd.read_csv(ocr_path)
        ocr_lookup = build_ocr_lookup(ocr_df)
    else:
        print(f"Warning: OCR file not found at {ocr_path}; all panels will have empty OCR")

    ad_pages = load_ad_pages(ad_pages_path)

    rows = []
    unmatched_images = []

    for image_path in panel_paths:
        meta = parse_panel_path(image_path)
        panel_id = meta["panel_id"]

        ocr_text_raw = ocr_lookup.get(panel_id, "")
        if not ocr_text_raw and meta["comic_id"] and meta["page_num"] is not None:
            # Try zero-padded fallback keys
            for fmt in [f"{meta['comic_id']}:{meta['page_num']:04d}:{meta['panel_num']:04d}"]:
                if fmt in ocr_lookup:
                    ocr_text_raw = ocr_lookup[fmt]
                    break

        if not ocr_text_raw:
            unmatched_images.append(str(image_path))

        ocr_text_clean = clean_ocr(ocr_text_raw)
        search_text = build_search_text(ocr_text_clean)

        # Check if this is an ad page
        is_ad = (
            meta["page_id"] in ad_pages
            or str(image_path) in ad_pages
            or image_path.name in ad_pages
        )

        rows.append({
            "panel_id": panel_id,
            "image_path": str(image_path),
            "ocr_text_raw": ocr_text_raw,
            "ocr_text_clean": ocr_text_clean,
            "search_text": search_text,
            "comic_id": meta["comic_id"],
            "book_id": meta["book_id"],
            "page_id": meta["page_id"],
            "page_num": meta["page_num"],
            "panel_num": meta["panel_num"],
            "is_ad_page": is_ad,
            "source": "COMICS",
        })

    manifest = pd.DataFrame(rows)

    # Validate: all panel_ids must be unique
    dupes = manifest["panel_id"].duplicated().sum()
    if dupes:
        print(f"Warning: {dupes} duplicate panel_ids found; using path-hash fallback for those rows")
        dup_mask = manifest["panel_id"].duplicated(keep=False)
        manifest.loc[dup_mask, "panel_id"] = manifest.loc[dup_mask, "image_path"].apply(
            lambda p: f"comics:{stable_hash(p)}"
        )

    manifest.to_parquet(PROCESSED_DIR / "panels_manifest.parquet", index=False)
    manifest.to_json(PROCESSED_DIR / "panels_manifest.jsonl", orient="records", lines=True)

    print(f"\nManifest written: {len(manifest)} panels")
    print(f"  With OCR text: {(manifest['ocr_text_clean'] != '').sum()}")
    print(f"  Empty OCR:     {(manifest['ocr_text_clean'] == '').sum()}")
    print(f"  Ad pages:      {manifest['is_ad_page'].sum()}")
    print(f"  Unmatched OCR: {len(unmatched_images)}")

    if unmatched_images[:5]:
        print(f"  Sample unmatched: {unmatched_images[:5]}")

    unmatched_ocr_keys = set(ocr_lookup) - {r["panel_id"] for r in rows}
    if unmatched_ocr_keys:
        print(f"  OCR rows with no matching image: {len(unmatched_ocr_keys)}")


if __name__ == "__main__":
    main()
