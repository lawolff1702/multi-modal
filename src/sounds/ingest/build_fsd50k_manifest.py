"""Build a commercial-safe manifest of FSD50K clips.

The HF dataset `Fhrozen/FSD50k` stores clips as individual wav files
(`clips/{dev,eval}/{freesound_id}.wav`) plus small metadata/label files. We pull
those directly with `huggingface_hub` rather than `datasets.load_dataset`,
because:

  * `Fhrozen/FSD50k` loads via a remote `fsd50k.py` script, which modern
    `datasets` (>=3.0) no longer executes — and this repo runs Python 3.13.
  * Individual-file access lets us download ONLY the commercial-safe clips we
    keep, and gives stable local paths for metadata + notebook playback.

License lives in `metadata/{split}_clips_info_FSD50K.json` as a Creative Commons
URL (NOT a clean "CC0"/"CC-BY" string), keyed by Freesound clip id. Labels live
in `labels/{split}.csv` keyed by the same id (== wav filename).

Run:
    python -m src.sounds.ingest.build_fsd50k_manifest --split eval
    python -m src.sounds.ingest.build_fsd50k_manifest --split eval --limit 25   # quick verify
    python -m src.sounds.ingest.build_fsd50k_manifest --split eval --no-audio   # metadata only
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from src.sounds import config

REPO = config.SOUND_DATASET_NAME
REPO_TYPE = "dataset"


# --------------------------------------------------------------------------- #
# License handling
# --------------------------------------------------------------------------- #
def normalize_license(url: str | None) -> str:
    """Map a Creative Commons license URL to a canonical short name.

    Order matters: the restrictive variants (by-nc/by-sa/by-nd) all contain
    "by", so they must be checked before plain CC-BY.
    """
    if not url:
        return "unknown"
    u = url.lower()
    if "publicdomain/zero" in u or "/zero/" in u:
        return "CC0"
    if "sampling" in u:
        return "CC Sampling+"
    if "by-nc" in u:
        return "CC-BY-NC"
    if "by-sa" in u:
        return "CC-BY-SA"
    if "by-nd" in u:
        return "CC-BY-ND"
    if "/by/" in u or u.rstrip("/").endswith("/by"):
        return "CC-BY"
    return "unknown"


def is_commercial_safe(license_name: str) -> bool:
    return license_name in config.COMMERCIAL_SAFE_LICENSES


def requires_attribution(license_name: str) -> bool:
    return license_name == "CC-BY"


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #
def _dl(path: str) -> str:
    return hf_hub_download(REPO, path, repo_type=REPO_TYPE)


def load_clip_info(split: str) -> dict[str, dict]:
    """{freesound_id: {title, description, tags, license, uploader}}."""
    path = _dl(f"metadata/{split}_clips_info_FSD50K.json")
    with open(path) as f:
        return json.load(f)


def load_label_lookup(split: str) -> dict[str, dict]:
    """{fname: {"labels": [...], "label_ids": [...mids...]}} from labels/{split}.csv."""
    path = _dl(f"labels/{split}.csv")
    lookup: dict[str, dict] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = str(row["fname"]).strip()
            labels = [s for s in (row.get("labels") or "").split(",") if s]
            mids = [s for s in (row.get("mids") or "").split(",") if s]
            lookup[fname] = {"labels": labels, "label_ids": mids}
    return lookup


def download_clip(split: str, clip_id: str) -> str:
    """Download a single wav; returns a stable local (HF cache) path."""
    return _dl(f"clips/{split}/{clip_id}.wav")


def audio_duration(path: str) -> float | None:
    try:
        import soundfile as sf

        info = sf.info(path)
        return round(info.frames / info.samplerate, 3)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Manifest build
# --------------------------------------------------------------------------- #
def build_manifest(split: str, limit: int | None, download_audio: bool) -> pd.DataFrame:
    info = load_clip_info(split)
    labels = load_label_lookup(split)

    # License counts over the FULL split (cheap; metadata only) for reporting.
    license_counts: Counter[str] = Counter()
    for v in info.values():
        license_counts[normalize_license(v.get("license"))] += 1

    safe_ids = [cid for cid, v in info.items() if is_commercial_safe(normalize_license(v.get("license")))]
    safe_ids.sort(key=lambda c: int(c) if c.isdigit() else c)

    print("=" * 60)
    print(f"Split: {split}")
    print(f"Total clips inspected: {len(info)}")
    print(f"Kept commercial-safe clips: {len(safe_ids)} "
          f"(CC0={license_counts.get('CC0', 0)}, CC-BY={license_counts.get('CC-BY', 0)})")
    print(f"Excluded CC-BY-NC: {license_counts.get('CC-BY-NC', 0)}")
    print(f"Excluded CC Sampling+: {license_counts.get('CC Sampling+', 0)}")
    other = {k: c for k, c in license_counts.items()
             if k not in {"CC0", "CC-BY", "CC-BY-NC", "CC Sampling+"}}
    print(f"Excluded other/unknown: {sum(other.values())} {dict(other) if other else ''}")
    print("=" * 60)

    selected = safe_ids[:limit] if limit else safe_ids
    if limit:
        print(f"Building {len(selected)} of {len(safe_ids)} rows (--limit {limit}).")

    rows = []
    n_audio_ok = 0
    n_audio_fail = 0
    for i, clip_id in enumerate(selected, start=1):
        meta = info[clip_id]
        lic = normalize_license(meta.get("license"))
        lab = labels.get(clip_id, {"labels": [], "label_ids": []})

        audio_path = None
        duration = None
        if download_audio:
            try:
                audio_path = download_clip(split, clip_id)
                duration = audio_duration(audio_path)
                n_audio_ok += 1
            except Exception as exc:  # network / missing file
                n_audio_fail += 1
                print(f"  [audio fail] {clip_id}: {exc}")

        attribution = None
        if requires_attribution(lic):
            attribution = f"{meta.get('uploader', 'unknown')} (Freesound) — {meta.get('license')}"

        rows.append({
            "sound_id": f"fsd50k:{clip_id}",
            "source": "FSD50K",
            "source_id": clip_id,
            "dataset_name": REPO,
            "split": split,
            "audio_path": audio_path,
            "labels": lab["labels"],
            "label_ids": lab["label_ids"],
            "title": meta.get("title"),
            "description": meta.get("description") or "",
            "tags": meta.get("tags") or [],
            "license": lic,
            "license_ok_for_commercial": True,
            "requires_attribution": requires_attribution(lic),
            "attribution": attribution,
            "duration_sec": duration,
        })

        if download_audio and i % 500 == 0:
            print(f"  ...{i}/{len(selected)} clips downloaded")

    if download_audio:
        print(f"Audio: {n_audio_ok} downloaded, {n_audio_fail} failed.")

    df = pd.DataFrame(rows)
    # Validation: every row must have a commercial-safe license + a sound_id.
    assert df["sound_id"].notna().all(), "rows missing sound_id"
    assert df["license"].isin(config.COMMERCIAL_SAFE_LICENSES).all(), "non-safe license leaked in"
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build commercial-safe FSD50K manifest.")
    parser.add_argument("--split", default="eval", choices=["eval", "dev"],
                        help="FSD50K split (v1 validates on eval first).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of commercial-safe clips (for quick verification).")
    parser.add_argument("--no-audio", dest="download_audio", action="store_false",
                        help="Skip downloading wavs (metadata-only manifest).")
    parser.set_defaults(download_audio=True)
    args = parser.parse_args()

    df = build_manifest(args.split, args.limit, args.download_audio)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.MANIFEST_PARQUET, index=False)
    df.to_json(config.MANIFEST_JSONL, orient="records", lines=True)
    print(f"\nWrote {len(df)} rows:")
    print(f"  {config.MANIFEST_PARQUET}")
    print(f"  {config.MANIFEST_JSONL}")

    if len(df):
        print("\n=== sample row ===")
        print(json.dumps(df.iloc[0].to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
