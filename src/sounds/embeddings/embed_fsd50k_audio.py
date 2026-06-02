"""Embed FSD50K audio clips with CLAP -> fsd50k_audio_vectors.parquet.

Reads the manifest, loads + resamples each clip to 48 kHz, embeds in batches,
and writes one {sound_id, audio_dense} row per clip. Resumable: clips already
present in the output parquet are skipped, so re-running after an interruption
(or after extending the manifest) only does the new work.

Run:
    python -m src.sounds.embeddings.embed_fsd50k_audio
    python -m src.sounds.embeddings.embed_fsd50k_audio --batch-size 16
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.sounds import config
from src.sounds.embeddings.clap_model import device, embed_audio_batch, load_audio


def _load_done() -> set[str]:
    if config.AUDIO_VECTORS_PARQUET.exists():
        existing = pd.read_parquet(config.AUDIO_VECTORS_PARQUET, columns=["sound_id"])
        return set(existing["sound_id"])
    return set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed FSD50K audio with CLAP.")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    manifest = pd.read_parquet(config.MANIFEST_PARQUET)
    done = _load_done()
    todo = manifest[~manifest["sound_id"].isin(done)].reset_index(drop=True)

    print(f"Device: {device()}")
    print(f"Manifest: {len(manifest)} clips | already embedded: {len(done)} | to embed: {len(todo)}")
    if todo.empty:
        print("Nothing to do.")
        return

    new_rows: list[dict] = []
    failed: list[str] = []
    batch_arrays: list = []
    batch_ids: list[str] = []

    def flush() -> None:
        if not batch_arrays:
            return
        vectors = embed_audio_batch(batch_arrays)
        for sid, vec in zip(batch_ids, vectors):
            new_rows.append({"sound_id": sid, "audio_dense": vec})
        batch_arrays.clear()
        batch_ids.clear()

    for i, row in enumerate(todo.itertuples(index=False), start=1):
        try:
            audio = load_audio(row.audio_path)
            batch_arrays.append(audio)
            batch_ids.append(row.sound_id)
        except Exception as exc:
            failed.append(row.sound_id)
            print(f"  [load fail] {row.sound_id}: {exc}")

        if len(batch_arrays) >= args.batch_size:
            flush()
        if i % 100 == 0:
            print(f"  ...{i}/{len(todo)} processed")

    flush()

    if failed:
        print(f"Failed to load {len(failed)} clips: {failed[:10]}{' ...' if len(failed) > 10 else ''}")

    out = pd.DataFrame(new_rows)
    if config.AUDIO_VECTORS_PARQUET.exists():
        prior = pd.read_parquet(config.AUDIO_VECTORS_PARQUET)
        out = pd.concat([prior, out], ignore_index=True)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.AUDIO_VECTORS_PARQUET, index=False)
    print(f"\nWrote {len(out)} total vectors -> {config.AUDIO_VECTORS_PARQUET}")
    print(f"  (added {len(new_rows)} this run; dim={len(new_rows[0]['audio_dense']) if new_rows else 'n/a'})")


if __name__ == "__main__":
    main()
