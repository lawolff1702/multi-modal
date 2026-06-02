"""Generate dense image embeddings for each panel using OpenCLIP.

Processes panels in chunks of CHUNK_SIZE, writing one parquet file per chunk.
Resumable: skips chunks whose output file already exists.
Run merge_chunks() at the end or pass --merge to produce the final parquet.
"""

from pathlib import Path
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip


MANIFEST_PATH = Path("data/comics/processed/panels_manifest.parquet")
OUTPUT_PATH = Path("data/comics/processed/image_dense_vectors.parquet")
CHUNKS_DIR = Path("data/comics/processed/image_dense_chunks")
FAILURES_PATH = Path("data/comics/processed/image_embedding_failures.csv")
CHUNK_SIZE = 10_000

_MODEL_NAME = os.environ.get("IMAGE_EMBED_MODEL", "open_clip_ViT-B-16")
_ARCH = _MODEL_NAME.replace("open_clip_", "")
_DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Lazy init so DataLoader workers don't each load the model on spawn
_model = None
_preprocess = None
_tokenizer = None


def _init_model():
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    model, _, preprocess = open_clip.create_model_and_transforms(_ARCH, pretrained="openai")
    model.eval()
    model = model.to(_DEVICE)
    _model = model
    _preprocess = preprocess
    _tokenizer = open_clip.get_tokenizer(_ARCH)


class PanelDataset(Dataset):
    def __init__(self, image_paths: list[str], preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            return self.preprocess(img), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def embed_images_batch(images: list[Image.Image]) -> list[list[float]]:
    _init_model()
    tensors = torch.stack([_preprocess(img) for img in images]).to(_DEVICE)
    with torch.no_grad():
        features = _model.encode_image(tensors)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().tolist()


def embed_text_query(text: str) -> list[float]:
    _init_model()
    tokens = _tokenizer([text]).to(_DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()


def merge_chunks() -> None:
    chunk_files = sorted(CHUNKS_DIR.glob("chunk_*.parquet"))
    if not chunk_files:
        print("No chunk files found to merge")
        return
    print(f"Merging {len(chunk_files)} chunk files...")
    df = pd.concat([pd.read_parquet(f) for f in chunk_files], ignore_index=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Merged {len(df):,} vectors → {OUTPUT_PATH}")


def done_chunks() -> set[int]:
    if not CHUNKS_DIR.exists():
        return set()
    return {int(f.stem.split("_")[1]) for f in CHUNKS_DIR.glob("chunk_*.parquet")}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--merge", action="store_true", help="Merge existing chunks and exit")
    args = parser.parse_args()

    if args.merge:
        merge_chunks()
        return

    _init_model()
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(MANIFEST_PATH)
    if args.limit:
        manifest = manifest.head(args.limit)

    already_done = done_chunks()
    total_chunks = (len(manifest) + CHUNK_SIZE - 1) // CHUNK_SIZE
    remaining = [i for i in range(total_chunks) if i not in already_done]

    print(f"Total panels:  {len(manifest):,}")
    print(f"Total chunks:  {total_chunks:,} (chunk_size={CHUNK_SIZE})")
    print(f"Done chunks:   {len(already_done):,}")
    print(f"Remaining:     {len(remaining):,}")
    print(f"Model: {_ARCH} on {_DEVICE}, batch_size={args.batch_size}, num_workers={args.num_workers}")

    all_failures = []

    for chunk_idx in remaining:
        start = chunk_idx * CHUNK_SIZE
        end = min(start + CHUNK_SIZE, len(manifest))
        chunk_df = manifest.iloc[start:end]

        panel_ids = chunk_df["panel_id"].tolist()
        image_paths = chunk_df["image_path"].tolist()

        dataset = PanelDataset(image_paths, _preprocess)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=False,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        rows = []
        failures = []
        item_offset = 0

        pbar = tqdm(loader, desc=f"chunk {chunk_idx+1}/{total_chunks}", leave=False)
        for tensors, valid_flags in pbar:
            batch_len = len(tensors)
            batch_ids = panel_ids[item_offset: item_offset + batch_len]
            batch_paths = image_paths[item_offset: item_offset + batch_len]
            item_offset += batch_len

            valid_mask = valid_flags.bool()
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            invalid_indices = (~valid_mask).nonzero(as_tuple=True)[0].tolist()

            for i in invalid_indices:
                failures.append({"panel_id": batch_ids[i], "image_path": batch_paths[i], "error": "image load failed"})

            if valid_indices.numel() > 0:
                valid_tensors = tensors[valid_indices].to(_DEVICE)
                with torch.no_grad():
                    features = _model.encode_image(valid_tensors)
                    features = features / features.norm(dim=-1, keepdim=True)
                for i, vec in enumerate(features.cpu().tolist()):
                    rows.append({"panel_id": batch_ids[valid_indices[i].item()], "image_dense": vec})

        chunk_path = CHUNKS_DIR / f"chunk_{chunk_idx:06d}.parquet"
        pd.DataFrame(rows).to_parquet(chunk_path, index=False)
        all_failures.extend(failures)

        del loader  # release worker processes and file descriptors before next chunk

        done_count = len(already_done) + (remaining.index(chunk_idx) + 1)
        print(f"  chunk {chunk_idx:06d}: {len(rows)} vectors saved ({done_count}/{total_chunks} chunks done)")

    if all_failures:
        pd.DataFrame(all_failures).to_csv(FAILURES_PATH, index=False)
        print(f"Failures: {len(all_failures)} → {FAILURES_PATH}")

    print("All chunks complete. Merging...")
    merge_chunks()


if __name__ == "__main__":
    main()
