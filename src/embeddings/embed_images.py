"""Generate dense image embeddings for each panel using OpenCLIP."""

from pathlib import Path
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
import open_clip


MANIFEST_PATH = Path("data/comics/processed/panels_manifest.parquet")
OUTPUT_PATH = Path("data/comics/processed/image_dense_vectors.parquet")
FAILURES_PATH = Path("data/comics/processed/image_embedding_failures.csv")

_MODEL_NAME = os.environ.get("IMAGE_EMBED_MODEL", "open_clip_ViT-B-32")
_ARCH = _MODEL_NAME.replace("open_clip_", "")
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model, _, _preprocess = open_clip.create_model_and_transforms(_ARCH, pretrained="openai")
_model.eval()
_model = _model.to(_DEVICE)
_tokenizer = open_clip.get_tokenizer(_ARCH)


def embed_image(image: Image.Image) -> list[float]:
    tensor = _preprocess(image).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        features = _model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()


def embed_text_query(text: str) -> list[float]:
    """Embed a text query into the same vector space as panel images."""
    tokens = _tokenizer([text]).to(_DEVICE)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().tolist()


def main():
    manifest = pd.read_parquet(MANIFEST_PATH)
    print(f"Embedding {len(manifest)} panels with {_ARCH} on {_DEVICE}")

    rows = []
    failures = []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        try:
            image = Image.open(row.image_path).convert("RGB")
            vector = embed_image(image)
            rows.append({"panel_id": row.panel_id, "image_dense": vector})
        except Exception as exc:
            failures.append({
                "panel_id": row.panel_id,
                "image_path": row.image_path,
                "error": str(exc),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(rows)} vectors → {OUTPUT_PATH}")

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_PATH, index=False)
        print(f"Failed: {len(failures)} panels → {FAILURES_PATH}")


if __name__ == "__main__":
    main()
