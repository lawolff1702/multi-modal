"""Shared embedding helpers for all notebooks."""
import base64
import os
from io import BytesIO
from pathlib import Path

import voyageai
from PIL import Image


def _image_to_base64(image_path: str | Path) -> str:
    img = Image.open(image_path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def embed_images_voyage(
    image_paths: list[str | Path],
    batch_size: int = 10,
) -> list[list[float]]:
    """Embed comic page images with voyage-multimodal-3. Returns list of 1024-dim vectors."""
    vo = voyageai.Client()
    all_vectors = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        inputs = []
        for p in batch:
            img = Image.open(p).convert("RGB")
            inputs.append([img])
        result = vo.multimodal_embed(inputs, model="voyage-multimodal-3", input_type="document")
        all_vectors.extend(result.embeddings)
    return all_vectors


def embed_query_voyage(query_text: str) -> list[float]:
    """Embed a text query with voyage-multimodal-3 for cross-modal search."""
    vo = voyageai.Client()
    result = vo.multimodal_embed([[query_text]], model="voyage-multimodal-3", input_type="query")
    return result.embeddings[0]


def embed_query_image_voyage(image_path: str | Path) -> list[float]:
    """Embed an image as a query with voyage-multimodal-3 for image-to-image search."""
    vo = voyageai.Client()
    img = Image.open(image_path).convert("RGB")
    result = vo.multimodal_embed([[img]], model="voyage-multimodal-3", input_type="query")
    return result.embeddings[0]
