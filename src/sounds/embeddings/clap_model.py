"""CLAP embedding wrapper (laion/clap-htsat-unfused).

Produces L2-normalized 512-dim vectors in a shared audio/text space, so a CLAP
text query can be matched against CLAP audio vectors with cosine similarity.

Mirrors the lazy-init + MPS/CUDA/CPU device convention from
src/embeddings/embed_images.py. The model was trained at 48 kHz; FSD50K is
44.1 kHz, so `load_audio` resamples every clip to 48 kHz with soxr before
embedding. Resample once, here — do not rely on the processor to do it.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, ClapModel

from src.sounds import config

_MODEL_NAME = config.CLAP_MODEL_NAME
_SR = config.CLAP_SAMPLE_RATE

# Same precedence as the image embedder; overridable for MPS-quirk fallback.
_DEVICE = os.environ.get("SOUND_EMBED_DEVICE") or (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

_model: ClapModel | None = None
_processor = None


def _init_model() -> None:
    global _model, _processor
    if _model is not None:
        return
    _processor = AutoProcessor.from_pretrained(_MODEL_NAME)
    model = ClapModel.from_pretrained(_MODEL_NAME).to(_DEVICE)
    model.eval()
    _model = model


def _pool(out):
    """get_{audio,text}_features returns a bare tensor (transformers <5) or a
    pooled-output object whose `pooler_output` is the projected 512-d embedding
    (transformers >=5). Normalize to a tensor here."""
    return out.pooler_output if hasattr(out, "pooler_output") else out


def load_audio(path: str, target_sr: int = _SR) -> np.ndarray:
    """Load a wav as mono float32 resampled to target_sr (1-D array)."""
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:                       # stereo -> mono
        data = data.mean(axis=1)
    if sr != target_sr:
        import soxr

        data = soxr.resample(data, sr, target_sr)
    return np.ascontiguousarray(data, dtype=np.float32)


@torch.no_grad()
def embed_text_batch(texts: list[str]) -> list[list[float]]:
    _init_model()
    inputs = _processor(text=texts, return_tensors="pt", padding=True).to(_DEVICE)
    vectors = _pool(_model.get_text_features(**inputs))
    vectors = F.normalize(vectors, p=2, dim=-1)
    return vectors.cpu().numpy().tolist()


@torch.no_grad()
def embed_audio_batch(audio_arrays: list[np.ndarray], sampling_rate: int = _SR) -> list[list[float]]:
    """Embed a batch of mono float32 arrays that are ALREADY at `sampling_rate`."""
    _init_model()
    inputs = _processor(
        audio=audio_arrays,  # transformers >=5 renamed this from `audios`
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    ).to(_DEVICE)
    vectors = _pool(_model.get_audio_features(**inputs))
    vectors = F.normalize(vectors, p=2, dim=-1)
    return vectors.cpu().numpy().tolist()


def device() -> str:
    return _DEVICE
