"""Configuration for the FSD50K sound-effect index.

Mirrors the env-var + sensible-default convention used elsewhere in the repo
(see src/embeddings/embed_images.py). Kept separate from the comic-panel
config so the two indexes never share state.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env so `python -m src.sounds...` works without a pre-sourced shell.
load_dotenv()

# --- Pinecone (sound index is SEPARATE from comic-panels) -------------------
PINECONE_SOUND_INDEX_NAME = os.environ.get("PINECONE_SOUND_INDEX_NAME", "comic-sounds")
PINECONE_SOUND_NAMESPACE = os.environ.get("PINECONE_SOUND_NAMESPACE", "fsd50k-commercial-v1")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")

# --- Dataset ----------------------------------------------------------------
SOUND_DATASET_NAME = os.environ.get("SOUND_DATASET_NAME", "Fhrozen/FSD50k")
SOUND_DATA_DIR = Path(os.environ.get("SOUND_DATA_DIR", "data/sounds/fsd50k"))
SOUND_LICENSE_MODE = os.environ.get("SOUND_LICENSE_MODE", "commercial_safe")

RAW_DIR = SOUND_DATA_DIR / "raw"
PROCESSED_DIR = SOUND_DATA_DIR / "processed"
MANIFEST_PARQUET = PROCESSED_DIR / "fsd50k_manifest.parquet"
MANIFEST_JSONL = PROCESSED_DIR / "fsd50k_manifest.jsonl"
AUDIO_VECTORS_PARQUET = PROCESSED_DIR / "fsd50k_audio_vectors.parquet"
SOUND_VECTORS_JSONL = PROCESSED_DIR / "fsd50k_sound_vectors.jsonl"

# --- CLAP model -------------------------------------------------------------
CLAP_MODEL_NAME = os.environ.get("CLAP_MODEL_NAME", "laion/clap-htsat-unfused")
CLAP_EMBED_DIM = int(os.environ.get("CLAP_EMBED_DIM", "512"))
# laion/clap-htsat-unfused was trained on 48 kHz audio; FSD50K is 44.1 kHz, so
# everything is resampled to this rate before embedding.
CLAP_SAMPLE_RATE = 48_000

# --- License policy ---------------------------------------------------------
# Commercial-safe means clips we can reuse commercially. FSD50K clips carry one
# of four Creative Commons licenses; only these two are unrestricted enough.
COMMERCIAL_SAFE_LICENSES = {"CC0", "CC-BY"}
