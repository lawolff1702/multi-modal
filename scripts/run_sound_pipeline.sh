#!/usr/bin/env bash
set -euo pipefail

# FSD50K sound-effect index pipeline (dense-only, CLAP).
# Mirrors run_pipeline.sh. v1 builds the commercial-safe EVAL split.
#
#   SOUND_SPLIT=eval        which FSD50K split to ingest (eval|dev)
#   SOUND_LIMIT=500         cap clips for a dry run (unset = full split)

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

set -a && source .env && set +a

PYTHON=".venv/bin/python"
SPLIT="${SOUND_SPLIT:-eval}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

LIMIT_ARG=""
[ -n "${SOUND_LIMIT:-}" ] && LIMIT_ARG="--limit ${SOUND_LIMIT}"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/sound_pipeline.log"; }

log "=== Sound pipeline start (split=$SPLIT ${LIMIT_ARG:-full}) ==="

log "Step 1: build_fsd50k_manifest"
$PYTHON -m src.sounds.ingest.build_fsd50k_manifest --split "$SPLIT" $LIMIT_ARG 2>&1 | tee -a "$LOG_DIR/build_fsd50k_manifest.log"
log "Step 1 done"

log "Step 2: embed_fsd50k_audio (CLAP)"
$PYTHON -m src.sounds.embeddings.embed_fsd50k_audio 2>&1 | tee -a "$LOG_DIR/embed_fsd50k_audio.log"
log "Step 2 done"

log "Step 3: create_sound_index (skips if exists)"
$PYTHON -m src.sounds.pinecone.create_sound_index 2>&1 | tee -a "$LOG_DIR/create_sound_index.log"
log "Step 3 done"

log "Step 4: build_sound_vectors"
$PYTHON -m src.sounds.pinecone.build_sound_vectors 2>&1 | tee -a "$LOG_DIR/build_sound_vectors.log"
log "Step 4 done"

log "Step 5: upsert_sounds"
$PYTHON -m src.sounds.pinecone.upsert_sounds 2>&1 | tee -a "$LOG_DIR/upsert_sounds.log"
log "Step 5 done"

log "=== Sound pipeline complete ==="
