#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

set -a && source .env && set +a

PYTHON=".venv/bin/python"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/pipeline.log"; }

log "=== Pipeline start ==="

log "Step 1: build_manifest"
$PYTHON -m src.ingest.build_manifest 2>&1 | tee -a "$LOG_DIR/build_manifest.log"
log "Step 1 done"

log "Step 2: embed_images"
$PYTHON -m src.embeddings.embed_images 2>&1 | tee -a "$LOG_DIR/embed_images.log"
log "Step 2 done"

log "Step 3: embed_sparse_text"
$PYTHON -m src.embeddings.embed_sparse_text 2>&1 | tee -a "$LOG_DIR/embed_sparse_text.log"
log "Step 3 done"

log "Step 4: create_index (skips if exists)"
$PYTHON -m src.pinecone.create_index 2>&1 | tee -a "$LOG_DIR/create_index.log"
log "Step 4 done"

log "Step 5: build_documents"
$PYTHON -m src.pinecone.build_documents 2>&1 | tee -a "$LOG_DIR/build_documents.log"
log "Step 5 done"

log "Step 6: upsert_panels"
$PYTHON -m src.pinecone.upsert_panels 2>&1 | tee -a "$LOG_DIR/upsert_panels.log"
log "Step 6 done"

log "=== Pipeline complete ==="
