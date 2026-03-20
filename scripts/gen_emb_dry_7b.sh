#!/bin/bash

# ============================================================================
# LLaVA 7B Multi-GPU Embedding Generation Script
# ============================================================================
#
# 실행 방법: bash scripts/gen_emb_7b.sh
# (PPD 루트 디렉토리에서 실행하세요)

NUM_CHUNKS=1000
WHICH_CHUNK=0
OUTPUT_DIR="./data/emb_test_7b"
NUM_SHOTS=2
PER_USER=1

echo "[INFO] Starting LLaVA 7B (Multi-GPU) embedding generation for chunk ${WHICH_CHUNK}/${NUM_CHUNKS}..."

python llava_embeddings/pick_a_pick_user_emb_7b.py \
  --device_map auto \
  --pretrained lmms-lab/llava-onevision-qwen2-7b-ov-chat \
  --num_shots ${NUM_SHOTS} \
  --num_chunks ${NUM_CHUNKS} \
  --which_chunk ${WHICH_CHUNK} \
  --per_user ${PER_USER} \
  --save_every 1 \
  --output_dir ${OUTPUT_DIR}

echo "[INFO] Execution completed successfully!"
