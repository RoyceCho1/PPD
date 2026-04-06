#!/bin/bash
set -euo pipefail

# 기존 환경에서 생성하던 User Embedding(총 100 청크 중 0~32 완료)을 현재 서버(단일 GPU, RTX 5090)에서
# 이어서(33~99) 생성하기 위해 작성된 스크립트입니다.

# Conda 환경 초기화 (기본 설정)
source /data/roycecho/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ppd

# ==========================================
# 사용자 설정 변수 (단일 GPU)
# ==========================================
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/data/roycecho/.cache/huggingface"

# 이전 구성에 맞춰 전체 100조각(num_chunks=100)으로 나눈 뒤 33번째부터 시작하도록 설정
OUTPUT_DIR="${OUTPUT_DIR:-/data/roycecho/PPD/data/user_emb_7b_full}"
NUM_SHOTS="${NUM_SHOTS:-4}"
PER_USER="${PER_USER:-20}"        
DEVICE_MAP="${DEVICE_MAP:-none}" 

NUM_CHUNKS=100
START_CHUNK=44
END_CHUNK=80

# ==========================================
# 실행
# ==========================================
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "    [RESUMING PPD USER EMBEDDING GENERATION (7B)]"
echo "=========================================================="
echo "- 모델: LLaVA-1.5 / Qwen2 7B"
echo "- 출력 디렉토리: ${OUTPUT_DIR}"
echo "- 청크 설정: 전체 ${NUM_CHUNKS}조각 기준으로 ${START_CHUNK}부터 $(($END_CHUNK - 1))까지 이어서 처리"
echo "- 사용 환경: RTX 5090 단일 GPU 설정"
echo "=========================================================="

for (( which_chunk=$START_CHUNK; which_chunk<$END_CHUNK; which_chunk++ )); do
    echo ">> [INFO] Running chunk ${which_chunk}"
    python llava_embeddings/pick_a_pick_user_emb_7b.py \
        --device cuda:0 \
        --device_map "${DEVICE_MAP}" \
        --num_shots "${NUM_SHOTS}" \
        --num_chunks "${NUM_CHUNKS}" \
        --which_chunk "${which_chunk}" \
        --per_user "${PER_USER}" \
        --output_dir "${OUTPUT_DIR}"
done

echo "User Embedding 이어서 생성이 모두 완료되었습니다!"
