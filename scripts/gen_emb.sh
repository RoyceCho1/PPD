#!/bin/bash
set -euo pipefail

# 기존 환경의 8개 GPU 분할 구성을 단일 GPU 환경에 맞게 최적화한 스크립트입니다.
# `pick_a_pick_user_emb_7b.py` 를 실행하여 User Embedding을 생성합니다.

# Conda 환경 초기화 (기본 설정)
source /data/roycecho/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ppd

# ==========================================
# 사용자 설정 변수 (단일 GPU)
# ==========================================
export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/data/roycecho/.cache/huggingface"

# 변수 기본 설정 (실행 시 환경 변수로 덮어쓰기 가능)
OUTPUT_DIR="${OUTPUT_DIR:-/data/roycecho/PPD/data/user_emb_7b_full}"
NUM_SHOTS="${NUM_SHOTS:-4}"
PER_USER="${PER_USER:-20}"        # -1은 제한 없이 모든 데이터를 사용
DEVICE_MAP="${DEVICE_MAP:-none}" # 1개 GPU 이므로 자동 맵핑 대신 명확하게 none 처리
NUM_CHUNKS="${NUM_CHUNKS:-1}"    # 전체 데이터를 한번에 처리하도록(1 chunk) 설정
WHICH_CHUNK="${1:-0}"            # 스크립트 실행 시 첫 번째 인자로 청크 번호 지정 가능 (기본값: 0)

# ==========================================
# 실행
# ==========================================
mkdir -p "${OUTPUT_DIR}"

echo "=========================================================="
echo "    [STARTING PPD USER EMBEDDING GENERATION (7B)]"
echo "=========================================================="
echo "- 모델: LLaVA-1.5 / Qwen2 7B"
echo "- 출력 디렉토리: ${OUTPUT_DIR}"
echo "- 청크 설정: 전체 ${NUM_CHUNKS}조각 중 ${WHICH_CHUNK}번째 처리"
echo "- 사용 GPU: RTX 5090 (cuda:0)"
echo "=========================================================="

python llava_embeddings/pick_a_pick_user_emb_7b.py \
    --device cuda:0 \
    --device_map "${DEVICE_MAP}" \
    --num_shots "${NUM_SHOTS}" \
    --num_chunks "${NUM_CHUNKS}" \
    --which_chunk "${WHICH_CHUNK}" \
    --per_user "${PER_USER}" \
    --output_dir "${OUTPUT_DIR}"

echo "User Embedding 생성이 완료되었습니다!"