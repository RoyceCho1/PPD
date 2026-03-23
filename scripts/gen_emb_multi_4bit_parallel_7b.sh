#!/bin/bash

# ============================================================================
# LLaVA 7B Multi-GPU FULL Dataset Embedding Script (4-way Parallel / 4-bit)
# ============================================================================

# 전체 데이터를 몇 개의 청크로 나눌지
num_shards=100
per_user=20

# Option A: 4-bit 양자화된 7B 모델을 GPU당 1개씩 총 4개의 Worker로 동시 실행합니다.
gpus=("0" "1" "2" "3")

dry_run=false
output_dir="data/emb_test_7b_full"
mkdir -p $output_dir

echo "Running all experiments 4-way parallel on GPUs: ${gpus[*]}"

# 진행 상황을 보기 쉽게 하기 위해
echo "--------------------------------------------------------"
echo "실시간으로 로그를 보시려면 다음 명령어를 병렬 창에서 사용하세요:"
echo "tail -f $output_dir/log_chunk_*.txt"
echo "--------------------------------------------------------"

for (( which_shard=0; which_shard<num_shards; which_shard++ )); do
    # 현재 청크 처리할 GPU 번호 계산 (0, 1, 2, 3 순서로 할당)
    gpu_idx=$((which_shard % 4))
    which_gpu=${gpus[$gpu_idx]}
    
    export CUDA_VISIBLE_DEVICES=$which_gpu
    
    echo "Running chunk $which_shard on GPU $which_gpu (Background)"
    
    # --load_4bit 추가하고, 개별 로그를 생성하도록 > $output_dir/log_... 로 출력 설정
    command="python llava_embeddings/pick_a_pick_user_emb_7b.py \
        --load_4bit \
        --pretrained lmms-lab/llava-onevision-qwen2-7b-ov-chat \
        --num_shots 4 \
        --num_chunks $num_shards \
        --which_chunk $which_shard \
        --per_user $per_user \
        --save_every 1 \
        --output_dir $output_dir > $output_dir/log_chunk_${which_shard}.txt 2>&1"
    
    if [ "$dry_run" = false ]; then
        eval "$command &"
    else
        echo "$command &"
    fi
    
    # 4개의 백그라운드 프로세스가 런칭되었으면 먼저 수행이 끝날 때까지 대기
    # (chunk 0,1,2,3 동시 실행 후 대기 -> 끝나면 4,5,6,7 실행)
    if [ $(( (which_shard + 1) % 4 )) -eq 0 ]; then
        echo "Launched 4 parallel workers. Waiting for them to finish before starting the next batch..."
        wait
    fi
done

# 혹시 남아있는 백그라운드 프로세스가 있다면 대기
wait
echo "All chunks processed successfully! :)"
