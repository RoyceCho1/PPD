num_shards=1
which_shards=(0)
include_cots=(false)
randomize_fewshots=(true)
no_captions=(false)
gpus=(0 1 2 3 4 5 6 7)

datasets=()

exp_num=0
dry_run=false

which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments"
fi

for which_shard in ${which_shards[@]}; do
for include_cot in ${include_cots[@]}; do
for randomize_fewshot in ${randomize_fewshots[@]}; do
for no_caption in ${no_captions[@]}; do
for dataset in ${datasets[@]}; do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi
    which_gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo "Running experiment $exp_num on GPU $which_gpu"
    command="python eval/eval_winrate_gpt4o.py \
        --num_chunks $num_shards \
        --which_chunk $which_shard \
        --dataset_name $dataset \
    "

    if [ $include_cot = true ]; then
        command="$command --include_cot"
    fi

    if [ $randomize_fewshot = true ]; then
        command="$command --randomize_fewshot"
    fi

    if [ $no_caption = true ]; then
        command="$command --no_caption"
    fi
    
    echo $command
    if [ $dry_run = false ]; then
        eval $command
    fi
    exp_num=$((exp_num+1))
done
done
done
done
done