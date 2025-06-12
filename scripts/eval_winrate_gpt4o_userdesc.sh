num_shards=1
which_shards=(0)
include_cots=(false true)
gpus=(0 1 2 3 4 5 6 7)

exp_num=0
dry_run=false

which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments"
fi

for which_shard in ${which_shards[@]}; do
for include_cot in ${include_cots[@]}; do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi
    which_gpu=${gpus[$exp_num % ${#gpus[@]}]}
    export CUDA_VISIBLE_DEVICES=$which_gpu

    echo "Running experiment $exp_num on GPU $which_gpu"
    command="python eval/eval_winrate_gpt4o_userdesc.py \
        --num_chunks $num_shards \
        --which_chunk $which_shard \
    "

    if [ $include_cot = true ]; then
        command="$command --include_cot"
    fi
    
    echo $command
    if [ $dry_run = false ]; then
        eval $command
    fi
    exp_num=$((exp_num+1))
done
done