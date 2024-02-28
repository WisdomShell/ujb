#!/bin/bash

function api_gen() {
    export OPENAI_API_BASE=''
    export OPENAI_API_KEY=''

    project_dir=$(cd "$(dirname $0)"/..; pwd)

    gen_mode=$1
    dataset=$2
    suffix=$3
    model_name_or_path=$4
    run_id=$5

    num_samples=20

    if [ $dataset == "codeujbdefectdetection" ]; then
        num_samples=1
    fi

    mkdir -p $project_dir/log/$run_id/$dataset
    python code_ujb/generate_api.py \
        --model-path $model_name_or_path \
        --model-id $run_id \
        --gen-mode $gen_mode \
        --bench-name $dataset \
        --max-new-tokens 1800 \
        --temperature 0.2 \
        --num-samples $num_samples  \
        --parallel 8 \
        --save-generations-path $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.json \
        | tee $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.log 2>&1
        
}

function tgi_gen() {
    project_dir=$(cd "$(dirname $0)"/..; pwd)
    mechine_id=$(ifconfig -a | grep ether | awk '{print $2}' | head -n 1)
    docker kill $(cat $project_dir/scripts/configs/tgi_container_ids_${mechine_id}.txt)

    gen_mode=$1
    dataset=$2
    suffix=$3
    model_name_or_path=$4
    run_id=$5

    num_samples=20
    min_share_size=1

    if [ $dataset == "codeujbdefectdetection" ]; then
        num_samples=5
    fi

    if [[ $run_id == *"-34b" ]]; then
        min_share_size=2
    fi
    
    python scripts/docker_deploy.py \
        --model_name_or_path $model_name_or_path \
        --run_id $run_id \
        --api_urls $project_dir/scripts/configs/tgi_api_urls_${mechine_id}.txt \
        --container_ids $project_dir/scripts/configs/tgi_container_ids_${mechine_id}.txt \
        --min_share_size $min_share_size \
        --max_gpu_nums 6

    export TGI_API_URL_${run_id//-/_}=$(cat $project_dir/scripts/configs/tgi_api_urls_${mechine_id}.txt)

    mkdir -p $project_dir/log/$run_id/$dataset
    python code_ujb/generate_api.py \
        --model-path $run_id \
        --model-id $run_id \
        --gen-mode $gen_mode \
        --bench-name $dataset \
        --max-new-tokens 1024 \
        --temperature 0.2 \
        --num-samples $num_samples  \
        --parallel 32 \
        --save-generations-path $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.json \
        | tee $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.log 2>&1
    
    docker kill $(cat $project_dir/scripts/configs/tgi_container_ids_${mechine_id}.txt)
}

function local_gen() {
    project_dir=$(cd "$(dirname $0)"/..; pwd)

    gen_mode=$1
    dataset=$2
    suffix=$3
    model_name_or_path=$4
    run_id=$5

    num_samples=20
    batch_size=20

    if [ $dataset == "codeujbdefectdetection" ]; then
        num_samples=10
        batch_size=10
    fi

    max_input_tokens=4096
    max_new_tokens=256
    if [ $run_id == "phi-1_5b" -o $run_id == "gpt2" ]; then
        max_input_tokens=512
        max_new_tokens=512
    fi

    mkdir -p $project_dir/log/$run_id/$dataset
    python code_ujb/generate_hf.py \
        --model-path $model_name_or_path \
        --model-id $run_id \
        --gen-mode $gen_mode \
        --bench-name $dataset \
        --max-input-tokens $max_input_tokens \
        --max-new-tokens $max_new_tokens \
        --temperature 0.2 \
        --dtype bfloat16 \
        --num-samples $num_samples  \
        --batch-size $batch_size \
        --num-gpus-total 4 \
        --save-generations-path $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.json \
        | tee $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.log 2>&1
        
}

function eval() {
    project_dir=$(cd "$(dirname $0)"/..; pwd)

    gen_mode=$1
    dataset=$2
    suffix=$3
    model_name_or_path=$4
    run_id=$5

    python3 code_ujb/evaluate.py \
        --model-path $model_name_or_path \
        --model-id $run_id \
        --gen-mode $gen_mode \
        --bench-name $dataset \
        --num-samples 20 \
        --load-generations-path $project_dir/log/$run_id/$dataset/generations-${suffix}-${gen_mode}.json \
        --eval-output-path $project_dir/log/$run_id/$dataset/evaluation-${suffix}-${gen_mode}.json
        
}

task=$1
gen_mode=$2
dataset=$3
model_name_or_path=$4
run_id=$5

if [ $task == "local_gen" ]; then
    local_gen $gen_mode $dataset default $model_name_or_path $run_id
elif [ $task == "api_gen" ]; then
    api_gen $gen_mode $dataset default $model_name_or_path $run_id
elif [ $task == "tgi_gen" ]; then
    tgi_gen $gen_mode $dataset default $model_name_or_path $run_id
elif [ $task == "eval" ]; then
    eval $gen_mode $dataset default $model_name_or_path $run_id
elif [ $task == "help" ]; then
    echo "./scripts/run_code_ujb.sh [local_gen|api_gen|tgi_gen|eval] [complete|chat] [codeujbrepair|codeujbcomplete|codeujbtestgen|codeujbtestgenissue|codeujbdefectdetection] model_name_or_path save_id"
else
    echo "task should be local_gen or api_gen"
fi