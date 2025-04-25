#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

gpu=2
method=dpp-epr-random
port=9927

# model settings
env_model_name=EleutherAI/gpt-neo-125m
n_tokens=3000
scr_batch_size=1
inf_batch_size=1

task_name=ultrafeedback
# Define parameter sets: len_qa,len_pd
param_sets=("10,5" "10,10")

for scale_factor in 0.1; do
  for pair in "${param_sets[@]}"; do
    # Parse QA and PD lengths and compute num_ice
    IFS=',' read len_qa len_pd <<< "${pair}"
    num_ice=$((len_qa + len_pd))
    echo "Running dense retriever with num_ice=${num_ice}"
    echo "Running inferencer with len_qa=${len_qa}, len_pd=${len_pd}"

    export WANDB_TAGS="${method},${task_name},${env_model_name}"
    run_dir=output_${len_qa}_${len_pd}/${method}/${task_name}/${env_model_name}
    index_data=index_data/${task_name}/index_dataset.json
    mkdir -p ${run_dir}
    mkdir -p index_data/${task_name}

    epr_model=output/epr/${task_name}/${env_model_name}/bert-fix_ctx-shared-bs64

    retrieve_file=${run_dir}/retrieved.json
    python dense_retriever.py \
        hydra.run.dir=${run_dir}/dense_retriever \
        output_file=${retrieve_file} \
        task_name=${task_name} \
        dataset_reader.dataset_split=test_prefs \
        +dataset_reader.ds_size=44000 \
        index_reader.dataset_path=${index_data} \
        model_config.norm_embed=true \
        faiss_index=${run_dir}/index \
        dpp_search=true \
        dpp_topk=100 \
        num_ice=${num_ice} \
        num_candidates=50 \
        model_config.scale_factor=${scale_factor} \
        mode=cand_random
    
    pred_file=${run_dir}/pred_${len_qa}_${len_pd}.json
    log_file=${run_dir}/inferencer_${len_qa}_${len_pd}.log
    echo "Starting inferencer run with params len_qa=${len_qa}, len_pd=${len_pd} at $(date '+%Y-%m-%d %H:%M:%S'), logging to ${log_file}"  
    CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes ${gpu} --main_process_port ${port} inferencer.py \
        hydra.run.dir=${run_dir}/inferencer \
        task_name=${task_name} \
        dataset_reader.dataset_path=${retrieve_file} \
        dataset_reader.n_tokens=${n_tokens} \
        index_reader.dataset_path=${index_data} \
        output_file=${pred_file} \
        model_name=${env_model_name} \
        batch_size=${inf_batch_size} \
        dataset_reader.len_qa=${len_qa} \
        dataset_reader.len_pd=${len_pd} 2>&1 | tee ${log_file}
  done
  
done
