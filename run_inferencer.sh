#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=  # change to your wandb account
export WANDB_API_KEY=  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1 

gpu=2
method=dpp-epr-random 
num_ice=50
port=9927

#model_name=gpt2-large 
#n_tokens=700
#scr_batch_size=128
#inf_batch_size=48

# model_name=EleutherAI/gpt-neo-1.3B
model_name=EleutherAI/gpt-neo-125m
n_tokens=1600
scr_batch_size=1
inf_batch_size=1

task_name=ultrafeedback
for scale_factor in 0.1
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${method}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  epr_model=output/epr/${task_name}/${model_name}/bert-fix_ctx-shared-bs64
  retrieve_file=${run_dir}/retrieved.json
  
  pred_file=${run_dir}/pred.json
  CUDA_LAUNCH_BLOCKING=1 accelerate launch --num_processes ${gpu} --main_process_port ${port}  inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      model_name=${model_name} \
      batch_size=${inf_batch_size} 
done