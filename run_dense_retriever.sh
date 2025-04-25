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
  mkdir -p ${run_dir}
  mkdir -p index_data/${task_name}

  epr_model=output/epr/${task_name}/${model_name}/bert-fix_ctx-shared-bs64 

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
      num_ice=5 \
      num_candidates=50 \
      model_config.scale_factor=${scale_factor} \
      mode=cand_random
done