#!/bin/bash

PWD="$(pwd)" 
cd $PWD

hits=5
max_length=1000000

torchrun --nproc_per_node 8 -m tasks.eval \
            --pipeline graphrag \
            --eval_data_path data/HawkBench \
            --result_dir data/results \
            --ret_model BAAI/bge-m3 \
            --eval_level 1 2 3 4 \
            --gen_model Qwen2.5-7B-Instruct \
            --mem_model memorag-qwen2-7b-inst \
            --gen_max_new_tokens 256 \
            --ret_hits $hits \
            --max_length $max_length \
            --eval_datasets tech novel arts humanities paper science finance law \
            --index_path data/index/emb/bge-m3 \
            --api_dict data/api_dict.json \
            --agent_provider deepseek \
            --agent_model deepseek-chat \
            --use_api false \
            --conv false
            # --use_minference true
