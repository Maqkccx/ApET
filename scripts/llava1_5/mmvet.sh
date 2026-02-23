#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path data/model/llava-v1.5-7b \
    --question-file data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder data/eval/mm-vet/images \
    --answers-file data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[96]' \
    --visual_token_num 288 \
    --basis_token_num 10 \
    --conv-mode vicuna_v1

mkdir -p data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src data/eval/mm-vet/answers/llava-v1.5-7b.jsonl \
    --dst data/eval/mm-vet/results/llava-v1.5-7b.json