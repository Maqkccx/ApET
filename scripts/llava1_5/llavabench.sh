#!/bin/bash

python -m llava.eval.model_vqa \
    --model-path data/model/llava-v1.5-7b \
    --question-file data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder data/eval/llava-bench-in-the-wild/images \
    --answers-file data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --layer_list '[16]' \
    --image_token_list '[96]' \
    --visual_token_num 288 \
    --basis_token_num 10 \
    --conv-mode vicuna_v1

mkdir -p data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        data/eval/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
    --output \
        data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b.jsonl

python llava/eval/summarize_gpt_review.py -f data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-7b.jsonl
