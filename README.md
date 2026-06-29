# [CVPR 2026 Highlight] ApET: Approximation-Error Guided Token Compression for Efficient VLMs

## 👀 Overview
TL;DR ApET introduces an attention-free, approximation-error guided token compression framework for VLMs that maximally preserves visual information by pruning tokens with minimal linear reconstruction error, enabling seamless FlashAttention integration while retaining 95.2% performance at 88.9% compression ratios.

<p align="center">
  <img src="./fig/showcase.png" alt="drawing", width="100%"/>
</p>


## 🔧 Environment Setup
We build and test our codebase with Python 3.10.18, PyTorch 2.1.2, and CUDA 12.1. Adapt the PyTorch and CUDA versions to your local environment as needed.

You can use the following command to install the required packages:
```
conda create -n ApET python=3.10 -y
conda activate ApET

cd /path/to/ApET
pip install -r requirements.txt
```


## 🎯 Usage
### Script Templates
```shell
bash scripts/llava1_5/[Benchmark].sh 
```

**For LLaVA-1.5 and LLaVA-NeXT, the script should look like this:**
```bash
python -m llava.eval.model_vqa_loader \
        --model-path data/model/llava-v1.5-7b \
        --question-file data/eval/gqa/$SPLIT.jsonl \
        --image-folder data/eval/gqa/images \
        --answers-file data/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --layer_list '[16]' \
        --image_token_list '[32]' \
        --visual_token_num 96 \
        --basis_token_num 10 \
        --conv-mode vicuna_v1 &
```
Configure the following required arguments:
- `--model-path`: Absolute path to the model checkpoint.
- `--question-file`: Path to the evaluation question file.
- `--image-folder`: Directory containing the input images.
- `--answers-file`: Output path for the generated answers.

Additionally, adjust the following hyper-parameters based on your experimental configuration:
- `--layer_list`: Layers to apply compression.
- `--image_token_list`: Token selection strategy.
- `--visual_token_num`: Number of visual tokens to retain.
- `--basis_token_num`: Number of basis tokens for decomposition.


**For Qwen models, please refer to the [Qwen-2.5-VL codebase](https://github.com/QwenLM/Qwen3-VL).**

**For Video-LLaVA models, please refer to the [Video-LLaVA codebase](https://github.com/PKU-YuanGroup/Video-LLaVA).**


## 🙏Acknowledgements
Our codebase is adapted from [VScan](https://github.com/Tencent/SelfEvolvingAgent) and [PyramidDrop](https://github.com/Cooperx521/PyramidDrop). We thank the authors for releasing their code! We also thank the authors of [LLaVA](https://github.com/haotian-liu/LLaVA), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) and [Qwen-2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for their open-sourced models and well-written instructions.


## 📌 BibTeX & Citation
If you find our paper and code useful in your research, please consider giving a star and citation.

```BibTeX
@article{ma2026apet,
  title={ApET: Approximation-Error Guided Token Compression for Efficient VLMs},
  author={Ma, Qiankun and Zhang, Ziyao and Wang, Haofei and Chen, Jie and Song, Zhen and Zheng, Hairong},
  journal={arXiv preprint arXiv:2602.19870},
  year={2026}
}
