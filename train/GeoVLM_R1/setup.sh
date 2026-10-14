#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
# Derived from om-ai-lab/VLM-R1 and huggingface/open-r1 (Apache-2.0); IBM modifications under CC-BY-NC-4.0.
#
# conda create -n vlm-r1 python=3.11 
# conda activate vlm-r1

# Install the packages in geovlm_r1 .
pip install -e ".[dev]"

# Addtional modules
pip install wandb==0.18.3
pip install tensorboardx
pip install qwen_vl_utils torchvision
pip install flash-attn --no-build-isolation
pip install babel
pip install python-Levenshtein
pip install matplotlib
pip install pycocotools
pip install openai
pip install httpx[socks]
pip install sentence-transformers
pip install git+https://github.com/openai/CLIP.git
pip install json-repair
pip install shapely
pip install rouge-score
pip install nltk
pip install colorama
pip install scikit-learn
pip install pycocoevalcap
