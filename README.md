# Constructing Codebooks and Using A$^2$ATS Models: A Simple Guide

Taking Llama-3.1-8B-Instruct as an example.

## Prerequisites

- Download the Llama-3.1-8B-Instruct model and place it in `../huggingface-models/Llama-3.1-8B-Instruct`.
- Download the FineWeb-Edu dataset and place it in `../huggingface-datasets/finweweb-edu`.

## Install

```bash
pip install -r requirements.txt
pip install -e ./kmeans-gpu
```

The `kmeans-gpu` package is a GPU-accelerated implementation of the k-means++ clustering algorithm, specifically optimized for efficient codebook construction of A$^2$ATS.

## Constructing Codebooks

Run the jupyter notebook: `./notebooks/construct_codebooks.ipynb`. Users can adjust the codebook size by modifying the `codebook_size` variable. The constructed codebooks and Cholesky factors will be placed in the `./codebooks` directory.

## Running A$^2$ATS Models

Import and initialize tokenizer and our custom model:

```python
from models.llama_masked import LLM
from transformers import AutoTokenizer

model_dir = "../huggingface-models/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = LLM(model_dir, "./codebooks", "llama-3-8b-codebooks.pt", "llama-3-8b-cholesky_factors.pt", topk=0.03)
```

Utilize our custom model to generate text:

```python
messages = [
    {
        "role": "system", 
        "content": "You are a bot that responds to weather queries."
    },
    {
        "role": "user", 
        "content": "Hey, what's the temperature in Paris right now?"
    }
]

input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

output_ids = model.generate(inputs)

print(tokenizer.decode(output_ids))
```

Our custom sparse attention kernel `selective_attention` used by `models.llama` has shown good performance on the Intel Xeon Platinum 8469C CPU, but performance on other CPUs may still need optimization and could encounter compatibility issues. We are actively optimizing our custom kernels to achieve the best efficiency on a broad range of CPUs. We will release these kernels once they become available. In the meantime, please use `models.llama_masked` as a mathematically equivalent fallback for accuracy evaluation.