# Fine-Tuning Llama 2 7B with LoRA for Persona-Based Conversations

[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/rudrajadon18/Llama-2-7b-chat-finetune)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🚀 Overview

This repository provides the code and methodology for fine-tuning the `NousResearch/Llama-2-7b-chat-hf` model on a persona-based chat dataset. The primary goal is to adapt the model to generate responses consistent with a specific persona, making conversations feel more natural and personalized.

To achieve this efficiently, the model was fine-tuned using **Low-Rank Adaptation (LoRA)** and **4-bit quantization**, making it possible to train on consumer-grade hardware like Google Colab.

---

## ✨ Key Features

- **Model**: Leverages the powerful `Llama-2-7b-chat-hf` model.
- **Efficient Fine-Tuning**: Utilizes LoRA for parameter-efficient fine-tuning, drastically reducing computational and memory requirements.
- **Quantization**: Employs 4-bit quantization through `bitsandbytes` to make training accessible on platforms like Google Colab.
- **Dataset**: Fine-tuned on the `Cynaptics/persona-chat` dataset, which contains dialogues designed to reflect distinct personalities.
- **Reproducibility**: All configurations and preprocessing steps are documented for easy replication.

---

## 📋 Table of Contents

1.  Model and Method Selection
2.  Fine-Tuning Details
3.  How to Use the Model
4.  Model Weights

---

## 🧠 Model and Method Selection

### Language Model: `NousResearch/Llama-2-7b-chat-hf`

The Llama 2 7B chat model was chosen for its exceptional balance of performance and resource requirements. It is pre-trained on a massive corpus of text, giving it a strong foundation in natural language that can be effectively adapted to new tasks with minimal fine-tuning.

### Fine-Tuning Method: LoRA (Low-Rank Adaptation)

Instead of traditional fine-tuning which updates all model weights, **LoRA** was used. This method freezes the original model weights and injects small, trainable low-rank matrices into the Transformer layers.

**Advantages of using LoRA:**
- **Resource Efficiency**: Dramatically reduces the number of trainable parameters, allowing fine-tuning on limited VRAM.
- **Faster Training**: Less computation per training step leads to faster fine-tuning cycles.
- **No Catastrophic Forgetting**: The original model weights remain unchanged, preserving the pre-trained knowledge.

---

## 🛠️ Fine-Tuning Details

### Dataset and Preprocessing

- **Dataset**: `Cynaptics/persona-chat`, a conversational dataset where each dialogue is associated with a specific persona.
- **Preprocessing Steps**:
    1. The dataset was shuffled, and a subset of 1000 samples was selected for training.
    2. The data was formatted to explicitly provide the persona's context to the model. Special tokens like `<persona_b>` and `[INST]` were used to delineate the persona information from the user's query, guiding the model's learning process.

### LoRA Configuration

The following LoRA parameters were used to configure the low-rank adaptation:

| Parameter      | Value | Description                                                    |
| :------------- | :---- | :------------------------------------------------------------- |
| `lora_r`       | `64`  | The rank (dimension) of the update matrices.                   |
| `lora_alpha`   | `16`  | The scaling factor for the LoRA activations.                   |
| `lora_dropout` | `0.1` | Dropout probability for the LoRA layers to prevent overfitting.|

### Training Parameters

The model was trained on **Google Colab** under the following conditions:

- **Quantization**: 4-bit via `bitsandbytes`.
- **Optimizer**: `paged_adamw_32bit`.
- **Learning Rate**: `2e-4` with a cosine scheduler.
- **Batch Size**: `4`.
- **Gradient Accumulation**: Used to stabilize training with a small batch size.

---

## 🚀 How to Use the Model

You can easily load and use the fine-tuned model from the Hugging Face Hub using the `transformers`, `accelerate`, and `peft` libraries.

First, make sure you have the required libraries installed:
```bash
pip install transformers torch accelerate bitsandbytes peft
```

Next, use the following Python script to load the model and run inference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# --- Configuration ---
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
peft_model_id = "rudrajadon18/Llama-2-7b-chat-finetune"

# --- Quantization Configuration ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# --- Load Base Model ---
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# --- Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# --- Load LoRA Adapter ---
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.eval()

# --- Prepare Input ---
persona_prompt = "<persona_b>I love to read books. I am a vegetarian. I have a dog named fluffy. I like to play video games."
query = "What are your favorite hobbies?"
prompt = f"<s>[INST] {persona_prompt} {query} [/INST]"

# --- Generate Response ---
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

This script loads the base Llama 2 model in 4-bit, attaches the fine-tuned LoRA weights, and generates a response based on a sample persona and query.

## 🤗 Model Weights
The fine-tuned model adapters are publicly available on the Hugging Face Hub. You can access them at the following link:

- **Hugging Face Model**: [rudrajadon18/Llama-2-7b-chat-finetune](https://huggingface.co/rudrajadon18/Llama-2-7b-chat-finetune)

