---

# Fine-Tuning LLaMA 2 7B Chat Model with LoRA for Persona-Based Chat

## Overview

This repository contains code and documentation for fine-tuning the **LLaMA 2 7B Chat Model** using the **LoRA (Low-Rank Adaptation)** method on a persona-based dataset. The goal is to personalize the model's responses based on a defined persona while ensuring efficient training using LoRA.

---

## 1. **Choice of LLM:**

For this task, the **LLaMA 2 7B Chat Model** from **NousResearch** was selected due to its state-of-the-art performance in natural language understanding and generation. LLaMA models have shown strong capabilities in a variety of tasks, including conversation generation. 

- **Model Name:** `NousResearch/Llama-2-7b-chat-hf`
- **Reason for choosing LLaMA 2:**
  - LLaMA 2 models are efficient and provide great generalization capabilities on various downstream tasks.
  - The 7B version offers a good balance between performance and computational resources, making it ideal for fine-tuning on a specific dataset.
  - It is pre-trained on a vast amount of data, allowing for significant adaptation to the task with fewer epochs.

---

## 2. **Choice of Fine-Tuning Method:**

To fine-tune the model efficiently, the **LoRA (Low-Rank Adaptation)** method was used. LoRA allows for adapting large pre-trained models with minimal computational resources and memory overhead by adding low-rank matrices to the model layers, rather than modifying the entire model.

- **LoRA Parameters:**
  - **LoRA Attention Dimension (`lora_r`)**: 64
  - **Alpha Scaling Factor (`lora_alpha`)**: 16
  - **Dropout (`lora_dropout`)**: 0.1
  
- **Why LoRA?**
  - LoRA provides a way to fine-tune large models like LLaMA 2 with significantly fewer resources compared to traditional fine-tuning.
  - It allows for more efficient memory usage, making it possible to run the model in low-resource environments, such as on Google Colab.
  - The addition of low-rank matrices enables fast adaptation without sacrificing performance, especially useful for tasks like persona-based conversation generation.

---

## 3. **Justifications:**

- **Dataset Choice:** The dataset used for fine-tuning is a persona-based conversation dataset (`Cynaptics/persona-chat`), which contains hypothetical dialogues between two personas. This dataset was chosen to train the model to generate more human-like responses that align with specific personalities.
  
- **Preprocessing:** The dataset was preprocessed to ensure it was in the required format for fine-tuning. Specifically:
  - **Shuffling** and **subsetting** the dataset to select 1000 samples for efficient training.
  - **Transforming the data** to incorporate persona context (Persona B's facts) and segment the conversation turns. This helps in personalizing responses based on persona-specific information.
  - **Special tokens** like `<persona_b>` and `[INST]` were added to separate persona context from the dialogue and guide the model during fine-tuning.
  
- **Efficiency:** Given the large size of the LLaMA 2 7B model, LoRA provides an efficient fine-tuning strategy without the need for extensive hardware resources. This method is perfect for this task as it adapts the model to the persona-based conversations without altering the entire model architecture.
  
- **Training Process:** The training process is carried out with a batch size of 4, learning rate of 2e-4, and cosine learning rate scheduler, optimized using the `paged_adamw_32bit` optimizer. Gradient accumulation is used to ensure stability during training.

- **Compute Constraints:** The fine-tuning was performed on **Google Colab** using **4-bit quantization** (via `bitsandbytes`) to reduce memory footprint, which allowed for training on limited resources while maintaining model performance.

---

## 4. **Model Weights:**

Once the model was fine-tuned, the weights were uploaded to Hugging Face for easy access and reuse.

- **Link to Model Weights:** [Llama-2-7b-chat-finetune - Hugging Face](https://huggingface.co/rudrajadon18/Llama-2-7b-chat-finetune)

---
