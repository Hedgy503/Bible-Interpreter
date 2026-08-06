# ONNX Bible Text Summarizer

A Transformer-based Bible summarization system optimized for efficient inference using ONNX Runtime. This project converts a fine-tuned language model into ONNX format, allowing it to run efficiently on edge devices such as NVIDIA Jetson boards.

The program takes a Bible passage as input and generates a concise summary using a causal language model. The model uses the same prompt structure from training to produce consistent summaries.

![ONNX Bible summarizer running on an NVIDIA Jetson device](https://example.com/bible-summarizer-image.png)

---

# The Algorithm

## Overview

This project uses a fine-tuned Transformer causal language model that has been exported to the ONNX format. Instead of running the original PyTorch model, the ONNX version allows faster and more lightweight inference through ONNX Runtime.

The summarization pipeline consists of five main stages:

1. **Text Input**
2. **Tokenization**
3. **ONNX Model Inference**
4. **Text Generation**
5. **Summary Decoding**

---

## 1. Loading the Tokenizer

The tokenizer converts human-readable Bible text into numerical token IDs that the Transformer model can process.

The tokenizer is loaded from the exported model directory:

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False
)
