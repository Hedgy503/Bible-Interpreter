import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM

MODEL_PATH = "./onnx_bible_model"   # or path to your model folder

# 1. Load tokenizer and set pad token
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 2. Load model
model = ORTModelForCausalLM.from_pretrained(MODEL_PATH)

# 3. Format input text cleanly
prompt = "Summarize the following text:\nIn the beginning God created the heaven and the earth.\n\nSummary:"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

# Debug line: Verify tokens are being generated properly
print(f"Input Token IDs: {inputs['input_ids']}")

# 4. Generate with explicit parameters
print("Generating summary...")
summary_ids = model.generate(
    **inputs,
    max_new_tokens=150,
    min_new_tokens=20,             # Prevents stopping immediately
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.2,        # Prevents looping
    do_sample=True,                # Enables sampling instead of greedy picking
    temperature=0.7,
    top_p=0.9
)

# 5. Decode output (only decode the NEW tokens generated)
input_length = inputs["input_ids"].shape[1]
output_text = tokenizer.decode(summary_ids[0][input_length:], skip_special_tokens=True)

print("================ GENERATED SUMMARY ================")
print(output_text.strip())
print("===================================================")
