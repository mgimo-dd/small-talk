#!/usr/bin/env python3
"""
Simple chat script for Qwen3-0.6B model
Based on examples from https://huggingface.co/Qwen/Qwen3-0.6B
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example prompt
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    print("Generating response...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {response}")

if __name__ == "__main__":
    main()
