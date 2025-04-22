from unsloth import FastLanguageModel
import torch

def inference(prompt, model, tokenizer, max_new_tokens=2048, temperature=0.7):
    """
    Run inference with the fine-tuned model
    
    Args:
        prompt: Input text prompt
        model: The fine-tuned model
        tokenizer: The tokenizer
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (higher = more random)
    
    Returns:
        Generated text
    """
    # Format the prompt properly
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Increased from 512 to 1024
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.05,  # Slightly reduced to allow more natural continuations
            pad_token_id=tokenizer.eos_token_id,  # Ensure proper padding
            eos_token_id=tokenizer.eos_token_id,  # Make sure it knows when to stop
            early_stopping=False  # Don't stop early
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (after "### Response:")
    response = generated_text.split("### Response:")[1].strip()
    
    return response

# Load the saved model
def load_model(model_path):
    """
    Load the saved fine-tuned model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model, tokenizer
    """
    # Load the model with same parameters used during training
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True
    )
    
    return model, tokenizer

# Example usage
if __name__ == "__main__":
    # Update this path to your saved model location
    model_path = os.path.join(base_path, "models", "poe_llama_final")
    
    # Load the model
    model, tokenizer = load_model(model_path)
    
    # Example prompt
    prompt = "Write a short story in the style of Edgar Allan Poe with the following characteristics:\n- Genre: Horror\n- Characters: A lonely scholar\n- Setting: An abandoned mansion\n- Atmosphere: Gloomy\n- Mood: Dread\n- Themes: Isolation and madness"
    
    # Run inference
    response = inference(prompt, model, tokenizer)
    
    print("Generated story:")
    print(response)