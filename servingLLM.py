from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Your Hugging Face repo
model_id = "InterFaze/Qwen14BMergedAndFineTuned"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load model (4-bit quantization for Colab GPU efficiency)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

def generate_text(prompt, max_new_tokens=512, temperature=0.7, top_p=0.9, disable_thinking=True):
    """
    Generates text using the loaded language model.

    Args:
        prompt (str): The input prompt for text generation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): Controls the randomness of the output.
        top_p (float): Uses nucleus sampling to focus generation.
        disable_thinking (bool): If True, disables thinking patterns and reduces verbose output.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": not disable_thinking,  # Disable sampling to reduce thinking
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "early_stopping": True,  # Stop at EOS tokens
    }
    
    if not disable_thinking:
        # Only add sampling parameters when thinking is enabled
        generation_kwargs.update({
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        # Add parameters to reduce verbose thinking patterns
        generation_kwargs.update({
            "repetition_penalty": 1.1,  # Reduce repetitive thinking
            "no_repeat_ngram_size": 3,  # Prevent repetitive n-grams
        })

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
