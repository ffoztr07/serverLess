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

def generate_text(prompt, max_new_tokens=200, temperature=0.7, top_p=0.9):
    """
    Generates text using the loaded language model.

    Args:
        prompt (str): The input prompt for text generation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): Controls the randomness of the output.
        top_p (float): Uses nucleus sampling to focus generation.

    Returns:
        str: The generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
