import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Import PEFT for LoRA adapters

# Define paths
BASE_MODEL_DIR = "TheBloke/Mistral-7B-Instruct-v0.2-AWQ"  # Your base model from Hugging Face
LORA_MODEL_DIR = "/home/user/saud/models/fine-tuned-mistral-bitagent-latest"  # Your fine-tuned LoRA model

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR)

# Load the LoRA adapters
model = PeftModel.from_pretrained(base_model, LORA_MODEL_DIR)
model.eval()  # Set model to evaluation mode

# Load the tokenizer (from the base model)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

def generate_response(prompt, max_length=200):
    """
    Generates a response using the fine-tuned LoRA model.

    :param prompt: Input text for the model
    :param max_length: Maximum length of the generated response
    :return: Model-generated response as text
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    test_prompt = "Send an email to a recipient"
    response = generate_response(test_prompt)
    print("\nGenerated Response:\n", response)
