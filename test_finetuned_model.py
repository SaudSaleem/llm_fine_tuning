import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_finetuned_model(model_path, prompt, max_length=100, num_beams=5):
    """
    Test a fine-tuned model by generating text for a given prompt.

    Parameters:
        model_path (str): Path to the fine-tuned model checkpoint.
        prompt (str): Input text to generate a response.
        max_length (int): Maximum length of the generated response.
        num_beams (int): Number of beams for beam search.

    Returns:
        str: The generated response from the model.
    """
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        # # Tokenize the input prompt
        # input_ids = tokenizer.encode(prompt, return_tensors="pt")
        # Tokenize the input with attention mask
        # Tokenize the input and move to device
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Generate response
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the generated response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"Error: {str(e)}"

# Define the path to your fine-tuned model checkpoint
model_path = "./fine-tuned-mistral-bitagent-latest"

# Ask the user for a prompt
user_prompt = input("Please enter a prompt: ")

# Test the model with the user's prompt and print the result
response = test_finetuned_model(model_path, user_prompt)
print("\nGenerated Response:")
print(response)
