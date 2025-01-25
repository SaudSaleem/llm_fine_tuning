from transformers import AutoTokenizer, AutoModelForCausalLM

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
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Tokenize the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate a response
        output = model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                no_repeat_ngram_size=2,
                temperature=0.7,      # Controls randomness (lower = more focused output)
                top_k=50,             # Limits to top 50 tokens for each step
                top_p=0.9,            # Nucleus sampling (only considers top 90% probable tokens)
                repetition_penalty=1.2  # Penalizes repetitive phrases
        )


        # Decode the generated response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"Error: {str(e)}"

# Define the path to your fine-tuned model checkpoint
model_path = "./fine-tuned-mistral"

# Ask the user for a prompt
user_prompt = input("Please enter a prompt: ")

# Test the model with the user's prompt and print the result
response = test_finetuned_model(model_path, user_prompt)
print("\nGenerated Response:")
print(response)
