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
        tokenizer = AutoTokenizer.from_pretrained('TheBloke/Mistral-7B-Instruct-v0.2-AWQ')
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Tokenize the input prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate a response
        output = model.generate(
            input_ids, 
            max_length=max_length, 
            num_beams=num_beams, 
            no_repeat_ngram_size=2, 
            early_stopping=True
        )

        # Decode the generated response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"Error: {str(e)}"

# Define the path to your fine-tuned model checkpoint and the example prompt
model_path = "./mistral-bitagent-finetune/checkpoint-500"
example_prompt = "Explain the concept of AI in simple terms:"

# Test the model and print the result
response = test_finetuned_model(model_path, example_prompt)
print("Generated Response:")
print(response)
