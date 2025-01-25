import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_finetuned_model(model_path, prompt, max_length=100, num_beams=5):
    try:
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device', device)
        # Load tokenizer and model, move model to device
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        # Tokenize input and move tensors to the same device
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

        # Decode and return response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

    except Exception as e:
        return f"Error: {str(e)}"

# Define model path and prompt
model_path = "./fine-tuned-mistral-bitagent-latest"
prompt = input("Please enter a prompt: ")

# Test model
response = test_finetuned_model(model_path, prompt)
print("Generated Response:", response)
