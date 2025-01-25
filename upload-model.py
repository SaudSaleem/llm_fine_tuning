import os
from huggingface_hub import Repository, HfApi, login

# Define your model path and the Hugging Face repo URL
model_path = "fine-tuned-mistral"  # Your model directory
repo_url = "https://huggingface.co/saudsaleem/distilbert-distilgpt2-bitagent"  # Your Hugging Face repo URL

hf_token = os.getenv("HF_TOKEN")
print('hf_token', hf_token)
# Log in to Hugging Face (if you haven't already)
login(token=hf_token)  # Replace with your Hugging Face access token

# Initialize the repository
repo = Repository(local_dir=model_path, clone_from=repo_url)

# Add all files and commit them
repo.git_add()
repo.git_commit("Upload fine-tuned model")

# Push the changes to Hugging Face
repo.push_to_hub()

print("Model has been successfully uploaded!")
