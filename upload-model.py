import os
from huggingface_hub import Repository, HfApi, login

# Define your model path
model_path = "fine-tuned-mistral-bitagent-latest"  # Your model directory
model_name = model_path.split("/")[-1]  # Extract the model name from the model path

# Log in to Hugging Face (if you haven't already)
hf_token = os.getenv("HF_TOKEN")
print('hf_token', hf_token)
login(token=hf_token)  # Replace with your Hugging Face access token

# Initialize the API to create a new repository
api = HfApi()

# Create a new repo with the name derived from the model path
repo_url = api.create_repo(repo_id=model_name, private=False)  # Private repo, you can change to False if public

# Initialize the repository
repo = Repository(local_dir=model_path, clone_from=repo_url)

# Add all files and commit them
repo.git_add()
repo.git_commit("Upload fine-tuned model")

# Push the changes to Hugging Face
repo.push_to_hub()

print(f"Model '{model_name}' has been successfully uploaded to Hugging Face!")
