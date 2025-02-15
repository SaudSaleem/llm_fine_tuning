# import os
# from huggingface_hub import Repository, HfApi, login

# # Define your model path
# model_path = "fine-tuned-mistral-bitagent-latest"  # Your model directory
# model_name = model_path.split("/")[-1]  # Extract the model name from the model path

# # Log in to Hugging Face (if you haven't already)
# hf_token = os.getenv("HF_TOKEN")
# print('hf_token', hf_token)
# login(token=hf_token)  # Replace with your Hugging Face access token
# api = HfApi()
# repo_url = api.create_repo(repo_id=model_name, private=False)
# # Initialize the repository
# repo = Repository(local_dir=model_path, clone_from=repo_url)
# # Add all files and commit them
# repo.git_add()
# repo.git_commit("Upload fine-tuned model")
# # Push the changes to Hugging Face
# repo.push_to_hub()

# print(f"Model '{model_name}' has been successfully uploaded to Hugging Face!")

from transformers import AutoConfig

model_name = "saudsaleem/mistral-bitagent-latest"  # Your model name or path
config = AutoConfig.from_pretrained(model_name)

# Save the configuration to the same folder as the model
config.save_pretrained("./fine-tuned-mistral-bitagent-latest")  # Replace with your model directory