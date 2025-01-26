import os
import json

# Path to the directory containing checkpoints
CHECKPOINT_DIR = "./Mistral-7B-Instruct-v0.2-AWQ-bitagent-finetune"

# Function to get evaluation loss from trainer_state.json
def get_evaluation_loss(checkpoint_path):
    trainer_state_file = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state_file):
        return None

    with open(trainer_state_file, "r") as file:
        data = json.load(file)
        # Extract the latest evaluation loss from the log_history
        eval_loss = None
        for entry in data.get("log_history", []):
            if "eval_loss" in entry:
                eval_loss = entry["eval_loss"]
        return eval_loss

# Main function to find the best checkpoint
def find_best_checkpoint():
    checkpoints = [os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")]

    best_checkpoint = None
    best_loss = float("inf")

    for checkpoint in checkpoints:
        eval_loss = get_evaluation_loss(checkpoint)
        if eval_loss is not None:
            # print(f"Checkpoint: {checkpoint}, Eval Loss: {eval_loss}")
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_checkpoint = checkpoint

    if best_checkpoint:
        print(f"Best Checkpoint: {best_checkpoint}, Eval Loss: {best_loss}")
        return best_checkpoint
    else:
        print("No valid evaluation loss found in any checkpoint.")
        return None

if __name__ == "__main__":
    best_checkpoint_path = find_best_checkpoint()
    if best_checkpoint_path:
        print(f"The best checkpoint path is: {best_checkpoint_path}")
