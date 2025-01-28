from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()

# Download and configure LoRA model
sql_lora_path = snapshot_download(repo_id="saudsaleem/fine-tuned-mistral")
print('SQL LoRA path:', sql_lora_path)

# Initialize the base model with LoRA enabled
model = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", enable_lora=True, device=device)

sampling_params = SamplingParams(
    temperature=0,  # Add some creativity (0-1)
    top_p=0.95,        # Controls diversity (0-1)
    max_tokens=256,   # Maximum tokens to generate
    repetition_penalty=1.2
)
class InputText(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(input: InputText):
    prompts = [input.text]
    mistralPrompt = f"""[INST]
<<SYS>>
You are a code-aware assistant. Follow these steps:
1. FIRST check if the answer exists in the provided code snippet data
2. If found, respond with code syntax from data
3. use general knowledge
Now handle this query:
User: {input.text} 
[/INST]
""",
    lora_request = LoRARequest("sql_adapter", 1, sql_lora_path)
    outputs = model.generate(mistralPrompt, sampling_params, lora_request=lora_request)
    response = ""
    for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      response = response + generated_text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    return {"generated_text": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)