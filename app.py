from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

app = FastAPI()

# Download and configure LoRA model
bit_agent_lora_path = snapshot_download(repo_id="saudsaleem/fine-tuned-mistral")
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
    stop=["[/assistant]"]
)

# Initialize the base model with LoRA enabled
model = LLM(model="TheBloke/Mistral-7B-Instruct-v0.2-AWQ", enable_lora=True)

class InputText(BaseModel):
    text: str

@app.post("/generate/")
async def generate_text(input: InputText):
    prompts = [input.text]
    lora_request = LoRARequest("sql_adapter", 1, bit_agent_lora_path)
    outputs = model.generate(prompts, sampling_params, lora_request=lora_request)
    response = ""
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        response = response + generated_text
        print(f"Generated text: {generated_text!r}")
    return {"generated_text": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

