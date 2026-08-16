import time
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load your transformer model. 
# We'll use a lightweight Qwen model for this example, but you can swap it out.
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B", device_map="auto")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "default-model"
    messages: list[Message]
    max_tokens: int = 50

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # Extract the actual prompt from the OpenAI-style message payload
    prompt = request.messages[-1].content
    
    # Run the transformer inference
    outputs = pipe(prompt, max_new_tokens=request.max_tokens)
    
    # Strip the original prompt from the output if the pipeline includes it
    generated_text = outputs[0]["generated_text"]
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    # Return the standardized JSON structure AIPerf expects
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text
            },
            "finish_reason": "stop"
        }]
    }


"""
Reference commands for this baseline (shell, not Python):

#
uvicorn server:app --host 0.0.0.0 --port 8000

#
aiperf profile \
  --model llama3-8b \
  --url $ENDPOINT_URL \
  --endpoint-type chat \
  --streaming \
  --concurrency 1 \
  --request-count 1000 \
  --isl 1000 \
  --osl 500 \
  --tokenizer "meta-llama/Meta-Llama-3-8B-Instruct"


VLLM_DISABLE_COMPILE_CACHE=1

# Terminal 1: The Server
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --enforce-eager \
    -cc.mode=0 \
    --model-impl transformers \
    --port 8000

# Terminal 2: The Benchmark
aiperf profile --model "llama3-pure-eager" --url "http://localhost:8000/v1" --concurrency 1




## metrics vs. concurrency

# Run the same benchmark at multiple concurrency levels
for c in 1 10 50 100 200 500; do
  aiperf profile --model qwen3-0.6b --url "$ENDPOINT_URL" \
    --endpoint-type chat --streaming --concurrency $c \
    --request-count 1000 --isl 1000 --osl 500 \
    --tokenizer Qwen/Qwen3-0.6B --artifact-dir "artifacts/pareto-c$c"
done
"""
