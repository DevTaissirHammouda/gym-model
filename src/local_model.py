from llama_cpp import Llama
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "claude2-alpaca-7b.Q3_K_S.gguf")  

# Initialize LLaMA
from llama_cpp import Llama

llm = Llama(
    model_path="claude2-alpaca-7b.Q3_K_S.gguf",
    n_gpu_layers=32,  # offload layers to GPU
    use_mlock=True,
    gpu=True           # enable GPU
)


# You **cannot print llm.n_gpu_layers**, remove this line
# print("GPU layers used:", llm.n_gpu_layers)

def generate_answer(contexts, question):
    
    prompt = (
        "You are a fitness coach AI. Use the following info to answer:\n"
        + "\n".join(contexts)
        + f"\nQuestion: {question}\nAnswer:"
    )
    output = llm(
        prompt=prompt,
        max_tokens=256,
        echo=False,
        stop=["Question:"]
    )
    return output["choices"][0]["text"].strip()
