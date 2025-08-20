from llama_cpp import Llama

llm = Llama(model_path="models/ggml-model-q4_0.bin")  # your GGML quantized model
  # can choose smaller CPU model if needed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.eval()

def generate_answer(contexts, question):
    prompt = "You are a fitness coach AI. Use the following info to answer:\n"
    prompt += "\n".join(contexts) + f"\nQuestion: {question}\nAnswer:"
    output = llm(prompt, max_tokens=256)
    return output['choices'][0]['text'].strip()
