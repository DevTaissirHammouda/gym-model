# 🏋️ AI Fitness Coach (LLaMA Local Inference)

This project is a local AI-powered **fitness coach assistant** that runs fully on your machine using [`llama.cpp`](https://github.com/ggerganov/llama.cpp) with Python bindings (`llama-cpp-python`).  
It takes **context (user fitness details, history, or notes)** and **answers fitness-related questions** without needing an internet connection.  

## ✨ Features
- Runs **locally** with no external API calls (private & offline).  
- Supports **GGUF quantized LLaMA / Alpaca / Claude-style models**.  
- Configurable for **CPU** or **GPU acceleration (CUDA/cuBLAS)**.  
- Simple Python interface for generating answers.  

---

## ⚡ Requirements

- Python **3.9+**  
- Git  
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) (for GPU support)  
- [CMake](https://cmake.org/download/) + Visual Studio (Windows) or GCC/Clang (Linux/Mac)  

---

## 🔧 Installation

### 1. Clone repository
```bash
git clone https://github.com/yourusername/ai-fitness-coach
cd ai-fitness-coach

2. (Optional) Build llama.cpp with GPU

If you want GPU acceleration on Windows:

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUBLAS=ON
cmake --build . --config Release


Then install Python bindings:

cd ../python
pip install .


If you only need CPU inference:

pip install llama-cpp-python

📂 Project Structure
ai-fitness-coach/
│── models/                        # Place GGUF models here
│   └── claude2-alpaca-7b.Q3_K_S.gguf
│── src/
│   ├── app.py                     # Example app entrypoint
│   └── local_model.py             # LLaMA wrapper logic
│── README.md

🚀 Usage
1. Download a model

Put a quantized .gguf model in the models/ folder.
(Example: claude2-alpaca-7b.Q3_K_S.gguf)

2. Run the app
python src/app.py

3. Example (in code)
from local_model import generate_answer

contexts = [
    "User is 25 years old, 75kg, wants to lose fat.",
    "Prefers home workouts and bodyweight training."
]

question = "Can you suggest a weekly workout plan?"

answer = generate_answer(contexts, question)
print(answer)

🖥️ GPU Acceleration

Use the n_gpu_layers parameter to offload layers to your GPU:

llm = Llama(
    model_path="models/claude2-alpaca-7b.Q3_K_S.gguf",
    n_gpu_layers=32,   # Adjust based on VRAM
    use_mlock=True
)


Monitor GPU usage with:

nvidia-smi

⚠️ Notes

Larger models (7B, 13B, 30B, etc.) require more VRAM.

Quantized models (Q3, Q4, Q5) are recommended for consumer GPUs like RTX 3070.

If GPU build fails, you can always run on CPU (slower but stable).