from src.preprocessing import preprocess_dataset
from src.retriever import Retriever

df = preprocess_dataset("data/fitness.csv")
contexts = df['context_clean'].tolist()
retriever = Retriever()
retriever.index_corpus(contexts)

while True:
    q = input("Question: ")
    if q.lower() in ["exit", "quit"]:
        break
    results = retriever.retrieve(q, top_k=2)
    print("Answer:", results[0][0] if results else "I don't know yet.")
