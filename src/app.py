import os
import streamlit as st
from preprocessing import preprocess_dataset
from retriever import Retriever
from conversation import Conversation
from local_model import generate_answer

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATA_DIR, "fitness.csv")

# Initialize retriever
retriever = Retriever(
    emb_path=os.path.join(DATA_DIR, "embeddings.pt"),
    doc_path=os.path.join(DATA_DIR, "documents.pt")
)
conv = Conversation()

# Load or build embeddings
if os.path.exists(retriever.emb_path) and os.path.exists(retriever.doc_path):
    retriever.load()
else:
    df = preprocess_dataset(CSV_PATH)
    contexts = df['context_clean'].tolist()
    retriever.index_corpus(contexts)
    retriever.save()

# Streamlit UI
st.title("🏋️ Fitness & Nutrition QA (Offline Smarter)")

user_q = st.text_input("Ask a question:")
if user_q:
    retrieved = retriever.retrieve(user_q, top_k=2)
    answer = generate_answer([ctx for ctx, _ in retrieved], user_q)
    conv.add_turn(user_q, answer)
    st.write("**Answer:**", answer)
