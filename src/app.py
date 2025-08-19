import streamlit as st
from preprocessing import preprocess_dataset
from retriever import Retriever
from conversation import Conversation

# Load dataset
df = preprocess_dataset("data/fitness.csv")
contexts = df['context_clean'].tolist()
retriever = Retriever()
retriever.index_corpus(contexts)

conv = Conversation()

st.title("Fitness & Nutrition QA")

user_q = st.text_input("Ask a question:")
if user_q:
    retrieved = retriever.retrieve(user_q, top_k=2)
    answer = retrieved[0][0] if retrieved else "I don't know yet."
    conv.add_turn(user_q, answer)
    st.write("Answer:", answer)
