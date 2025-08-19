import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

def preprocess_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['context'] = df['context'].fillna('')
    df['question'] = df['question'].fillna('')
    df['answer'] = df['answer'].fillna('')
    
    df['context_clean'] = df['context'].apply(clean_text)
    df['question_clean'] = df['question'].apply(clean_text)
    df['answer_clean'] = df['answer'].apply(clean_text)
    
    return df
