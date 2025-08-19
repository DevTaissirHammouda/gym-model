from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

def chunk_text(text, max_len=100):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    for sent in sentences:
        if len(chunk.split()) + len(sent.split()) > max_len:
            if chunk:
                chunks.append(chunk)
            chunk = sent
        else:
            chunk += " " + sent
    if chunk:
        chunks.append(chunk)
    return chunks
