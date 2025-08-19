def chunk_text(text, max_len=400, stride=50):
    tokens = text.split()
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_len, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += max_len - stride
    return chunks
