import json
import pandas as pd

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def fill_missing_context(data, structured_fields=None):
    """
    Fill missing context from structured fields or question itself.
    structured_fields: list of keys in the JSON object to use as context.
    """
    for entry in data:
        if not entry.get("context"):
            if structured_fields:
                context_parts = [f"{k}: {entry.get(k, '')}" for k in structured_fields]
                entry["context"] = " ".join(context_parts)
            else:
                entry["context"] = entry["question"]
    return data

def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    data = load_jsonl("../data/samples.jsonl")
    data = fill_missing_context(data, structured_fields=["Title", "Desc", "BodyPart", "Equipment"])
    save_jsonl(data, "../data/preprocessed_samples.jsonl")
