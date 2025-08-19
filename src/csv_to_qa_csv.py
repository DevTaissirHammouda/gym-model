import pandas as pd

def generate_qa_from_row(row):
    qa_pairs = []

    # Build context from available fields
    context_fields = ["Title", "Desc", "Equipment", "BodyPart", "Level"]
    context = ". ".join([f"{field}: {row[field]}" for field in context_fields if pd.notna(row.get(field, None))])

    # Muscle / body part QA
    if pd.notna(row.get("BodyPart")):
        question = f"Which muscles does {row.get('Title', '')} target?"
        answer = row["BodyPart"]
        qa_pairs.append({"context": context, "question": question, "answer": answer})

    # Equipment QA
    if pd.notna(row.get("Equipment")):
        question = f"What equipment is needed for {row.get('Title', '')}?"
        answer = row["Equipment"]
        qa_pairs.append({"context": context, "question": question, "answer": answer})

    # Difficulty QA
    if pd.notna(row.get("Level")):
        question = f"What is the difficulty level of {row.get('Title', '')}?"
        answer = row["Level"]
        qa_pairs.append({"context": context, "question": question, "answer": answer})

    # Description QA
    if pd.notna(row.get("Desc")):
        question = f"Describe the {row.get('Title', '')} exercise."
        answer = row["Desc"]
        qa_pairs.append({"context": context, "question": question, "answer": answer})

    return qa_pairs

def csv_to_qa_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    all_qa = []

    for _, row in df.iterrows():
        all_qa.extend(generate_qa_from_row(row))

    qa_df = pd.DataFrame(all_qa)
    qa_df.to_csv(output_csv, index=False)
    print(f"Generated {len(all_qa)} QA pairs in {output_csv}")

if __name__ == "__main__":
    csv_to_qa_csv("data/megaGymDataset.csv", "data/megaGymQA.csv")
