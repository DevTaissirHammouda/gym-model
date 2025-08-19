import pandas as pd

def generate_qa_from_row(row, food_col):
    qa_pairs = []

    # Build context: combine all nutrient columns
    context = ". ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])

    # Generate QA for all columns except the food column
    for col in row.index:
        if col == food_col or pd.isna(row[col]):
            continue
        food_name = row[food_col]
        question = f"What is the {col.lower()} of {food_name}?"
        answer = row[col]
        qa_pairs.append({
            "context": context,
            "question": question,
            "answer": answer
        })

    return qa_pairs

def nutrition_composition_csv_to_qa(input_csv, output_csv):
    # Read tab-separated CSV
    df = pd.read_csv(input_csv, sep='\t')
    # Strip column names
    df.columns = df.columns.str.strip()
    food_col = 'food'  # first column

    all_qa = []

    for _, row in df.iterrows():
        all_qa.extend(generate_qa_from_row(row, food_col))

    qa_df = pd.DataFrame(all_qa)
    qa_df.to_csv(output_csv, index=False)
    print(f"Generated {len(all_qa)} QA pairs in {output_csv}")

if __name__ == "__main__":
    nutrition_composition_csv_to_qa("data/nutrition.csv", "data/nutrition_composition_QA.csv")
