import pandas as pd

def generate_qa_from_row(row):
    qa_pairs = []
    # Combine all fields as context
    context = ". ".join([f"{col}: {row[col]}" for col in row.index if pd.notna(row[col])])

    # Column-to-question mapping
    column_questions = {
        "Age": "What is the age group of the respondent?",
        "Gender": "What is the gender of the respondent?",
        "How many meals do you have a day? (number of regular occasions in a day when a significant and reasonably filling amount of food is eaten)": 
            "How many meals does the respondent have in a day?",
        "What would best describe your diet:":
            "How would you best describe the respondent’s diet?",
        "Choose all that apply: [I skip meals]": "Does the respondent skip meals?",
        "Choose all that apply: [I experience feelings of hunger during the day]": "Does the respondent experience hunger during the day?",
        "Choose all that apply: [I consult a nutritionist/dietician]": "Does the respondent consult a nutritionist/dietician?",
        "Choose all that apply: [I cook my own meals]": "Does the respondent cook their own meals?",
        "What would you consider to be the main meal of YOUR day?": "What is considered the main meal of the day?",
        "What does your diet mostly consist of and how is it prepared?": "What does the respondent’s diet mostly consist of and how is it prepared?",
        "How many times a week do you order-in or go out to eat?": "How many times a week does the respondent order-in or go out to eat?",
        "Are you allergic to any of the following? (Tick all that apply)": "Is the respondent allergic to any foods?",
        "What is your weekly food intake frequency of the following food categories: [Sweet foods]": "How often does the respondent consume Sweet foods weekly?",
        "What is your weekly food intake frequency of the following food categories: [Salty foods]": "How often does the respondent consume Salty foods weekly?",
        "What is your weekly food intake frequency of the following food categories: [Fresh fruit]": "How often does the respondent consume Fresh fruit weekly?",
        "What is your weekly food intake frequency of the following food categories: [Fresh vegetables]": "How often does the respondent consume Fresh vegetables weekly?",
        "What is your weekly food intake frequency of the following food categories: [Oily, fried foods]": "How often does the respondent consume Oily/fried foods weekly?",
        "What is your weekly food intake frequency of the following food categories: [Meat]": "How often does the respondent consume Meat weekly?",
        "What is your weekly food intake frequency of the following food categories: [Seafood ]": "How often does the respondent consume Seafood weekly?",
        "How frequently do you consume these beverages [Tea]": "How frequently does the respondent consume Tea?",
        "How frequently do you consume these beverages [Coffee]": "How frequently does the respondent consume Coffee?",
        "How frequently do you consume these beverages [Aerated (Soft) Drinks]": "How frequently does the respondent consume Soft drinks?",
        "How frequently do you consume these beverages [Fruit Juices (Fresh/Packaged)]": "How frequently does the respondent consume Fruit juices?",
        "How frequently do you consume these beverages [Dairy Beverages (Milk, Milkshakes, Smoothies, Buttermilk, etc)]": "How frequently does the respondent consume Dairy beverages?",
        "How frequently do you consume these beverages [Alcoholic Beverages]": "How frequently does the respondent consume Alcoholic beverages?",
        "What is your water consumption like (in a day, 1 cup=250ml approx)": "How much water does the respondent drink per day?"
    }

    for col, question in column_questions.items():
        if col in row and pd.notna(row[col]):
            qa_pairs.append({"context": context, "question": question, "answer": row[col]})

    return qa_pairs

def nutrition_csv_to_qa(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    all_qa = []
    for _, row in df.iterrows():
        all_qa.extend(generate_qa_from_row(row))

    qa_df = pd.DataFrame(all_qa)
    qa_df.to_csv(output_csv, index=False)
    print(f"Generated {len(all_qa)} QA pairs in {output_csv}")

if __name__ == "__main__":
    nutrition_csv_to_qa("data/Dietary Habits Survey Data.csv", "data/megaNutritionQA.csv")
