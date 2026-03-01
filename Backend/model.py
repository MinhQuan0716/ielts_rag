# Backend/model.py
import os
from google import genai
from dotenv import load_dotenv
import pandas as pd

# 1. Load the hidden API key from your .env file
load_dotenv()

# 2. Initialize the modern GenAI client
# (It automatically finds the GEMINI_API_KEY we just loaded)
client = genai.Client()


def evaluate_baseline_essay(essay_text):
    """Sends a single essay to the LLM for a baseline evaluation."""

    system_prompt = """
    You are an expert, strict IELTS examiner. 
    Evaluate the following Task 2 essay based on the official IELTS criteria.
    Provide an Overall Band Score (0-9) and a brief explanation of the Task Response.

    Format your output exactly like this:
    Overall Score: [Score]
    Reasoning: [Your brief explanation]
    """

    full_prompt = f"{system_prompt}\n\nEssay to grade:\n{essay_text}"

    try:
        # Using the new SDK syntax and the latest fast model
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Error communicating with AI: {e}"


# --- Testing Block ---
if __name__ == "__main__":
    print("Loading cleaned dataset...")

    # Adjust path based on your folder structure
    dataset_path = '../Data/processed/cleaned_ielts_task2_essays.csv'

    try:
        df = pd.read_csv(dataset_path)

        # Grab the first essay and its actual human score
        test_essay = df.iloc[1]['essay']
        actual_score = df.iloc[1]['overall']

        print(f"\n--- Actual Human Examiner Score: {actual_score} ---")
        print("\nSending essay to AI for baseline evaluation...\n")

        # Get the AI's grade
        ai_evaluation = evaluate_baseline_essay(test_essay)
        print(ai_evaluation)

    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {dataset_path}")