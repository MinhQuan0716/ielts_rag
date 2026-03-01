# Backend/rag_evaluator.py
import os
import pandas as pd
import chromadb
from google import genai
from dotenv import load_dotenv

# 1. Setup API and Database Connections
load_dotenv()
client = genai.Client()

# Connect to the ChromaDB we just built
db_path = os.path.join(os.path.dirname(__file__), '../Data/vector_db')
chroma_client = chromadb.PersistentClient(path=db_path)

# Retrieve our specific collection
collection = chroma_client.get_collection(name="ielts_high_scores")


def evaluate_with_rag(user_essay,question_text):
    """Retrieves context from ChromaDB and evaluates the essay."""

    print("\n1. Searching memory for similar high-scoring examples...")
    # Query the database for the 2 most similar essays
    results = collection.query(
        query_texts=[question_text],
        n_results=2
    )

    # 2. Extract the data and build our "Context" string
    context_string = ""
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    for i in range(len(documents)):
        score = metadatas[i]['overall_score']
        comment = metadatas[i]['examiner_comment']
        text = documents[i]

        context_string += f"\n--- Reference Essay {i + 1} (Human Score: {score}) ---\n"
        context_string += f"Text: {text}\n"
        context_string += f"Examiner Justification: {comment}\n"

    print("2. Constructing the RAG prompt...")
    # 3. Build the master prompt
    system_prompt = f"""
    You are a highly experienced, strict IELTS Writing Task 2 examiner with over 20 years of official marking experience.
    You are NOT a writing coach. You do NOT reward effort. You ONLY reward demonstrated performance.
    You must grade exactly according to official IELTS band descriptors. Every score must be justified with concrete linguistic evidence from the essay.

    ════════════════════════════════════════
    IELTS TASK 2 QUESTION:
    "{question_text}"
    ════════════════════════════════════════

    BAND DESCRIPTOR ANCHORS (Apply these precisely):
      Band 9 – Expert user. Fully accurate, flexible, sophisticated. Near-flawless.
      Band 8 – Very good user. Occasional minor errors. Handles complexity with ease.
      Band 7 – Good user. Some inaccuracies but manages complex language well.
      Band 6 – Competent. Meaning is generally clear but noticeable errors and limitations exist.
      Band 5 – Modest. Partial task completion. Frequent errors that cause difficulty.
      Band 4 – Limited. Frequent breakdowns in communication. Inadequate task response.
      Band 3 – Extremely limited. Very little relevant content. Severe errors throughout.
      Band 1-2 – Non-user. Essentially no communicative ability.

    ════════════════════════════════════════
    HIGH-SCORING REFERENCE ESSAYS (Band 7.5–9 Benchmarks):
    {context_string}

    Before grading, internalize the argument depth, vocabulary range, and grammatical 
    complexity demonstrated in the reference essays above. If the user essay falls 
    clearly below this standard in any criterion, score it lower accordingly.
    ════════════════════════════════════════

    MANDATORY EVALUATION PROCEDURE — FOLLOW IN ORDER:

    ──────────────────────────────────────
    STEP 1 — PRE-CHECK (BEFORE SCORING)
    ──────────────────────────────────────
    A) WORD COUNT CHECK:
       If the essay is under 250 words       → Cap Task Response at Band 5 maximum.
       If the essay is under 50 words or incoherent → Assign Band 1 across all criteria and stop.

    B) TASK MATCH CHECK:
       Fully answers all parts of the question → Proceed normally.
       Partially addresses the task            → Task Response maximum = Band 5.
       Ignores one part of a multi-part question → Task Response maximum = Band 5.
       Fails to present a clear position (if required) → Task Response maximum = Band 5.
       Discusses a related but different topic → Task Response maximum = Band 4.
       Completely off-topic                   → Task Response = Band 3 or lower.

       Apply these caps without exception.

    ──────────────────────────────────────
    STEP 2 — BAND 8+ GATEKEEPING (STRICT CEILING CONTROL)
    ──────────────────────────────────────
    Do NOT award Band 8 or higher unless ALL of the following are clearly demonstrated 
    with specific evidence from the essay:

      ✦ Sophisticated, non-generic argument development
      ✦ Analytical depth beyond surface-level reasoning
      ✦ Natural cohesion — no reliance on formulaic linking phrases
      ✦ Flexible, varied grammatical structures used with precision
      ✦ Precise and less common lexical choices used naturally (not memorised)
      ✦ Clear critical thinking or intellectual nuance

      If the essay is clear but conventional, safe, or formulaic → Overall maximum = Band 7.5
      If ideas are relevant but underdeveloped                   → Task Response maximum = Band 6
      If vocabulary is topic-appropriate but not flexible        → Lexical Resource maximum = Band 7
      If grammar is mostly accurate but not genuinely varied     → Grammar maximum = Band 7.5

    ──────────────────────────────────────
    STEP 3 — CRITERION-BY-CRITERION SCORING
    ──────────────────────────────────────
    Score each criterion independently on the 0–9 band scale (whole or half bands only).

    1. TASK RESPONSE
       - Does it fully answer ALL parts of the question?
       - Is a clear position maintained throughout?
       - Are ideas sufficiently developed with specific reasoning (not vague generalisations)?
       - Is the depth of argument proportional to a high-band response?

    2. COHERENCE & COHESION
       - Is there a logical, easy-to-follow progression of ideas?
       - Does each paragraph have a clear central idea?
       - Are cohesive devices used naturally, or mechanically/repetitively?
       - Does the essay rely on a rigid template structure?

    3. LEXICAL RESOURCE
       - Is vocabulary range wider than standard IELTS topic words?
       - Are collocations accurate and natural?
       - Is there repetition or overuse of the same phrases?
       - Are there signs of memorised or recycled language?

    4. GRAMMATICAL RANGE & ACCURACY
       - Is there genuine variety in clause types and sentence structures?
       - Are complex structures controlled accurately?
       - Are errors frequent, occasional, or rare?
       - Does grammatical complexity feel natural or forced?

    ──────────────────────────────────────
    STEP 4 — OVERALL SCORE CALCULATION
    ──────────────────────────────────────
    Overall Band Score = average of all 4 criterion scores, rounded to the nearest 0.5.
    When the result falls exactly between two bands, ALWAYS round DOWN.

    Example: (6.5 + 7.0 + 6.5 + 7.0) / 4 = 6.75 → rounds DOWN to 6.5

    ──────────────────────────────────────
    STEP 5 — ANTI-INFLATION REFLECTION (MANDATORY)
    ──────────────────────────────────────
    After assigning scores, you MUST explicitly address:
      - Why this essay does NOT qualify for a higher band.
      - What specific linguistic or argumentative elements are missing.
      - Whether the writing feels formulaic or exam-trained.

    If you cannot justify a higher band with concrete evidence from the essay, keep the score where it is.

    ════════════════════════════════════════
    USER ESSAY TO GRADE:
    {user_essay}
    ════════════════════════════════════════

    OUTPUT FORMAT — USE THIS EXACT STRUCTURE:

    Overall Band Score: X.X

    1. Task Response: X.X
       [Detailed justification with specific examples from the essay]

    2. Coherence & Cohesion: X.X
       [Detailed justification with specific examples from the essay]

    3. Lexical Resource: X.X
       [Detailed justification with specific examples from the essay]

    4. Grammatical Range & Accuracy: X.X
       [Detailed justification with specific examples from the essay]

    Why This Essay Does Not Score Higher:
       [Precise analysis of what is missing. No praise language. Evidence-based only.]

    Score Calculation:
       (TR + CC + LR + GRA) / 4 = X.X → Final Band: X.X
    """

    print("3. Sending to Gemini for evaluation...")
    # 4. Send to the AI
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"


# --- Testing Block ---
if __name__ == "__main__":
    # Let's test this with the same essay we used in the baseline test
    # to see if the RAG system grades it differently!
    dataset_path = os.path.join(os.path.dirname(__file__), '../Data/processed/cleaned_ielts_task2_essays.csv')

    try:
        df = pd.read_csv(dataset_path)

        # Grab the very first essay (which previously got a 6.5 from a human and a 5 from raw AI)
        test_essay = df.iloc[0]['essay']
        actual_score = df.iloc[0]['overall']

        print(f"\n--- Actual Human Examiner Score: {actual_score} ---")

        # Get the new RAG-powered grade
        rag_evaluation = evaluate_with_rag(test_essay)

        print("\n--- RAG AI Evaluation ---")
        print(rag_evaluation)

    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {dataset_path}")