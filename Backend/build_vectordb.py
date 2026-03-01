# Backend/build_vectordb.py
import pandas as pd
import chromadb
import os
from google import genai
from dotenv import load_dotenv

# 1. Setup API and Database Client
load_dotenv()
client = genai.Client()

# This creates a local folder called 'vector_db' inside your Data folder
db_path = os.path.join(os.path.dirname(__file__), '../Data/vector_db')
chroma_client = chromadb.PersistentClient(path=db_path)

# Create a collection (like a table in a traditional database)
# We use get_or_create so we don't cause an error if we run the script twice
collection = chroma_client.get_or_create_collection(name="ielts_high_scores")

# 2. Load and Filter the Data
dataset_path = os.path.join(os.path.dirname(__file__), '../Data/processed/cleaned_ielts_task2_essays.csv')
df = pd.read_csv(dataset_path)

# CRITICAL: We only want the AI to reference GREAT essays.
# Let's filter for overall scores of 7.5 or higher.
high_score_df = df[df['overall'] >= 7.5].reset_index(drop=True)
print(f"Found {len(high_score_df)} high-scoring essays to add to the database.")

# 3. Process and Insert into ChromaDB
print("Embedding and storing essays... (This might take a minute)")

documents = []
metadatas = []
ids = []

# Loop through the high-scoring essays to prepare them
for index, row in high_score_df.iterrows():
    # The text the AI will read
    essay_text = str(row['essay'])

    # The background info we want to keep attached to the text
    metadata = {
        "overall_score": float(row['overall']),
        "task_response": float(row.get('task_response', 0)),
        "examiner_comment": str(row.get('examiner_commen', 'No comment provided'))
    }

    documents.append(essay_text)
    metadatas.append(metadata)
    ids.append(f"essay_{index}")

# We add the data in batches to avoid overwhelming the system
# ChromaDB will automatically convert the text into embeddings by default!
batch_size = 50
for i in range(0, len(documents), batch_size):
    collection.add(
        documents=documents[i:i + batch_size],
        metadatas=metadatas[i:i + batch_size],
        ids=ids[i:i + batch_size]
    )
    print(f"Added batch {i} to {i + batch_size}...")

print("\nDatabase built successfully! Your AI now has a memory.")