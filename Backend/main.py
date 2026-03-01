# Backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import the RAG function you just built!
from Backend.rag_evaluator import evaluate_with_rag

# Initialize the API
app = FastAPI(title="IELTS RAG Evaluator API")

# --- NEW CORS SECTION ---
# This allows your React frontend to talk to this FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define the format we expect to receive from the frontend website
class EssayRequest(BaseModel):
    question_text: str
    essay_text: str


# Create the endpoint that the website will send data to
@app.post("/api/evaluate")
async def evaluate_essay(request: EssayRequest):
    if not request.essay_text or len(request.essay_text) < 50:
        raise HTTPException(status_code=400, detail="Essay is too short or empty.")
    if not request.question_text or len(request.question_text) < 10:
        raise HTTPException(status_code=400, detail="Assignment prompt is missing.")

    print(f"\n--- Evaluating essay for prompt: {request.question_text[:50]}... ---")

    # Pass the essay to your ChromaDB/Gemini script
    try:
        feedback = evaluate_with_rag(request.essay_text,request.question_text)

        # Send the final grade back to the user's browser
        return {"status": "success", "evaluation": feedback}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# This block actually starts the local server
if __name__ == "__main__":
    print("Starting the FastAPI web server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)