# 📝 IELTS Writing Master: RAG-Powered AI Evaluator

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google)

An intelligent, full-stack web application that evaluates IELTS Task 2 essays. Instead of relying on a generic LLM prompt, this project utilizes **Retrieval-Augmented Generation (RAG)**. By embedding official, high-scoring examiner essays into a local vector database, the AI references factual grading rubrics and contextual examples before evaluating user submissions, completely eliminating AI hallucinations and harsh grading biases.

---

## ✨ Key Features
* **RAG Architecture:** Contextually searches a ChromaDB vector database for Band 8/9 essays matching the user's specific assignment topic.
* **Prompt-Aware Grading:** Evaluates the user's essay against the specific prompt to accurately score the "Task Response" criteria.
* **Real-Time Word Tracking:** Dynamic frontend word counter to ensure users meet the strict 250-word minimum requirement.
* **Markdown Rendering:** Beautifully formatted, easy-to-read examiner feedback using `react-markdown`.
* **Decoupled Deployment:** Cloud-hosted via Render (Backend) and Vercel (Frontend) with strict CORS security.

## 🏗️ Architecture & Tech Stack

**Frontend (Client)**
* **Framework:** React.js (via Vite)
* **Styling:** CSS3 with modern UI/UX principles
* **Deployment:** Vercel

**Backend (API & AI)**
* **Framework:** FastAPI (Python 3.12) & Uvicorn
* **LLM Engine:** Google Gemini 2.5 Flash API (`google-genai` SDK)
* **Vector Database:** ChromaDB (Local persistent storage)
* **Data Processing:** Pandas
* **Deployment:** Render

### 📂 Project Structure
```text
IELTS_RAG/
├── Backend/
│   ├── build_vectordb.py      # Script to embed CSV data into ChromaDB
│   ├── main.py                # FastAPI server and CORS configuration
│   ├── rag_evaluator.py       # Core RAG logic and Gemini API integration
│   └── __init__.py
├── Data/
│   ├── processed/
│   │   └── cleaned_ielts_task2_essays.csv  # Cleaned source dataset
│   └── vector_db/             # Auto-generated ChromaDB local storage
├── Frontend/
│   ├── src/
│   │   ├── App.jsx            # Main React component and API fetch logic
│   │   ├── App.css            # UI styling and spinner animations
│   │   └── prompts.js         # Library of IELTS Task 2 assignments
│   ├── package.json           # Node.js dependencies
│   └── vite.config.js
├── .env                       # Secret API keys (Git-ignored)
├── .gitignore                 # Security rules for Git
└── requirements.txt           # Python backend dependencies
```
---

## 🚀 Local Setup Instructions

To run this project locally, you will need **Python 3.12+** and **Node.js** installed on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/ielts-ai-evaluator.git
```
```bash
cd ielts-ai-evaluator
```

### 2. Backend Setup (FastAPI & Vector Database)

Open a terminal in the root directory and set up the Python environment:

```bash
# Create and activate a virtual environment
python -m venv .venv
# On Windows: .\.venv\Scripts\activate
# On Mac/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Environment Variables:**
Create a `.env` file in the root directory and add your Google Gemini API key:

```text
GEMINI_API_KEY=your_secret_api_key_here
```

**Build the Vector Database:**
Before running the server, you must generate the local ChromaDB database from the dataset.

```bash
python Backend/build_vectordb.py
```

**Start the API Server:**

```bash
python Backend/main.py
# The API will be available at http://127.0.0.1:8000
```

### 3. Frontend Setup (React)

Open a *second* terminal window and navigate to the frontend folder:

```bash
cd Frontend

# Install Node dependencies
npm install

# Start the development server
npm run dev
# The UI will be available at http://localhost:5173
```

---

## 🧠 How the RAG Pipeline Works

1. **Assignment Generation:** The React frontend assigns a specific IELTS topic to the user.
2. **Data Submission:** The user's essay and the assignment prompt are sent to the FastAPI backend.
3. **Vector Retrieval:** The backend queries ChromaDB using the *assignment prompt* to find the most relevant top-tier essays and examiner comments.
4. **Context Injection:** The retrieved "Gold Standard" essays are injected into the Gemini System Prompt alongside the user's essay.
5. **Evaluation:** Gemini evaluates the submission using the injected contextual rubric, returning a highly accurate, human-like score and detailed justification.

---

## 🔮 Future Improvements

* Integrate user authentication (JWT) to save past essays and track score improvements over time.
* Add a "Grammar Highlight" feature to visually flag specific sentences in the frontend.
* Expand the vector database with IELTS Task 1 (Graph/Chart analysis) datasets.

---

> *Disclaimer: This is an educational project and is not affiliated with the official IELTS organization.*
