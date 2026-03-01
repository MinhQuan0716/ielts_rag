import { useState, useEffect } from "react";
import "./App.css";
import { ieltsPrompts } from "./prompt";
import ReactMarkdown from "react-markdown";
function App() {
  const [essay, setEssay] = useState("");
  const [currentPrompt, setCurrentPrompt] = useState("");
  const [feedback, setFeedback] = useState("");
  const [loading, setLoading] = useState(false);
  const wordCount = essay
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0).length;
  // Pick a random prompt when the page first loads
  useEffect(() => {
    generateNewPrompt();
  }, []);

  const generateNewPrompt = () => {
    const randomIndex = Math.floor(Math.random() * ieltsPrompts.length);
    setCurrentPrompt(ieltsPrompts[randomIndex]);
    setFeedback(""); // Clear old feedback when starting a new challenge
  };

  const submitEssay = async () => {
    if (essay.length < 50) {
      alert("Please enter a longer essay.");
      return;
    }

    setLoading(true);
    setFeedback("");

    try {
      // This sends the essay to your FastAPI backend!
      const response = await fetch(
        "https://ielts-ai-backend.onrender.com/api/evaluate",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question_text: currentPrompt, // Sending the prompt
            essay_text: essay, // Sending the essay
          }),
        },
      );

      const data = await response.json();

      if (response.ok) {
        setFeedback(data.evaluation);
      } else {
        setFeedback("Error: " + data.detail);
      }
    } catch (error) {
      setFeedback(
        "Failed to connect to the server. Is FastAPI running?",
        error,
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>IELTS AI Evaluator</h1>
      <p>Powered by RAG and Official Examiner Rubrics</p>

      {/* --- ASSIGNMENT SECTION --- */}
      <div
        style={{
          backgroundColor: "#1a73e8",
          padding: "20px",
          borderRadius: "10px",
          border: "1px solid #f1c40f",
          marginBottom: "20px",
        }}
      >
        <h3 style={{ marginTop: 0, color: "#d4ac0d" }}>Current Assignment:</h3>
        <p style={{ fontStyle: "italic", fontSize: "1.1rem" }}>
          "{currentPrompt}"
        </p>
        <button
          onClick={generateNewPrompt}
          style={{ fontSize: "0.8rem", padding: "5px 10px" }}
        >
          Get Different Question
        </button>
      </div>

      <textarea
        placeholder="Paste your Task 2 essay here..."
        value={essay}
        onChange={(e) => setEssay(e.target.value)}
        rows={12}
        style={{
          width: "100%",
          padding: "10px",
          fontSize: "16px",
          borderRadius: "8px",
        }}
      />

      <br />

      <div
        style={{
          display: "flex",
          justifyContent: "flex-end",
          marginTop: "5px",
        }}
      >
        <p
          style={{
            margin: 0,
            fontSize: "0.9rem",
            fontWeight: "bold",
            color:
              wordCount >= 250
                ? "#388e3c"
                : "#d32f2f" /* Green if >= 250, Red if < 250 */,
          }}
        >
          Word Count: {wordCount} / 250 minimum
        </p>
      </div>

      <button
        onClick={submitEssay}
        disabled={loading}
        style={{
          marginTop: "10px",
          padding: "10px 20px",
          fontSize: "16px",
          cursor: "pointer",
          borderRadius: "8px",
          backgroundColor: loading ? "#ccc" : "#1a73e8",
          color: "white",
          border: "none",
        }}
      >
        {loading ? "Grading..." : "Evaluate Essay"}
      </button>

      {loading && (
        <div className="spinner-container">
          <div className="spinner"></div>
          <p className="spinner-text">
            AI is reading official rubrics and evaluating your essay...
          </p>
        </div>
      )}

      {!loading && feedback && (
        <div
          className="feedback-box"
          style={{
            marginTop: "30px",
            padding: "25px",
            backgroundColor: "#e8f0fe",
            borderRadius: "12px",
            textAlign: "left",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
          }}
        >
          <h2 style={{ color: "#1a73e8" }}>Official Evaluation:</h2>
          <div
            style={{ lineHeight: "1.6", fontSize: "1.05rem", color: "#2c3e50" }}
          >
            <ReactMarkdown>{feedback}</ReactMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
