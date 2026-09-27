"""
predict.py
----------
Loads the trained model artefacts and predicts the issue category for customer
support ticket descriptions. Generates suggested customer responses using
either a fast Groq LLM completion or a domain-specific template fallback.

Usage
-----
    # Predict a single ticket description
    python src/predict.py "I forgot my password and cannot log in."

    # Predict multiple tickets
    python src/predict.py "App crashes on load" "I was charged twice" "Fraud on my card"
"""

import os
import re
import sys

import joblib
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None

# ---------------------------------------------------------------------------
# Paths & Artifacts
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_artifacts():
    """Return (vectorizer, model) loaded from the model/ directory."""
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODEL_DIR, "classifier.pkl")

    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artefacts not found in '{MODEL_DIR}'. "
            "Please run `python src/train.py` first."
        )

    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    return vectorizer, model


def _model_probabilities(model, features):
    """Return probability scores for probabilistic and margin models."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)

    if not hasattr(model, "decision_function"):
        raise AttributeError(
            "The saved classifier exposes neither probabilities nor decision scores."
        )

    scores = np.asarray(model.decision_function(features), dtype=float)
    if scores.ndim == 1:
        scores = np.column_stack((-scores, scores))
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(scores)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Text Cleaning & Prediction
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip boilerplate greetings, remove non-alpha characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"hi support,?", "", text)
    text = re.sub(r"hello,?", "", text)
    text = re.sub(r"dear team,?", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(texts: list[str]) -> list[dict]:
    """
    Predict the issue category for each text in *texts*.

    Parameters
    ----------
    texts : list[str]
        Raw ticket descriptions.

    Returns
    -------
    list[dict]
        Each dict contains:
        - text: original ticket description
        - predicted_category: predicted class name
        - confidence: confidence score percentage
        - probabilities: dict of category -> probability percentage
    """
    vectorizer, model = load_artifacts()

    cleaned = [clean_text(t) for t in texts]
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    probabilities = _model_probabilities(model, features)

    results = []
    for text, pred, probs in zip(texts, predictions, probabilities):
        prob_dict = dict(zip(model.classes_, (float(p) for p in probs)))
        results.append(
            {
                "text": text,
                "predicted_category": pred,
                "confidence": round(float(max(probs)) * 100, 2),
                "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()},
            }
        )
    return results


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------
FALLBACK_RESPONSES = {
    "Account": (
        "Please use the 'Forgot Password' option on the login page to reset your "
        "password. If the problem continues, please contact the support team with your registered email."
    ),
    "Billing": (
        "Your billing inquiry has been received. Our finance team will review the transaction "
        "details and process any required adjustments within 1 to 2 business days."
    ),
    "Fraud": (
        "We have flagged your account for urgent security review. A risk specialist will investigate "
        "the reported activity immediately. In the meantime, please monitor your account statements."
    ),
    "General Inquiry": (
        "Thank you for contacting our support team. We have received your query and will provide "
        "the requested information shortly. You can also browse our online Help Center FAQs."
    ),
    "Technical": (
        "Our technical support team is investigating the issue. Please try restarting the application "
        "or clearing your browser cache. If the issue persists, reply with your device details and error logs."
    ),
}

DEFAULT_FALLBACK = (
    "Thank you for reaching out to support. We have received your ticket and an agent "
    "will respond shortly."
)


def generate_suggested_response(
    ticket_text: str,
    predicted_category: str,
    api_key: str | None = None,
    model_name: str = "llama-3.3-70b-versatile",
) -> dict:
    """
    Generate an actionable customer response using Groq API (if configured),
    or fall back to the category template response.
    """
    key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()

    if not key or Groq is None:
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": "Template Fallback",
            "model": "rule-based",
            "success": False,
            "error": "Groq API key not provided or groq package not installed.",
        }

    prompt_content = (
        "You are an empathetic, concise, and professional customer support specialist.\n"
        f"A customer submitted a support ticket, and our machine learning classifier identified its category as: '{predicted_category}'.\n\n"
        f"Customer Ticket Description:\n\"\"\"{ticket_text}\"\"\"\n\n"
        f"Predicted Category: {predicted_category}\n\n"
        "Task: Generate a clear, polite, and actionable resolution response directly addressed to the customer "
        "(2 to 4 sentences maximum). Acknowledge their issue, provide immediate helpful instructions tailored to "
        "the predicted category, and invite them to reply if they need further assistance. "
        "Do NOT include subject lines, placeholders, or meta-explanations; reply only with the message body.\n\n"
        "Suggested Response to Customer:"
    )

    try:
        client = Groq(api_key=key)
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt_content}],
            model=model_name,
            temperature=0.4,
            max_tokens=250,
        )
        text = completion.choices[0].message.content.strip()
        return {
            "response": text,
            "provider": "Groq AI",
            "model": model_name,
            "success": True,
            "error": None,
        }
    except Exception as exc:
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": "Template Fallback",
            "model": model_name,
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"<ticket description>\" [\"<ticket 2>\" ...]")
        sys.exit(1)

    inputs = sys.argv[1:]
    results = predict(inputs)

    for r in results:
        resp_info = generate_suggested_response(
            ticket_text=r["text"],
            predicted_category=r["predicted_category"],
        )
        print("-" * 60)
        print(f"Ticket             : {r['text']}")
        print(f"Predicted Category : {r['predicted_category']}")
        print(f"Confidence         : {r['confidence']:.2f}%")
        print("Probabilities      :")
        for cat, prob in sorted(r["probabilities"].items(), key=lambda x: -x[1]):
            print(f"  {cat:<18} {prob:.2f}%")
        print(f"Suggested Response [{resp_info['provider']}]:")
        # Ensure safe printing on Windows consoles
        safe_response = resp_info["response"].encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8")
        print(f"  {safe_response}")
    print("-" * 60)
