"""
predict.py
----------
Loads the trained model artefacts and predicts the issue category for one
or more ticket descriptions supplied from the command line or via import.

Usage
-----
    # Predict a single ticket description
    python src/predict.py "I forgot my password and cannot log in."

    # Predict multiple tickets (separate each with a comma-free shell approach)
    python src/predict.py "App crashes on load" "I was charged twice" "Fraud on my card"
"""

import os
import re
import sys

import joblib

try:
    from groq import Groq
except ImportError:
    Groq = None

# ---------------------------------------------------------------------------
# Paths
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


def predict(texts: list[str]) -> list[dict]:
    """
    Predict the issue category for each text in *texts*.

    Parameters
    ----------
    texts : list[str]
        Raw (uncleaned) ticket descriptions.

    Returns
    -------
    list[dict]
        Each dict has keys ``text``, ``predicted_category``, ``confidence``,
        and ``probabilities`` (a dict mapping each class to its probability).
    """
    vectorizer, model = load_artifacts()

    # Minimal cleaning (match preprocessing step)
    def _clean(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r"hi support,?", "", text)
        text = re.sub(r"hello,?", "", text)
        text = re.sub(r"dear team,?", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    cleaned = [_clean(t) for t in texts]
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

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
# Rule-based fallback responses by category
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


# ---------------------------------------------------------------------------
# Groq API response generation
# ---------------------------------------------------------------------------
def generate_suggested_response(
    ticket_text: str,
    predicted_category: str,
    api_key: str | None = None,
    model_name: str = "openai/gpt-oss-20b",
) -> dict:
    """
    Generate an automatic customer response using the Groq API conditioned on
    the predicted category and customer ticket description.

    API / Provider Used:
        Groq Cloud API (https://console.groq.com) via official `groq` Python SDK.

    Model Used:
        openai/gpt-oss-20b (or user selected: openai/gpt-oss-120b, qwen/qwen3.8-27b, allam-2-7b).

    How the API was Integrated:
        1. API key detection: Retrieves key from argument or `GROQ_API_KEY` env var.
        2. Prompt engineering: Constructs a system prompt instructing the LLM to act
           as an empathetic, concise support specialist, conditioning generation
           specifically on the ML-predicted category and customer ticket.
        3. Inference call: Invokes `client.chat.completions.create` with temperature 0.4.
        4. Graceful fallback: If key is missing or an API error occurs, seamlessly returns
           a category-specific curated response so the workflow never breaks.

    Returns
    -------
    dict
        {
            "response": str,
            "provider": str ("Groq Cloud" or "Fallback Template"),
            "model": str,
            "success": bool,
            "error": str | None
        }
    """
    key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()

    if not key:
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": "Fallback Template (No Groq API Key)",
            "model": "rule-based",
            "success": False,
            "error": "No Groq API key provided. Set GROQ_API_KEY or provide key in UI.",
        }

    if Groq is None:
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": "Fallback Template (groq package missing)",
            "model": "rule-based",
            "success": False,
            "error": "The 'groq' package is not installed. Install via `pip install groq`.",
        }

    # Guard against prompt-guard / classification models being chosen for text generation
    is_classification_model = any(
        term in model_name.lower()
        for term in ["prompt-guard", "safeguard", "guard-2", "classification"]
    )
    if is_classification_model:
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": "Fallback Template (Model Type Mismatch)",
            "model": model_name,
            "success": False,
            "error": (
                f"'{model_name}' is a safety classification model, not a generative chat model. "
                "Please select a generative model such as 'openai/gpt-oss-20b', 'openai/gpt-oss-120b', "
                "or 'qwen/qwen3.8-27b' in the sidebar."
            ),
        }

    try:
        client = Groq(api_key=key)
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

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt_content},
            ],
            model=model_name,
            temperature=0.4,
            max_tokens=250,
        )

        response_text = chat_completion.choices[0].message.content.strip()
        return {
            "response": response_text,
            "provider": "Groq Cloud API",
            "model": model_name,
            "success": True,
            "error": None,
        }

    except Exception as exc:
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": "Fallback Template (Groq API Error)",
            "model": "rule-based",
            "success": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# CLI entry-point
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
        print(f"  {resp_info['response']}")
    print("-" * 60)
