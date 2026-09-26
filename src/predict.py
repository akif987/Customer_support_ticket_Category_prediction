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
import random
import re
import sys

import httpx
import joblib
import numpy as np

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


def _model_probabilities(model, features):
    """Return probability-like scores for probabilistic and margin models."""
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
# Universal Provider Configurations & Model Catalogs
# ---------------------------------------------------------------------------
RANDOM_MODEL_LABEL = "🎲 Random Model (Auto-Select Behind the Scenes)"

PROVIDER_CONFIGS = {
    "Groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_models": [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "allam-2-7b",
        ],
        "env_key": "GROQ_API_KEY",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "default_models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ],
        "env_key": "OPENAI_API_KEY",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_models": [
            "meta-llama/llama-3.3-70b-instruct",
            "mistralai/mistral-7b-instruct",
            "qwen/qwen-2.5-72b-instruct",
            "google/gemini-2.0-flash-exp:free",
        ],
        "env_key": "OPENROUTER_API_KEY",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "default_models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "env_key": "DEEPSEEK_API_KEY",
    },
    "Custom (OpenAI-Compatible)": {
        "base_url": "http://localhost:11434/v1",
        "default_models": [
            "llama3",
            "mistral",
            "qwen",
        ],
        "env_key": "LLM_API_KEY",
    },
}

NON_CHAT_KEYWORDS = [
    "guard", "safeguard", "whisper", "tts", "embedding", "moderation",
    "dall-e", "davinci", "babbage", "curie", "audio", "embed"
]


def fetch_available_models(
    provider: str = "Groq",
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """
    Fetch active chat models from the LLM provider API.
    Filters out non-chat models (audio, moderation, guard, embedding).
    """
    cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["Groq"])
    defaults = cfg["default_models"]
    key = (
        api_key
        or os.environ.get(cfg.get("env_key", ""), "")
        or os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    ).strip()

    if not key and provider != "Custom (OpenAI-Compatible)":
        return list(defaults)

    # 1. Native Groq SDK model discovery
    if provider == "Groq" and Groq is not None and not base_url:
        try:
            client = Groq(api_key=key)
            models = client.models.list()
            chat_models = [
                m.id for m in models.data
                if not any(k in m.id.lower() for k in NON_CHAT_KEYWORDS)
            ]
            if chat_models:
                return chat_models
        except Exception:
            pass

    # 2. Universal /models endpoint over HTTPX
    endpoint_url = (base_url or cfg["base_url"]).rstrip("/") + "/models"
    try:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = httpx.get(endpoint_url, headers=headers, timeout=4.0)
        if resp.status_code == 200:
            data = resp.json()
            models_data = data.get("data", [])
            chat_models = [
                m["id"] for m in models_data
                if isinstance(m, dict) and "id" in m and not any(k in m["id"].lower() for k in NON_CHAT_KEYWORDS)
            ]
            if chat_models:
                return chat_models
    except Exception:
        pass

    return list(defaults)


# ---------------------------------------------------------------------------
# Universal response generation with random model selection behind the scenes
# ---------------------------------------------------------------------------
def generate_suggested_response(
    ticket_text: str,
    predicted_category: str,
    api_key: str | None = None,
    provider: str = "Groq",
    model_name: str | None = None,
    base_url: str | None = None,
    model_pool: list[str] | None = None,
) -> dict:
    """
    Generate an automatic customer response using any LLM API, with the ability
    to randomly select a model behind the scenes.

    Parameters
    ----------
    ticket_text : str
        The customer's problem description.
    predicted_category : str
        The ML-classified issue category.
    api_key : str, optional
        API key for the selected provider.
    provider : str, default "Groq"
        Target provider ("Groq", "OpenAI", "OpenRouter", "DeepSeek", "Custom").
    model_name : str, optional
        Specific model name or RANDOM_MODEL_LABEL to auto-pick randomly.
    base_url : str, optional
        Custom API base URL.
    model_pool : list[str], optional
        Candidate models pool to randomly select from.
    """
    cfg = PROVIDER_CONFIGS.get(provider, PROVIDER_CONFIGS["Groq"])
    key = (
        api_key
        or os.environ.get(cfg.get("env_key", ""), "")
        or os.environ.get("GROQ_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    ).strip()

    if not key and provider != "Custom (OpenAI-Compatible)":
        return {
            "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
            "provider": f"Fallback Template (No {provider} API Key)",
            "model": "rule-based",
            "randomly_selected": False,
            "candidate_pool": [],
            "success": False,
            "error": f"No {provider} API key provided. Set in sidebar or environment.",
        }

    # Discover candidate models
    candidates = model_pool or fetch_available_models(provider, key, base_url)
    candidates = [
        m for m in candidates
        if not any(k in m.lower() for k in NON_CHAT_KEYWORDS)
    ]
    if not candidates:
        candidates = list(cfg["default_models"])

    # Determine whether random model selection was requested
    is_random = (
        not model_name
        or model_name == RANDOM_MODEL_LABEL
        or "random" in str(model_name).lower()
    )

    if is_random:
        chosen_model = random.choice(candidates)
    else:
        chosen_model = model_name

    # Build sequence of models to try (failsafe: retry another candidate if first fails)
    models_to_try = [chosen_model]
    if is_random and len(candidates) > 1:
        other_candidates = [m for m in candidates if m != chosen_model]
        random.shuffle(other_candidates)
        models_to_try.extend(other_candidates[:2])

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
    messages = [{"role": "user", "content": prompt_content}]

    last_error = None

    for current_model in models_to_try:
        try:
            # Option 1: Native Groq SDK if provider is Groq
            if provider == "Groq" and Groq is not None and not base_url:
                client = Groq(api_key=key)
                completion = client.chat.completions.create(
                    messages=messages,
                    model=current_model,
                    temperature=0.4,
                    max_tokens=250,
                )
                text = completion.choices[0].message.content.strip()
                return {
                    "response": text,
                    "provider": f"{provider} Cloud API",
                    "model": current_model,
                    "randomly_selected": is_random,
                    "candidate_pool": candidates,
                    "success": True,
                    "error": None,
                }

            # Option 2: Universal HTTPX client for any OpenAI-compatible provider
            target_base = (base_url or cfg["base_url"]).rstrip("/")
            endpoint = f"{target_base}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if key:
                headers["Authorization"] = f"Bearer {key}"

            payload = {
                "model": current_model,
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 250,
            }
            resp = httpx.post(endpoint, headers=headers, json=payload, timeout=25.0)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()
                return {
                    "response": text,
                    "provider": f"{provider} API",
                    "model": current_model,
                    "randomly_selected": is_random,
                    "candidate_pool": candidates,
                    "success": True,
                    "error": None,
                }
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text}"

        except Exception as exc:
            last_error = str(exc)

    return {
        "response": FALLBACK_RESPONSES.get(predicted_category, DEFAULT_FALLBACK),
        "provider": f"Fallback Template ({provider} Error)",
        "model": chosen_model,
        "randomly_selected": is_random,
        "candidate_pool": candidates,
        "success": False,
        "error": last_error,
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
