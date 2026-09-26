"""
app.py
------
Streamlit web application for interactive customer-support ticket
classification.  Run with:

    streamlit run src/app.py
"""

import os
import sys

import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup so we can import predict.py from the same src/ folder
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from predict import generate_suggested_response, predict  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Support Ticket Classifier | AI-Powered Triage",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }
        .hero {
            background: linear-gradient(135deg, #1f2d5a 0%, #2e4fa3 55%, #3a6bd8 100%);
            border-radius: 14px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 1.6rem;
            color: #ffffff;
            box-shadow: 0 6px 18px rgba(31, 45, 90, 0.25);
        }
        .hero h1 { margin: 0; font-size: 2rem; font-weight: 700; color: #ffffff; }
        .hero p  { margin: 0.45rem 0 0 0; font-size: 1.02rem; color: #d7e1f7; }

        .card {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128,128,128,0.18);
            border-radius: 14px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .card h3 { margin-top: 0; font-size: 1.05rem; font-weight: 600; }

        .badge {
            display: inline-block;
            padding: 0.5rem 1.4rem;
            border-radius: 999px;
            font-size: 1.25rem;
            font-weight: 700;
            color: #ffffff;
        }

        .reply-box {
            border-left: 4px solid #3a6bd8;
            background: rgba(58,107,216,0.08);
            border-radius: 0 10px 10px 0;
            padding: 1.1rem 1.3rem;
            font-size: 1rem;
            line-height: 1.6;
        }

        .provider-badge {
            display: inline-block;
            padding: 0.28rem 0.8rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-right: 0.5rem;
            margin-bottom: 0.6rem;
            border: 1px solid rgba(128,128,128,0.25);
        }
        .groq-badge {
            background: rgba(245, 80, 54, 0.12);
            color: #d9381e;
            border-color: rgba(245, 80, 54, 0.35);
        }
        .fallback-badge {
            background: rgba(100, 116, 139, 0.12);
            color: #64748b;
            border-color: rgba(100, 116, 139, 0.3);
        }

        div.stButton > button[kind="primary"] {
            border-radius: 10px;
            font-weight: 600;
            font-size: 1.05rem;
            padding: 0.55rem 0;
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(58,107,216,0.35);
        }

        .footer {
            text-align: center;
            color: #9aa0a6;
            font-size: 0.82rem;
            margin-top: 2.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(BASE_DIR, "model")


@st.cache_resource
def load_model_artifacts():
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    model_path = os.path.join(MODEL_DIR, "classifier.pkl")
    if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
        return None, None
    return joblib.load(vectorizer_path), joblib.load(model_path)


vectorizer, model = load_model_artifacts()

# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------
CATEGORY_META = {
    "Account":         {"icon": "🔐", "color": "#2563eb"},
    "Billing":         {"icon": "💳", "color": "#7c3aed"},
    "Fraud":           {"icon": "🚨", "color": "#dc2626"},
    "General Inquiry": {"icon": "💬", "color": "#059669"},
    "Technical":       {"icon": "🛠️", "color": "#d97706"},
}

SUGGESTED_RESPONSES = {
    "Account":         "Please follow the credential update instructions sent to your registered email address.",
    "Billing":         "Your request has been routed to our billing team. We will review your transaction details shortly.",
    "Fraud":           "Your security flag has been escalated. Our risk management team is investigating this issue urgently.",
    "General Inquiry": "Thank you for reaching out! Please refer to our help center FAQs for quick assistance.",
    "Technical":       "Our engineering team has received your ticket. Please try re-logging into the application.",
}

DEFAULT_REPLY = "Thank you for contacting support. Our team will review your query shortly."

SAMPLE_TICKETS = {
    "— Select a sample ticket —":  "",
    "Account: login issue":        "I forgot my password and cannot log into my account dashboard.",
    "Billing: double charge":      "I was charged twice for my monthly subscription and need a refund.",
    "Fraud: suspicious activity":  "There are unauthorized transactions on my card that I did not make.",
    "General: product info":       "Can you tell me what plans you offer and their pricing?",
    "Technical: app crash":        "The mobile app crashes every time I try to upload a document.",
}


def confidence_label(score: float) -> str:
    if score >= 85:
        return "🟢 High confidence"
    if score >= 60:
        return "🟡 Medium confidence"
    return "🔴 Low confidence — recommend manual review"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📂 Supported Categories")
    for name, meta in CATEGORY_META.items():
        st.markdown(f"{meta['icon']} **{name}**")
    st.divider()

    st.subheader("⚡ Groq API Settings")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Enter your Groq API key (starts with gsk_). Can also be set in GROQ_API_KEY env var.",
        placeholder="gsk_...",
    )
    GROQ_MODELS = [
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "qwen/qwen3.8-27b",
        "allam-2-7b",
    ]
    groq_model = st.selectbox(
        "Groq Chat Model",
        options=GROQ_MODELS,
        index=0,
        help="Select a generative chat model to draft replies (Prompt Guard models are classifiers and cannot generate text).",
    )
    if groq_api_key.strip():
        st.success("🟢 Groq API Key Configured")
    else:
        st.info("💡 Paste your Groq API Key to enable dynamic LLM responses.")

    st.divider()

    st.subheader("⚙️ Classifier Model Status")
    if vectorizer is not None and model is not None:
        st.success("Model artefacts loaded ✅")
    else:
        st.error("Model not found — run `python src/train.py` first.")

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎫 Customer Support Ticket Classifier</h1>
        <p>AI-powered triage — paste a support request to instantly predict
           its category, confidence level, and a suggested agent reply.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Guard: model availability
# ---------------------------------------------------------------------------
if vectorizer is None or model is None:
    st.error(
        "⚠️ **Model artefacts not found.**  "
        "Run `python src/train.py` to train the model, then restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
input_col, side_col = st.columns([2.2, 1], gap="large")

with input_col:
    st.markdown('<div class="card"><h3>📝 Ticket Description</h3>', unsafe_allow_html=True)

    if "ticket_text" not in st.session_state:
        st.session_state.ticket_text = ""

    ticket_description = st.text_area(
        label="Ticket Description",
        value=st.session_state.ticket_text,
        placeholder=(
            "Describe the customer's issue here…\n\n"
            "e.g., I forgot my password and cannot log into my account dashboard."
        ),
        height=170,
        label_visibility="collapsed",
    )

    st.caption(f"{len(ticket_description)} characters")

    predict_clicked = st.button(
        "🔍 Classify & Generate Response",
        type="primary",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with side_col:
    st.markdown('<div class="card"><h3>⚡ Try a Sample Ticket</h3>', unsafe_allow_html=True)
    sample_choice = st.selectbox(
        "Sample tickets",
        options=list(SAMPLE_TICKETS.keys()),
        label_visibility="collapsed",
    )
    if sample_choice != "— Select a sample ticket —":
        st.session_state.ticket_text = SAMPLE_TICKETS[sample_choice]
        st.info(SAMPLE_TICKETS[sample_choice])
    else:
        st.caption("Pick an example to auto-fill the ticket box.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Prediction & output
# ---------------------------------------------------------------------------
if predict_clicked:
    if not ticket_description.strip():
        st.warning("⚠️ Please enter a ticket description before predicting.")
    else:
        results = predict([ticket_description])
        r = results[0]

        with st.spinner("🤖 Generating suggested response with Groq LLM..."):
            response_info = generate_suggested_response(
                ticket_text=ticket_description,
                predicted_category=r["predicted_category"],
                api_key=groq_api_key,
                model_name=groq_model,
            )

        st.session_state.last_result = {
            "ticket_text": ticket_description,
            "category": r["predicted_category"],
            "probs": r["probabilities"],
            "score": r["confidence"],
            "response_info": response_info,
        }

# ---------------------------------------------------------------------------
# Results rendering (persisted across reruns)
# ---------------------------------------------------------------------------
result = st.session_state.get("last_result")

if result:
    import pandas as pd

    predicted_category = result["category"]
    probs = result["probs"]
    confidence_score = result["score"]
    resp_info = result.get("response_info") or generate_suggested_response(
        ticket_text=result.get("ticket_text", ""),
        predicted_category=predicted_category,
        api_key=groq_api_key,
        model_name=groq_model,
    )

    meta = CATEGORY_META.get(predicted_category, {"icon": "🏷️", "color": "#64748b"})

    st.divider()
    st.subheader("📊 Classification & Prediction Output")

    metric_col, chart_col = st.columns([1, 1.6], gap="large")

    with metric_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Predicted Category**")
        st.markdown(
            f'<span class="badge" style="background:{meta["color"]};">'
            f'{meta["icon"]} {predicted_category}</span>',
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.metric(label="Confidence Level", value=f"{confidence_score:.2f}%")
        st.progress(min(max(int(confidence_score), 0), 100) / 100)
        st.caption(confidence_label(confidence_score))
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col:
        st.markdown('<div class="card"><h3>📈 Category Probability Distribution</h3>', unsafe_allow_html=True)
        prob_df = (
            pd.DataFrame(
                {"Category": list(probs.keys()), "Probability (%)": list(probs.values())}
            )
            .sort_values("Probability (%)", ascending=True)
            .set_index("Category")
        )
        st.bar_chart(prob_df, horizontal=True, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Suggested automatic customer response
    # -----------------------------------------------------------------------
    st.subheader("💬 Suggested Customer Response")

    is_groq = resp_info.get("success", False)
    provider_name = resp_info.get("provider", "Rule-based Fallback")
    model_name = resp_info.get("model", "N/A")

    badge_cls = "groq-badge" if is_groq else "fallback-badge"
    badge_html = (
        f'<span class="provider-badge {badge_cls}">⚡ Provider: {provider_name}</span>'
        f'<span class="provider-badge {badge_cls}">🧠 Model: {model_name}</span>'
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    if not is_groq and not groq_api_key.strip():
        st.caption(
            "💡 **Notice**: Enter your Groq API key in the sidebar to enable dynamic "
            "LLM-generated responses conditioned on the customer's problem. Showing standard template below."
        )
    elif not is_groq and resp_info.get("error"):
        st.warning(f"Groq Notice: {resp_info.get('error')}. Displaying fallback response.")

    st.markdown(
        f'<div class="reply-box">{resp_info["response"]}</div>',
        unsafe_allow_html=True,
    )

    action_col1, action_col2, _ = st.columns([1, 1.4, 3])
    with action_col1:
        if st.button("📋 Copy Response"):
            st.toast("Response text ready — select and copy the text box above.", icon="📋")
    with action_col2:
        if st.button("🔄 Regenerate with Groq"):
            with st.spinner("Regenerating with Groq..."):
                new_resp = generate_suggested_response(
                    ticket_text=result.get("ticket_text", ""),
                    predicted_category=predicted_category,
                    api_key=groq_api_key,
                    model_name=groq_model,
                )
                st.session_state.last_result["response_info"] = new_resp
                st.rerun()

    # -----------------------------------------------------------------------
    # Integration architecture details (API, Model, Integration)
    # -----------------------------------------------------------------------
    with st.expander("ℹ️ Groq API Integration & Architecture Details", expanded=False):
        st.markdown(
            """
            ### 📌 System Architecture & Groq Integration

            | Attribute | Details |
            | :--- | :--- |
            | **API / Provider Used** | **Groq Cloud API** (`https://console.groq.com`) via official `groq` Python SDK |
            | **Model Used** | **`openai/gpt-oss-20b`** (Primary chat model; `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `allam-2-7b` also supported) |
            | **Inference Hardware** | Groq Language Processing Units (LPUs) delivering ultra-low-latency generation |
            | **Task** | Automated, category-conditioned resolution drafting for support tickets |

            #### 🔄 Step-by-Step Integration Pipeline:
            1. **ML Ticket Classification**: The customer's raw ticket text is cleaned and transformed via TF-IDF vectorization. A Scikit-Learn Logistic Regression model predicts the primary issue category (`Account`, `Billing`, `Fraud`, `General Inquiry`, `Technical`) and computes class probabilities.
            2. **Prompt Conditioning**: The predicted category and original ticket text are injected into a structured system and user prompt. The system prompt directs the LLM to act as a concise, empathetic customer support specialist.
            3. **Groq API Execution**: The payload is dispatched to Groq's Chat Completions endpoint (`client.chat.completions.create`) using the selected model (`openai/gpt-oss-20b` by default) with temperature `0.4` and max tokens `250`.
            4. **Graceful Fallback**: If an API key is not supplied or if rate limits/network issues occur, the system smoothly falls back to a curated category-specific resolution template, ensuring zero customer downtime.
            """
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer">Support Ticket Classifier — automated ticket triage '
    "for faster, more consistent customer service.</div>",
    unsafe_allow_html=True,
)
