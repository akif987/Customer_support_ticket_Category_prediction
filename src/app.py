import html
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import pandas as pd
import streamlit as st

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from predict import (  # noqa: E402
    generate_suggested_response,
    predict,
)

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
        <style>
    :root { color-scheme: light; }
    .stApp {
        background: #0b1a3d;
        color: #000000;
        font-family: -apple-system, sans-serif;
    }
    .block-container { max-width: 900px; padding-top: 2.5rem; padding-bottom: 3rem; }
    header[data-testid="stHeader"] { background: transparent; }
    header[data-testid="stHeader"] button { color: white; }
    #MainMenu { visibility: hidden; }

    /* Outer Ring */
    .st-key-ticket_front_card, .st-key-ticket_result_card {
        position: relative;
        background: #ffffff;
        border: 4px solid #6b9e59;
        border-radius: 56px;
        padding: 14px;
        box-shadow: 0 23px 70px rgba(0, 9, 39, .37);
        transform-origin: center center;
        transform-style: preserve-3d;
        backface-visibility: visible;
    }
    
    /* Blue inner card */
    .st-key-card_inside {
        background: #4776c4;
        border-radius: 42px;
        padding: 0;
        min-height: 520px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
    }

    /* Plaque */
    .title-plaque {
        background: linear-gradient(180deg, #6c9ce3 0%, #2f569b 100%);
        clip-path: polygon(7% 0, 93% 0, 100% 18%, 100% 100%, 0 100%, 0 18%);
        padding: 2.5rem 2rem 2.5rem;
        margin: 1.5rem auto 0;
        width: 82%;
        text-align: center;
    }
    .title-bar {
        background: #23438f;
        color: #ffffff;
        font-size: 1.35rem;
        font-weight: 800;
        padding: 0.8rem;
        border-radius: 4px;
    }
    .title-sub {
        background: #ffffff;
        border: 1px solid #000000;
        color: #000000;
        padding: 1.2rem;
        font-size: 1rem;
        margin-top: 0;
    }
    
    /* White Content Panels */
    .st-key-input_panel, .st-key-result_panel {
        background: #ffffff;
        padding: 2.5rem 3rem;
        margin-top: 0;
        flex: 1;
    }

    /* Text & Input Styling on White Background */
    .field-label { color: #000000; font-weight: 800; font-size: 1rem; margin-bottom: 0.5rem; }
    .mini-label { font-size: 0.8rem; color: #555555; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.5rem; }
    .result-value { font-size: 1.6rem; font-weight: 850; line-height: 1.3; color: #000000; }
    .reply-text { color: #000000; white-space: pre-wrap; line-height: 1.7; font-size: 1rem; }
    .response-panel { background: #f3f7ff; border-left: 4px solid #4776c4; padding: 1rem 1.2rem; border-radius: 8px; }
    
    .st-key-card_inside [data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        border: 1px solid #000000 !important;
        border-radius: 4px !important;
        color: #000000 !important;
        min-height: 180px;
    }
    .st-key-card_inside [data-testid="stButton"] button[kind="primary"] {
        background: #ff4b55 !important;
        color: #ffffff !important;
        border: 0 !important;
        border-radius: 12px !important;
        min-height: 3.5rem !important;
        font-weight: 750 !important;
        font-size: 1.1rem !important;
    }
    .st-key-card_inside [data-testid="stCaptionContainer"] { color: #888888 !important; }
    .st-key-card_inside hr { border-color: #dddddd !important; margin: 1.5rem 0 !important; }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Animations */
    .st-key-ticket_front_card { animation: flipBack .65s cubic-bezier(.2,.72,.22,1) both; }
    .st-key-ticket_result_card { animation: ringGlowThenFlip 1.9s cubic-bezier(.22,.8,.25,1) forwards; }

    @keyframes ringGlowThenFlip {
        0% { transform: perspective(1400px) rotateY(0deg); box-shadow: 0 0 30px 10px rgba(107, 158, 89, 0.8); }
        35% { transform: perspective(1400px) rotateY(0deg); box-shadow: 0 0 60px 20px rgba(107, 158, 89, 1); }
        45% { transform: perspective(1400px) rotateY(30deg); box-shadow: 0 0 40px 10px rgba(107, 158, 89, 0.6); }
        65% { transform: perspective(1400px) rotateY(180deg); box-shadow: 0 23px 70px rgba(0, 9, 39, .37); }
        100% { transform: perspective(1400px) rotateY(360deg); box-shadow: 0 23px 70px rgba(0, 9, 39, .37); }
    }
    @keyframes flipBack {
        from { transform: perspective(1400px) rotateY(88deg); }
        to { transform: perspective(1400px) rotateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = os.path.join(BASE_DIR, "model")


@st.cache_resource
def load_model_artifacts():
    paths = (os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
             os.path.join(MODEL_DIR, "classifier.pkl"))
    if not all(os.path.isfile(path) for path in paths):
        return None, None
    return joblib.load(paths[0]), joblib.load(paths[1])


CATEGORY_META = {
    "Account": ("🔐", "#2563eb"),
    "Billing": ("💳", "#7c3aed"),
    "Fraud": ("🚨", "#dc2626"),
    "General Inquiry": ("💬", "#059669"),
    "Technical": ("🛠️", "#d97706"),
}
SUGGESTED_RESPONSES = {
    "Account": "Please follow the credential update instructions sent to your registered email address.",
    "Billing": "Your request has been routed to our billing team. We will review your transaction details shortly.",
    "Fraud": "Your security flag has been escalated. Our risk management team is investigating this issue urgently.",
    "General Inquiry": "Thank you for reaching out! Please refer to our help center FAQs for quick assistance.",
    "Technical": "Our engineering team has received your ticket. Please try re-logging into the application.",
}
SAMPLES = {
    "Select a sample ticket": "",
    "🔐 Account — login issue": "I forgot my password and cannot log into my account dashboard.",
    "💳 Billing — double charge": "I was charged twice for my monthly subscription and need a refund.",
    "🚨 Fraud — suspicious activity": "There are unauthorized transactions on my card that I did not make.",
    "💬 General — product information": "Can you tell me what plans you offer and their pricing?",
    "🛠️ Technical — app crash": "The mobile app crashes every time I try to upload a document.",
}


def choose_sample():
    st.session_state.ticket_text = SAMPLES[st.session_state.sample_choice]


def fallback_info(category, error=None):
    return {
        "response": SUGGESTED_RESPONSES.get(category, "Thank you for contacting support. Our team will review your request shortly."),
        "success": False,
        "provider": "Template fallback",
        "model": "N/A",
        "error": error,
    }


if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = ""
if "card_side" not in st.session_state:
    st.session_state.card_side = "front"

try:
    vectorizer, model = load_model_artifacts()
    model_error = None
except Exception as exc:
    vectorizer, model = None, None
    model_error = str(exc)

groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
can_query_models = bool(groq_api_key)

st.markdown('<div class="page-kicker">SMART SUPPORT · INSTANT TRIAGE</div>', unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.error(f"Model artifacts could not be loaded. Run `python src/train.py` and restart. {model_error or ''}")
    st.stop()

side = st.session_state.card_side
card_key = "ticket_result_card" if side == "back" and st.session_state.get("last_result") else "ticket_front_card"
with st.container(key=card_key):
    with st.container(key="card_inside"):
        if card_key == "ticket_front_card":
            categories = "".join(f"<span>{icon} {html.escape(name)}</span>" for name, (icon, _) in CATEGORY_META.items())
            st.markdown(
                '<div class="title-plaque"><div class="title-bar">🎫 Customer Support Ticket Classifier</div>'
                f'<div class="title-sub"><div class="category-pills">{categories}</div></div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="input_panel"):
                st.markdown('<div class="field-label">📝 Describe the customer’s issue</div>', unsafe_allow_html=True)
                st.selectbox("Try a sample ticket", SAMPLES.keys(), key="sample_choice", on_change=choose_sample)
                ticket_description = st.text_area(
                    "Ticket description", key="ticket_text", height=170,
                    placeholder="Enter ticket description…\n\ne.g. I forgot my password and cannot log in.",
                    label_visibility="collapsed",
                )
                st.caption(f"{len(ticket_description)} characters · Be specific for a better classification")
                classify = st.button("🔍 Classify & Generate Response", type="primary",
                                     use_container_width=True, key="classify")
            if classify:
                if not ticket_description.strip():
                    st.warning("Please enter a ticket description first.")
                else:
                    try:
                        with st.spinner("Classifying ticket…"):
                            r = predict([ticket_description])[0]
                    except Exception as exc:
                        st.error(f"Classification failed: {exc}")
                    else:
                        if not can_query_models:
                            response_info = fallback_info(
                                r["predicted_category"],
                                "GROQ_API_KEY is not set in the .env file. Showing template response."
                            )
                        else:
                            try:
                                with st.spinner("Generating AI response with Groq…"):
                                    response_info = generate_suggested_response(
                                        ticket_text=ticket_description,
                                        predicted_category=r["predicted_category"],
                                        api_key=groq_api_key,
                                    )
                            except Exception as exc:
                                response_info = fallback_info(r["predicted_category"], str(exc))
                        st.session_state.last_result = {
                            "ticket_text": ticket_description,
                            "category": r["predicted_category"],
                            "probs": r["probabilities"],
                            "score": r["confidence"],
                            "response_info": response_info,
                        }
                        st.session_state.card_side = "back"
                        st.rerun()
        else:
            result = st.session_state.last_result
            category = result["category"]
            icon, color = CATEGORY_META.get(category, ("🏷️", "#64748b"))
            score = float(result["score"])
            resp_info = result["response_info"] or fallback_info(category)
            response_text = str(resp_info.get("response") or fallback_info(category)["response"])
            st.markdown(
                '<div class="title-plaque"><div class="title-bar">✨ Ticket analyzed</div>'
                '<div class="title-sub">Your classification and suggested customer reply are ready.</div></div>',
                unsafe_allow_html=True,
            )
            with st.container(key="result_panel"):
                st.markdown('<div class="mini-label">CLASSIFICATION</div>', unsafe_allow_html=True)
                a, b = st.columns(2, gap="medium")
                with a:
                    st.markdown(f'<div class="mini-label">PREDICTED CATEGORY</div><div class="result-value" '
                                f'style="color:{color}">{icon} {html.escape(str(category))}</div>', unsafe_allow_html=True)
                with b:
                    st.markdown(f'<div class="mini-label">CONFIDENCE</div>'
                                f'<div class="result-value">{score:.2f}%</div>', unsafe_allow_html=True)
                    st.progress(max(0.0, min(score / 100, 1.0)))
                    confidence_note = "High confidence" if score >= 85 else (
                        "Medium confidence" if score >= 60 else "Low confidence — manual review advised")
                    st.caption(confidence_note)
                st.divider()
                st.markdown('<div class="mini-label">CATEGORY PROBABILITIES</div>', unsafe_allow_html=True)
                prob_df = pd.DataFrame({"Category": list(result["probs"]),
                                        "Probability (%)": list(result["probs"].values())})
                st.bar_chart(prob_df.set_index("Category"), horizontal=True, height=210)
                st.divider()
                st.markdown('<div class="mini-label">SUGGESTED CUSTOMER RESPONSE</div>', unsafe_allow_html=True)
                st.markdown('<div class="response-panel"><div class="reply-text">'
                            + html.escape(response_text) + '</div></div>', unsafe_allow_html=True)
                source = resp_info.get("provider", "LLM") if resp_info.get("success") else "Template fallback"
                model_name = resp_info.get("model") or "N/A"
                st.caption(f"Source: {source}" + (f" · Model: {model_name}" if resp_info.get("success") else ""))
                if resp_info.get("error"):
                    st.warning(f"AI response note: {resp_info['error']}")
                st.code(response_text, language=None)
                st.caption("Use the copy icon in the box above to copy the reply.")
                if st.button("🔄 Regenerate response", use_container_width=True, key="regenerate"):
                    if not can_query_models:
                        st.info("GROQ_API_KEY is not set in the .env file.")
                    else:
                        try:
                            with st.spinner("Generating a fresh Groq reply…"):
                                new_response = generate_suggested_response(
                                    ticket_text=result["ticket_text"],
                                    predicted_category=category,
                                    api_key=groq_api_key,
                                )
                        except Exception as exc:
                            new_response = fallback_info(category, str(exc))
                        st.session_state.last_result["response_info"] = new_response
                        st.rerun()
                if st.button("← Back to ticket", use_container_width=True, key="back_to_ticket"):
                    st.session_state.card_side = "front"
                    st.rerun()
    
st.markdown('<div class="page-footer">Built for faster, more thoughtful customer support · Powered by Groq AI</div>',
            unsafe_allow_html=True)