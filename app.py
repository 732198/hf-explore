import os

import streamlit as st
from transformers import pipeline

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up GEMINI_API_KEY from .env for the stretch step
except ImportError:
    pass

MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
MAX_CHARS = 1000

LABEL_STYLE = {
    "positive": ("🟢", "Positive"),
    "negative": ("🔴", "Negative"),
    "neutral": ("⚪", "Neutral"),
}


@st.cache_resource
def load_classifier():
    """Load the sentiment model once and reuse it across reruns."""
    return pipeline("text-classification", model=MODEL)


st.title("Financial Headline Sentiment")
st.caption(f"Runs locally with {MODEL}")

# Fail fast: the app is useless if the model can't load.
try:
    classifier = load_classifier()
except OSError as e:
    st.error(
        "Couldn't download or open the model. The first run needs an internet "
        f"connection to fetch it from Hugging Face. Details: {e}"
    )
    st.stop()

headline = st.text_input(
    "Paste a financial headline",
    placeholder="e.g. Shares jumped 18% after the company raised guidance",
)

if st.button("Analyze"):
    text = headline.strip()

    # Check 1: empty input
    if not text:
        st.warning("Type or paste a headline first.")
        st.stop()

    # Check 2: overly long input (the model only reads about 512 tokens)
    if len(text) > MAX_CHARS:
        st.warning(f"Keep it under {MAX_CHARS} characters. Try a single headline.")
        st.stop()

    # Check 3: the model call itself can fail
    try:
        result = classifier(text)[0]
    except Exception as e:
        st.error(f"The model couldn't classify that text. Details: {e}")
        st.stop()

    icon, label = LABEL_STYLE.get(result["label"], ("❔", result["label"].title()))
    st.subheader(f"{icon} {label}")
    st.metric("Confidence", f"{result['score']:.1%}")
    st.progress(result["score"])

    st.session_state["last"] = (text, label, result["score"])

# Stretch: hand the local model's result to Gemini for an open-ended explanation.
if "last" in st.session_state and os.getenv("GEMINI_API_KEY"):
    if st.button("Explain with Gemini"):
        text, label, score = st.session_state["last"]
        try:
            from google import genai

            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            prompt = (
                f'A financial sentiment model labeled this headline "{label}" '
                f"with {score:.0%} confidence:\n\n{text}\n\n"
                "In 2-3 sentences, explain why an investor might read it that way, "
                "and note anything that could make the label wrong."
            )
            response = client.models.generate_content(
                model="gemini-3.8-flash", contents=prompt
            )
            st.info(response.text)
        except Exception as e:
            st.error(f"Gemini couldn't explain this one. Details: {e}")
