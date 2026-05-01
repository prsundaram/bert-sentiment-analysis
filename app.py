import streamlit as st
from transformers import pipeline
import torch

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------
# Custom CSS (Beautiful UI)
# ---------------------------
st.markdown("""
<style>
.main {
    background-color: #0f172a;
}
.title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model (Cached)
# ---------------------------
@st.cache_resource
def load_model():
    model_path = "psundara/bert-base-uncased-sentiment-hugging-face"
    
    classifier = pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    
    return classifier

classifier = load_model()

# ---------------------------
# Emoji Mapping
# ---------------------------
emoji_map = {
    "sadness": "😢",
    "joy": "😊",
    "fear": "😨",
    "surprise": "😲",
    "anger": "😡",
    "love": "❤️"
}

color_map = {
    "sadness": "#1e293b",
    "joy": "#065f46",
    "fear": "#7c2d12",
    "surprise": "#1e40af",
    "anger": "#7f1d1d",
    "love": "#831843"
}

# ---------------------------
# UI Header
# ---------------------------
st.markdown('<div class="title">🤖 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Fine-Tuned BERT + Hugging Face 🚀</div>', unsafe_allow_html=True)

# ---------------------------
# Input Box
# ---------------------------
user_input = st.text_area("💬 Enter your text", height=120, placeholder="Type something like: I feel amazing today!")

# ---------------------------
# Predict Button
# ---------------------------
if st.button("Analyze Sentiment 🚀"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        with st.spinner("Analyzing..."):

            result = classifier(user_input)[0]
            label = result["label"]
            score = result["score"]

            emoji = emoji_map.get(label, "")
            color = color_map.get(label, "#334155")

            st.markdown(f"""
            <div class="result-box" style="background-color:{color}; color:white;">
                {emoji} {label.upper()} <br><br>
                Confidence: {score:.2f}
            </div>
            """, unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Transformers & Streamlit</center>",
    unsafe_allow_html=True
)
