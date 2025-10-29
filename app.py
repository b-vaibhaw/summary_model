import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

st.set_page_config(page_title="Universal Summarizer", page_icon="🧠", layout="wide")
st.title("🧠 Universal Content Summarizer")
st.write("Summarize any kind of text: story, skit, movie plot, or research paper — free and automatic!")

# Load model (cached to avoid reload)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def web_fetch(query):
    """Fetch relevant web content for context (if enabled)."""
    try:
        url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            paragraphs = " ".join([p.text for p in soup.select("p")[:5]])
            return paragraphs
    except Exception:
        return ""
    return ""

def summarize_text(text, max_words=130):
    """Use BART model; fallback to Sumy if text too long."""
    try:
        if len(text.split()) > 800:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer_lsa = Summarizer()
            summary_sentences = summarizer_lsa(parser.document, 5)
            return " ".join([str(s) for s in summary_sentences])
        else:
            result = summarizer(text, max_length=150, min_length=60, do_sample=False)
            return result[0]['summary_text']
    except Exception as e:
        return f"⚠️ Error: {e}"

with st.form("summarize_form"):
    user_input = st.text_area("✍️ Enter content here (story, paper, skit, etc.)", height=250)
    fetch_web = st.checkbox("🔍 Add info from web (Wikipedia)", value=True)
    submitted = st.form_submit_button("Generate Summary")

if submitted and user_input.strip():
    st.info("⏳ Generating summary... Please wait")
    
    content = user_input
    if fetch_web:
        web_data = web_fetch(user_input.split(".")[0][:10])
        if web_data:
            content += "\n\n" + web_data

    summary = summarize_text(content)
    st.subheader("🪄 Summary:")
    st.write(summary)
