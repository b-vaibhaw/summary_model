import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.tokenizers import Tokenizer
from PyPDF2 import PdfReader
from docx import Document
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langdetect import detect
from googletrans import Translator
from collections import Counter
import re
import textwrap
import difflib

# ---------------------- Streamlit Setup ----------------------
st.set_page_config(page_title="Universal AI Summarizer v4.1 Turbo", page_icon="🧠", layout="wide")
st.title("🧠 Universal AI Summarizer — v4.1 Turbo Edition")
st.caption("Summarize, Evaluate, and Study — multi-language, multi-file, and fully free ⚡")

# ---------------------- Model ----------------------
@st.cache_resource
def load_model(fast=True):
    """Load summarization model based on speed preference."""
    if fast:
        model_name = "csebuetnlp/mT5_multilingual_XLSum"  # smaller, multilingual
    else:
        model_name = "sshleifer/distilbart-cnn-12-6"  # high-quality
    return pipeline("summarization", model=model_name)

# Sidebar option for mode
st.sidebar.header("⚙️ Options")
fast_mode = st.sidebar.checkbox("⚡ Turbo Mode (faster, lighter summaries)", value=True)
summarizer = load_model(fast_mode)
translator = Translator()

# ---------------------- Helper Functions ----------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs]).strip()

def web_fetch(query):
    try:
        url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")
            return " ".join([p.text for p in soup.select("p")[:5]])
    except Exception:
        pass
    return ""

def summarize_text(text, length="medium", fast=True):
    """Hybrid summarizer: LSA for fast, transformer for deep summarization."""
    if not text.strip():
        return "⚠️ No valid content found."

    if len(text.split()) > 1500 or fast:
        # Use LSA for fast mode or large inputs
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer_lsa = Summarizer()
        sentence_count = {"short": 3, "medium": 5, "long": 8}.get(length, 5)
        summary_sentences = summarizer_lsa(parser.document, sentence_count)
        return " ".join([str(s) for s in summary_sentences])
    else:
        # Use transformer for detailed summaries
        max_len, min_len = {"short": (80, 30), "medium": (150, 60), "long": (250, 100)}[length]
        try:
            result = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
            return result[0]["summary_text"]
        except Exception as e:
            return f"⚠️ Error generating summary: {e}"

def export_pdf(summary):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text_object = c.beginText(40, height - 60)
    text_object.setFont("Helvetica", 12)
    for line in textwrap.wrap(summary, 90):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    return buffer

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_keywords(text, top_n=10):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    common = Counter(words).most_common(top_n)
    return [w for w, _ in common]

def analyze_tone(text):
    positive = ["good", "great", "excellent", "positive", "happy", "success"]
    negative = ["bad", "sad", "poor", "negative", "failure"]
    pos_count = sum(text.lower().count(w) for w in positive)
    neg_count = sum(text.lower().count(w) for w in negative)
    if pos_count > neg_count:
        return "🙂 Positive"
    elif neg_count > pos_count:
        return "☹️ Negative"
    else:
        return "😐 Neutral"

def compare_summaries(human, ai):
    seq = difflib.SequenceMatcher(None, human, ai)
    ratio = seq.ratio() * 100
    diff = difflib.ndiff(human.split(), ai.split())
    highlighted = []
    for token in diff:
        if token.startswith("-"):
            highlighted.append(f"❌ **{token[2:]}**")
        elif token.startswith("+"):
            highlighted.append(f"✅ **{token[2:]}**")
    return ratio, " ".join(highlighted)

def generate_notes(summary):
    lines = summary.split(". ")
    qna = []
    for line in lines[:8]:
        words = line.split()
        if len(words) > 6:
            q = f"Q: What is the main point about '{' '.join(words[:4])}'?"
            a = f"A: {line.strip()}."
            qna.append((q, a))
    return qna

# ---------------------- Sidebar Extended ----------------------
summary_size = st.sidebar.radio("Summary Length", ["short", "medium", "long"], index=1)
fetch_web = st.sidebar.checkbox("🌍 Add Wikipedia context", value=True)
output_format = st.sidebar.multiselect("📤 Export Formats", ["PDF", "TXT", "DOCX"], default=["TXT"])
multi_doc = st.sidebar.checkbox("📚 Multi-document summarization", value=False)

# ---------------------- Input Section ----------------------
if multi_doc:
    uploaded_files = st.file_uploader("📂 Upload multiple files (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
else:
    uploaded_files = [st.file_uploader("📂 Upload a single file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])]

text_input = st.text_area("✍️ Or Paste Your Content Here", height=200)
human_summary_input = st.text_area("🧑‍💻 (Optional) Paste Your Human-Written Summary for Comparison", height=150)
generate = st.button("✨ Generate Summary and Insights")

# ---------------------- Main Logic ----------------------
if generate:
    with st.spinner("Processing... please wait ⏳"):
        content = ""

        # Combine multiple docs
        for file in uploaded_files:
            if file is not None:
                if file.type == "application/pdf":
                    content += extract_text_from_pdf(file) + "\n"
                elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    content += extract_text_from_docx(file) + "\n"
                elif file.type == "text/plain":
                    content += file.read().decode("utf-8") + "\n"

        if text_input:
            content += text_input

        content = clean_text(content)
        if not content:
            st.warning("⚠️ Please upload or paste content.")
            st.stop()

        # Detect language
        try:
            lang = detect(content)
            st.info(f"🌐 Detected language: {lang.upper()}")
            if lang != "en":
                content = translator.translate(content, src=lang, dest="en").text
                st.success("✅ Translated to English for summarization.")
        except Exception:
            lang = "en"

        # Fetch Wikipedia info
        if fetch_web:
            web_data = web_fetch(content.split(".")[0][:15])
            if web_data:
                content += "\n\n" + web_data

        # ⚡ Adaptive Summarization Logic
        if fast_mode or len(content) < 2500:
            final_summary = summarize_text(content, length=summary_size, fast=fast_mode)
        else:
            chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
            summaries = [summarize_text(chunk, length=summary_size, fast=fast_mode) for chunk in chunks]
            final_summary = " ".join(summaries)

        # Translate back if needed
        if lang != "en":
            final_summary = translator.translate(final_summary, src="en", dest=lang).text

        # ---------------------- Output ----------------------
        st.subheader("🪄 AI Summary:")
        st.write(final_summary)
        st.progress(min(len(final_summary) / len(content), 1.0))
        st.caption(f"📝 Summary length: {len(final_summary.split())} words")

        # ---------------------- Evaluation ----------------------
        if human_summary_input.strip():
            st.subheader("📊 Summary Evaluation")
            ratio, diff_text = compare_summaries(human_summary_input, final_summary)
            st.write(f"**Similarity Score:** {ratio:.2f}%")
            st.markdown(diff_text)

        # ---------------------- AI Notes ----------------------
        st.subheader("🧮 AI Study Notes (Auto Q&A)")
        notes = generate_notes(final_summary)
        for q, a in notes:
            st.markdown(f"**{q}**  \n{a}")

        # ---------------------- Insights ----------------------
        st.subheader("🔍 Text Insights")
        st.write(f"**Tone:** {analyze_tone(content)}")
        st.write(f"**Top Keywords:** {', '.join(extract_keywords(content))}")

        # ---------------------- Downloads ----------------------
        if final_summary.strip():
            if "TXT" in output_format:
                txt_bytes = final_summary.encode("utf-8")
                st.download_button("⬇️ Download as TXT", data=txt_bytes, file_name="summary.txt")

            if "DOCX" in output_format:
                doc = Document()
                doc.add_paragraph(final_summary)
                buf = BytesIO()
                doc.save(buf)
                buf.seek(0)
                st.download_button("⬇️ Download as DOCX", data=buf, file_name="summary.docx")

            if "PDF" in output_format:
                pdf_buf = export_pdf(final_summary)
                st.download_button("⬇️ Download as PDF", data=pdf_buf, file_name="summary.pdf", mime="application/pdf")

st.markdown("---")
st.caption("🧠 Built by Aditya — Universal AI Summarizer v4.1 Turbo | Hugging Face + Streamlit + Free APIs")
