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

st.set_page_config(page_title="Universal AI Summarizer", page_icon="🧠", layout="wide")
st.title("🧠 Universal AI Summarizer")
st.caption("Summarize anything — story, paper, skit, PDF, or docx — all free and automatic.")

# ---- Load summarizer model ----
@st.cache_resource
def load_model():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_model()

# ---- Helper functions ----
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

def summarize_text(text, length="medium"):
    if not text.strip():
        return "⚠️ No valid content found."

    # Adjust max/min length dynamically
    if length == "short":
        max_len, min_len = 80, 30
    elif length == "long":
        max_len, min_len = 220, 100
    else:
        max_len, min_len = 150, 60

    try:
        if len(text.split()) > 800:
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer_lsa = Summarizer()
            summary_sentences = summarizer_lsa(parser.document, 6)
            return " ".join([str(s) for s in summary_sentences])
        else:
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
    for line in summary.split("\n"):
        text_object.textLine(line)
    c.drawText(text_object)
    c.save()
    buffer.seek(0)
    return buffer

# ---- Sidebar ----
st.sidebar.header("⚙️ Options")
summary_size = st.sidebar.radio("Summary Length", ["short", "medium", "long"], index=1)
fetch_web = st.sidebar.checkbox("🔍 Enhance with Wikipedia context", value=True)
output_format = st.sidebar.multiselect("📤 Export Formats", ["PDF", "TXT", "DOCX"], default=["TXT"])

# ---- Input section ----
uploaded_file = st.file_uploader("📂 Upload a file (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
text_input = st.text_area("✍️ Or paste your content here", height=200)
generate = st.button("✨ Generate Summary")

# ---- Main logic ----
if generate:
    with st.spinner("Generating summary... please wait ⏳"):
        content = ""
        if uploaded_file:
            if uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = extract_text_from_docx(uploaded_file)
            elif uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode("utf-8")
        else:
            content = text_input

        if fetch_web and content:
            web_data = web_fetch(content.split(".")[0][:15])
            if web_data:
                content += "\n\n" + web_data

        summary = summarize_text(content, length=summary_size)
        st.subheader("🪄 Generated Summary:")
        st.write(summary)

        # ---- Download Options ----
        if summary.strip():
            if "TXT" in output_format:
                txt_bytes = summary.encode("utf-8")
                st.download_button("⬇️ Download as TXT", data=txt_bytes, file_name="summary.txt")

            if "DOCX" in output_format:
                doc = Document()
                doc.add_paragraph(summary)
                buf = BytesIO()
                doc.save(buf)
                buf.seek(0)
                st.download_button("⬇️ Download as DOCX", data=buf, file_name="summary.docx")

            if "PDF" in output_format:
                pdf_buf = export_pdf(summary)
                st.download_button("⬇️ Download as PDF", data=pdf_buf, file_name="summary.pdf", mime="application/pdf")

st.markdown("---")
st.caption("🧠 Built with Streamlit, Hugging Face Transformers, and Sumy — 100% free, local, and open-source.")
