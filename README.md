# AI Summarizer

### 🔍 Summarize • Compare • Study — All in One Smart App

![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-brightgreen)

A **powerful Streamlit-based AI app** that can summarize *any type of content* — research papers, stories, skits, articles, movie plots, and even multiple documents at once.  
Built entirely using **free and open-source tools** — no paid APIs, no keys, just pure intelligence on CPU 💻.

---

## 🚀 Live Demo
👉 [Launch on Streamlit Cloud](https://summarymodel.streamlit.app/#universal-ai-summarizer-v4-pro) *(deployed)*

---

## 🌟 Features

| Feature | Description |
|----------|--------------|
| 🧠 **Universal Summarizer** | Works for text, PDFs, DOCX, stories, research papers, etc. |
| 📚 **Multi-Document Support** | Upload multiple PDFs/TXT/DOCX files and summarize them together. |
| 🌍 **Web-Assisted Context** | Fetches info from Wikipedia to improve summaries. |
| 🌐 **Multi-Language Support** | Auto-detects and translates any language before summarizing. |
| 📏 **Adjustable Summary Length** | Choose between short, medium, or long summaries. |
| 📊 **Summary Evaluation** | Compare your human-written and AI-generated summaries. |
| 🧮 **AI Study Notes Generator** | Converts summaries into Q&A-style study notes. |
| 🧩 **Text Insights** | Get tone analysis and keyword extraction instantly. |
| 📤 **Download Options** | Export summaries as TXT, DOCX, or PDF. |
| ⚡ **Runs 100% Free** | No OpenAI keys or paid APIs required — runs locally or on Streamlit Cloud. |

---

## 🧩 Tech Stack

- **Frontend/UI** → Streamlit  
- **Summarization Model** → `distilbart-cnn-12-6` (via Hugging Face Transformers)  
- **Text Preprocessing** → Sumy + BeautifulSoup  
- **Language Translation** → Googletrans  
- **File Handling** → PyPDF2, python-docx  
- **Export Tools** → ReportLab  
- **Evaluation** → difflib (text similarity)  
- **Notes/Q&A Generation** → custom NLP logic  

---

---
### © 2025 Aditya — All Rights Reserved  
Unauthorized use, reproduction, or modification of this software is prohibited.
