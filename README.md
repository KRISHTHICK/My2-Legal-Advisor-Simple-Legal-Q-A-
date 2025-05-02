# My2-Legal-Advisor-Simple-Legal-Q-A-
GenAI

Here's a **simple end-to-end project** called **"A2A Legal Advisor"**, built with:

- **Streamlit** (frontend)
- **Python** (backend logic)
- **Ollama** (local LLM for question answering)
- **A2A (Ask to Answer)**: A pattern where user questions are answered via LLM over extracted data

---

### 🔹 Project: A2A Legal Advisor (Simple Legal Q&A on uploaded PDF)

**Functionality**:
1. User uploads a legal PDF document (e.g., terms of service, contracts).
2. System extracts text.
3. User types questions (e.g., "What are the refund policies?")
4. LLM (Ollama + LLaMA3) answers using extracted data.

---

### 📁 Folder Structure
```
a2a-legal-advisor/
│
├── app.py                  # Streamlit app
├── requirements.txt        # Package list
└── sample.pdf              # Example legal doc (optional)
```

---

### ✅ `requirements.txt`
```txt
streamlit
langchain
pdfplumber
ollama
```

---

### 🧠 `app.py`
```python
import streamlit as st
import pdfplumber
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import tempfile

# --- Streamlit UI ---
st.title("📚 A2A Legal Advisor")
st.caption("Upload a legal document and ask questions.")

uploaded_file = st.file_uploader("Upload Legal PDF", type="pdf")
question = st.text_input("Ask a legal question based on the uploaded document:")

# --- Text Extraction ---
def extract_text(file):
    full_text = ""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text

# --- Vector Store ---
def make_qa_chain(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(docs, embedding)
    retriever = vectordb.as_retriever()
    llm = Ollama(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- Logic Execution ---
if uploaded_file:
    st.success("PDF Uploaded Successfully")
    with st.spinner("Processing..."):
        content = extract_text(uploaded_file)
        qa_chain = make_qa_chain(content)

    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.markdown(f"**Answer:** {answer}")
else:
    st.info("Upload a legal document to begin.")

```

---

### 🚀 How to Run (No venv required)

In **VS Code Terminal**:

```bash
# Clone or move your folder into GitHub repo if needed
# Then run the app
streamlit run app.py
```

---

### 🧾 Example Questions:
- What is the cancellation policy?
- Who holds liability in case of a breach?
- Are there any refund terms?

---
Here are the additional files you'll need to make your **A2A Legal Advisor** project GitHub-ready and runnable in VS Code:

---

### 📄 `README.md`
```markdown
# 📚 A2A Legal Advisor

An AI-powered assistant that helps users understand legal documents by answering questions using LLMs.

### 🧠 Features

- Upload any legal PDF (terms, agreements, contracts)
- Ask natural language questions about the document
- Answers are generated using local LLM (LLaMA3 via Ollama)
- No internet or API keys needed

---

### 🚀 Getting Started

#### 1. Install Requirements
```bash
pip install -r requirements.txt
```

#### 2. Run Locally
```bash
streamlit run app.py
```

---

### 📦 Tech Stack

- Python
- Streamlit
- LangChain
- Ollama (`llama3`)
- PDFPlumber (for extracting text from PDFs)

---

### 📝 Example Usage

1. Upload a legal PDF
2. Ask:
   - _"What are the refund terms?"_
   - _"Who is liable in case of a dispute?"_
3. Receive clear and summarized answers

---

### ⚙️ Notes

- Ollama must be installed and running.
- You can use other models by editing the `Ollama(model="...")` line in `app.py`.

---

### 📄 License

MIT License
```

---

### ⚙️ `.gitignore`
```gitignore
# Ignore temp files
*.pyc
__pycache__/
*.DS_Store
*.log
*.tmp

# Ignore local PDF uploads
*.pdf

# VS Code settings
.vscode/
```

---

You can now push this project to GitHub and run it using:

```bash
streamlit run app.py
```
