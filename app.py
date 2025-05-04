import streamlit as st
import pdfplumber
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore  # Import the correct docstore

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
    # Split the document into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    
    # Use Ollama to create embeddings
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = embedding.embed_documents([doc.page_content for doc in docs])
    
    # Convert embeddings to numpy array (FAISS requires float32)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
    index.add(embeddings)
    
    # Create an InMemoryDocstore to store the documents
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
    
    # Map index to document ID
    index_to_docstore_id = {i: i for i in range(len(docs))}
    
    # Create FAISS vector store
    vectorstore = FAISS(index=index, embedding_function=embedding, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    
    # Create a retriever
    retriever = vectorstore.as_retriever()
    
    # Use the LLM (Ollama)
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
