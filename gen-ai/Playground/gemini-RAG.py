import streamlit as st
import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List

# Define a custom embedding function using Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Download the PDF from the specified URL and save it to the given path
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# Load the PDF file and extract text from each page
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Split the text into chunks based on double newlines
def split_text(text):
    return [i for i in re.split('\n\n', text) if i.strip()]

# Create a Chroma database with the given documents
def create_chroma_db(documents: List[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db, name

# Load an existing Chroma collection
def load_chroma_collection(path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    return chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

# Retrieve the most relevant passages based on the query
def get_relevant_passage(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

# Construct a prompt for the generation model based on the query and retrieved data
def make_rag_prompt(query: str, relevant_passage: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and conversational tone.
QUESTION: '{query}'
PASSAGE: '{escaped_passage}'

ANSWER:
"""
    return prompt

# Generate an answer using the Gemini Pro API
def generate_answer(prompt: str):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    result = model.generate_content(prompt)
    return result.text

# Streamlit UI
st.title("PDF Document Analyzer")
st.markdown("""
    Upload a PDF document, enter a query, and receive an answer based on the document content.
""")

# File uploader for PDF files
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# Create directory for database if it doesn't exist
db_folder = "chroma_db"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

if uploaded_file is not None:
    # Save uploaded PDF to a temporary file
    temp_pdf_path = os.path.join(db_folder, uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract text from the uploaded PDF
    pdf_text = load_pdf(temp_pdf_path)
    chunked_text = split_text(pdf_text)

    # Specify the path and collection name for Chroma database
    db_name = "rag_experiment"
    db_path = os.path.join(os.getcwd(), db_folder)
    db, db_name = create_chroma_db(chunked_text, db_path, db_name)

    st.success(f"PDF {uploaded_file.name} processed and indexed successfully!")

    # Query input
    query = st.text_input("Enter your query:")

    if st.button("Get Answer"):
        if query:
            db = load_chroma_collection(db_path, db_name)
            relevant_text = get_relevant_passage(query, db, n_results=1)
            if relevant_text:
                final_prompt = make_rag_prompt(query, "".join(relevant_text))
                answer = generate_answer(final_prompt)
                st.write("Generated Answer:")
                st.write(answer)
            else:
                st.warning("No relevant information found for the given query.")
