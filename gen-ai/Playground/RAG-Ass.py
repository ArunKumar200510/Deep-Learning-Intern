import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from groq import Groq
import os

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="D:\Gen AI Intern\gen-ai\Playground\chroma_DB")

def create_or_get_collection(name):
    try:
        # Try to get the collection
        collection = chroma_client.get_collection(name=name)
    except Exception as e:
        # Collection does not exist, create it
        collection = chroma_client.create_collection(name=name)
    return collection

collection = create_or_get_collection(name='RAG-HDFC')

# Function to split text into chunks
def make_chunks(text):
    return text_splitter.split_text(text)

# Function to retrieve context based on a query
def get_context(ques, tot2):
    ques_emb = model.encode(ques).tolist()  # Convert to list
    DB_response = collection.query(
        query_embeddings=[ques_emb],
        n_results=3,
        include=['embeddings','metadatas']
    )

    if not DB_response['documents']:
        st.error("No matches found in the database response.")
        return ""

    cont = ""
    for doc in DB_response['documents']:
        try:
            cont += doc
        except (IndexError, ValueError) as e:
            st.error(f"Error accessing document: {e}")
    return cont

# Function to extract text from a PDF file
def extract_pdf(path):
    reader = PdfReader(path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

# Custom CSS for better visuals
st.markdown("""
    <style>
    .stSidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        background-color: #007bff;
        color: white;
    }
    .stTextInput>div>div>input {
        width: 100%;
        padding: 8px;
    }
    .spinner-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .spinner {
        border: 6px solid #f3f3f3;
        border-top: 6px solid #007bff;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 2s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .home-container {
        text-align: center;
        padding: 50px;
    }
    .home-container h1 {
        font-size: 3rem;
        color: #007bff;
    }
    .home-container p {
        font-size: 1.2rem;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation


if 'tot5' not in st.session_state:
    st.session_state.tot5 = []




if True:
    st.title("Document Analyzer")

    # Define default local path for PDF
    default_file_path = "C:/Users/rrak/Downloads/HDFC-Life-Sanchay-Plus-Life-Long-Income-Option-101N134V19-Policy-Document.pdf"

    # Check if file exists at the default path
    if not os.path.exists(default_file_path):
        st.error("Default file path does not exist. Please check the path.")
    else:
        extracted = extract_pdf(default_file_path)
        tot_chunks = make_chunks(extracted)
        st.session_state.tot5 = tot_chunks  
        st.write("Total chunks created:", len(tot_chunks))  

        tot_embeddings = model.encode(tot_chunks).tolist()  # Convert to list
        documents = tot_chunks
        ids = [f"{i}+1" for i, (vec, chunk) in enumerate(zip(tot_embeddings, tot_chunks))]
        try:
            print("Indexing started")
            collection.add(documents=documents, embeddings=tot_embeddings, ids=ids)
            st.success("Documents processed and indexed successfully!")
        except Exception as e:
            st.error(f"Error adding documents: {e}")

    query = st.text_input("Enter your query:")
    if st.button("Get Answer"):
        if query:
            context = get_context(query, st.session_state.tot5)  
            if context:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": f"Context: {context}, Analyse and understand the above context completely and answer the below query, Query: {query}",
                        }
                    ],
                    model="llama3-8b-8192",
                )
                response_text = chat_completion.choices[0].message.content
                st.write("Answer:")
                st.write(response_text)

    if st.button("Clear Database"):
        with st.spinner('Clearing database...'):
            try:
                chroma_client.delete_collection('NEW_COLLECTION')
                st.success("Database cleared successfully!")
            except Exception as e:
                st.warning(f"Error clearing database: {e}")
