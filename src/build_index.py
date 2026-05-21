from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# THE LangChain 0.3 UPDATE:
from langchain_huggingface import HuggingFaceEmbeddings

def build_local_database():
    print("Initializing Healthcare Data Ingestion...")
    
    # 1. Mock Healthcare Data (In reality, you'd load PDFs/Markdown)
    mock_data = [
        "Policy D001: Refund Policy. Patients are eligible for a full refund if they cancel their appointment 48 hours in advance. No refunds for no-shows.",
        "Scheduling: To schedule an appointment with Dr. Basyal (Cardiology), patients must first secure a referral from their primary care provider.",
        "Disease Info: Type 2 Diabetes management requires routine A1C blood tests every 3 to 6 months. Dietary consultations are covered under standard insurance.",
        "Billing: Copays are due at the time of the visit. For billing disputes, contact the patient advocacy line at 555-0199."
    ]
    
    # Convert strings to LangChain Document objects
    docs = [Document(page_content=text) for text in mock_data]

    # 2. Chunking (Small chunks because we are using a small local LLM)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50
    )
    chunked_docs = text_splitter.split_documents(docs)

    # 3. Embeddings (CPU-friendly HuggingFace model)
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and Save FAISS Vector Store
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    vectorstore.save_local("../artifacts/faiss_index")
    print("Database built and saved successfully to /artifacts/faiss_index")

if __name__ == "__main__":
    build_local_database()