from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# --- Core LangChain Imports ---
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize our local LLM using the new dedicated package
local_llm = OllamaLLM(model="phi3")

# 2. Load our local Vector DB using the new dedicated embeddings package
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("../artifacts/faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

def retrieve_node(state):
    """Fetches documents from FAISS based on the user's question."""
    print("---NODE: RETRIEVING DOCUMENTS---")
    question = state["question"]
    docs = retriever.invoke(question)
    
    # Extract text from LangChain Document objects
    doc_texts = [d.page_content for d in docs]
    return {"documents": doc_texts, "question": question}

def grade_documents_node(state):
    """Evaluates if the retrieved documents actually contain the answer."""
    print("---NODE: GRADING DOCUMENTS---")
    question = state["question"]
    documents = state.get("documents", [])
    
    # Define the LangChain Prompt for grading
    prompt = PromptTemplate(
        template="""You are a medical data grader. Look at the retrieved document and the user question.
        Does the document contain information relevant to answering the question?
        Document: {context}
        Question: {question}
        Reply ONLY with 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    
    # LangChain Expression Language (LCEL) chain
    grader_chain = prompt | local_llm | StrOutputParser()
    
    filtered_docs = []
    for doc in documents:
        # Ask the LLM to grade each document
        score = grader_chain.invoke({"question": question, "context": doc})
        if "yes" in score.lower():
            filtered_docs.append(doc)
            
    return {"documents": filtered_docs, "question": question}

def generate_node(state):
    """Generates the final answer using the filtered documents."""
    print("---NODE: GENERATING ANSWER---")
    question = state["question"]
    documents = state.get("documents", [])
    
    # Standard RAG Prompt with strict guardrails
    prompt = PromptTemplate(
        template="""You are a helpful healthcare assistant. Answer the question based ONLY on the following context. 
        If the context does not contain the answer, say "I cannot find this information in the official hospital policies."
        
        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=["context", "question"],
    )
    
    generate_chain = prompt | local_llm | StrOutputParser()
    
    # Combine docs into one string
    context_str = "\n\n".join(documents)
    generation = generate_chain.invoke({"context": context_str, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}