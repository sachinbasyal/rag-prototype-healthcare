import streamlit as st
import sys
import os

# Ensure the src/ directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from build_graph import compile_healthcare_agent

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Enterprise Healthcare RAG",
    page_icon="🏥",
    layout="centered"
)

# --- 2. CACHE THE AGENT ---
# We use @st.cache_resource so Streamlit doesn't reload the FAISS database 
# and local LLM into memory every time you click a button.
@st.cache_resource
def load_agent():
    return compile_healthcare_agent()

agent = load_agent()

# --- 3. UI HEADER ---
st.title("🏥 Enterprise Healthcare Assistant")
st.markdown("""
*A local, privacy-first Corrective RAG (CRAG) pipeline.* **Architecture:** LangGraph State Machine, LangChain 0.3, Local FAISS, Ollama (`phi3`).
""")
st.divider()

# --- 4. USER INPUT ---
query = st.text_input("Ask a question about hospital policies, billing, or scheduling:", placeholder="e.g., What is the refund policy?")

# --- 5. EXECUTION & VISUALIZING THE GRAPH ---
if st.button("Ask Agent") and query:
    
    initial_state = {"question": query, "retries": 0}
    final_state = None
    
    # st.status creates a beautiful expanding box to show the agent's thought process
    with st.status("Agent processing request...", expanded=True) as status:
        
        # Stream the LangGraph execution step-by-step
        for output in agent.stream(initial_state):
            for node_name, state_value in output.items():
                
                # Dynamically update the UI based on which node LangGraph is executing
                if node_name == "retrieve":
                    st.write("🔍 **Node:** Retrieving medical policies from FAISS...")
                elif node_name == "grade_documents":
                    st.write("⚖️ **Node:** Grading context to prevent hallucinations...")
                elif node_name == "generate":
                    st.write("✍️ **Node:** Generating grounded response...")
                
                final_state = state_value
                
        status.update(label="Workflow Complete!", state="complete", expanded=False)

    # --- 6. DISPLAY RESULTS ---
    st.subheader("Agent Response")
    
    # Check if the guardrail let a generation through
    if final_state and final_state.get("generation"):
         st.success(final_state["generation"])
    else:
         st.warning("⚠️ **Agent Abstained:** The retrieval guardrails blocked this response because no highly relevant medical policies were found in the database. This prevents hallucination.")

    # --- 7. UNDER THE HOOD ---
    st.write("")
    with st.expander("🛠️ View Agent Context (For Technical Review)"):
        st.markdown("These are the specific chunks that passed the `grade_documents` node and were used to ground the LLM:")
        if final_state and final_state.get("documents"):
            for i, doc in enumerate(final_state["documents"]):
                st.info(f"**Document {i+1}:**\n{doc}")
        else:
            st.error("No documents passed the grading phase.")