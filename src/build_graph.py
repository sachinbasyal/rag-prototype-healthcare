from langgraph.graph import StateGraph, END
from graph_state import GraphState
from nodes import retrieve_node, grade_documents_node, generate_node, rewrite_query_node
from edges import decide_to_generate

def compile_healthcare_agent():
    # 1. Initialize the State Graph
    workflow = StateGraph(GraphState)

    # 2. Define the Nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("generate", generate_node)

    # 3. Define the Flow (Edges)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional Edge (The Guardrail): Use the logic from edges.py to decide the next step
    workflow.add_conditional_edges(
        "grade_documents", 
        decide_to_generate, 
        {
            "generate": "generate", 
            "rewrite": "rewrite",
            "end": END              
        }
    )
    
    # THE CYCLE: After rewriting, loop back to retrieve again!
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    # 4. Compile the application
    app = workflow.compile()
    return app