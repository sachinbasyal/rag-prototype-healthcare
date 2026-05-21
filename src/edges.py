from langgraph.graph import END

def decide_to_generate(state):
    """
    Router logic: Determines where to go after grading documents.
    This acts as our strict hallucination guardrail.
    """
    filtered_documents = state.get("documents", [])
    
    if not filtered_documents:
        # If all docs were graded "no" (irrelevant), we don't have enough context.
        # We route to "end" to safely abstain from answering.
        print("---EDGE: ALL DOCS IRRELEVANT. ROUTING TO END---")
        return "end"
    else:
        # If we have valid documents, proceed to generation.
        print("---EDGE: RELEVANT DOCS FOUND. ROUTING TO GENERATION---")
        return "generate"