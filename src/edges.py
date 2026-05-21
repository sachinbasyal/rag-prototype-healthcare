from langgraph.graph import END

def decide_to_generate(state):
    """
    Router logic: Determines where to go after grading documents.
    This acts as our strict hallucination guardrail.
    """
    filtered_documents = state.get("documents", [])
    retries = state.get("retries", 0)
    
    if not filtered_documents:
        # If we have no good documents, but we haven't retried too many times...
        if retries < 2:
            print(f"---EDGE: DOCS IRRELEVANT. ROUTING TO REWRITE (Attempt {retries + 1})---")
            return "rewrite"
        else:
            # We tried rewriting twice and still failed. Time to safely abstain.
            print("---EDGE: MAX RETRIES REACHED. SAFELY ABSTAINING---")
            return "end"
    else:
        print("---EDGE: RELEVANT DOCS FOUND. ROUTING TO GENERATION---")
        return "generate"