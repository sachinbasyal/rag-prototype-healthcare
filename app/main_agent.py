import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from build_graph import compile_healthcare_agent

if __name__ == "__main__":
    print("Initializing Healthcare RAG Agent...\n")
    agent = compile_healthcare_agent()
    
    # Test Question 1: Should succeed
    test_question = "What is the policy for getting a refund on an appointment?"
    
    # Test Question 2: Should fail the grading and safely end (Guardrail test)
    # test_question = "What are the visiting hours for the pediatric ward?"
    
     # Test Question 3: Should succeed
    # test_question = "When should we pay copay?"
    
    print(f"USER QUERY: {test_question}\n")
    
    # Initialize the state
    initial_state = {"question": test_question, "retries": 0}
    
    # We will store the final state here as it streams
    final_state = None 
    
    # Run the graph
    for output in agent.stream(initial_state):
        for key, value in output.items():
            # Update final_state with the output of the most recent node
            final_state = value 
            
    print("\n---FINAL AGENT RESPONSE---")
    # Check if we successfully generated an answer
    if final_state and final_state.get("generation"):
         print(final_state["generation"])
    else:
         print("Agent Output: I abstained from answering because no relevant medical policies were retrieved.")