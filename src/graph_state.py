from typing import List, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our CRAG pipeline.
    """
    question: str
    documents: List[str]
    generation: str
    retries: int