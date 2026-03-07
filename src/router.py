"""
Query Router
------------
Decides which reasoning path should handle a user query.
"""

def route_query(query: str) -> str:
    q = query.lower()

    # Cross-paper reasoning
    if "across" in q:
        print("[router] route → corpus")
        return "corpus"

    # Research gap questions
    if "gap" in q or "future work" in q:
        print("[router] route → gap")
        return "gap"

    print("[router] route → agent")
    return "agent"