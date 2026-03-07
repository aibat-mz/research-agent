"""
Corpus Synthesizer
------------------
Performs corpus-level reasoning across all papers using structured metadata.
Used for questions like:
- What methodologies are used across these papers?
- What limitations appear across the papers?
"""

def synthesize_corpus(paper_metadata_list, query, llm):
    """
    Generate a synthesis across all papers based on their metadata.

    Parameters
    ----------
    paper_metadata_list : list[PaperMetadata]
    query : str
        User question.
    llm : LangChain chat model

    Returns
    -------
    str
        Synthesized answer.
    """

    summary_blocks = []

    for pm in paper_metadata_list:
        findings = "; ".join(pm.key_findings) if pm.key_findings else "Not extracted"
        limitations = "; ".join(pm.limitations) if pm.limitations else "Not extracted"

        block = f"""
Paper: {pm.title}
Methodology: {pm.methodology or "Not extracted"}
Key Findings: {findings}
Limitations: {limitations}
"""
        summary_blocks.append(block)

    summaries = "\n\n".join(summary_blocks)

    prompt = f"""
You are analyzing a collection of research papers.

Here are structured summaries of the papers:

{summaries}

User question:
{query}

Answer the question using only the information from these papers.
Provide a concise synthesis and reference the relevant papers when appropriate.
"""

    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)