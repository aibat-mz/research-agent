"""
src/agent.py
------------
Wires together the tools and LLM into a LangChain ReAct agent.

WHAT IS THE REACT LOOP?
------------------------
ReAct (Reason + Act) is a prompting strategy where the LLM alternates between:

  Thought     – the model reasons about what to do next
  Action      – the model picks a tool and writes an input for it
  Observation – the tool runs and its output is appended to the prompt
  … repeat until …
  Final Answer – the model decides it has enough information

Example:
  Thought : I need to find papers about transformers. I'll search.
  Action  : search_papers
  Action Input: transformer self-attention mechanism
  Observation : [Result 1] Paper: "Attention Is All You Need" …
  Thought : I found a relevant paper. Now I'll summarize it.
  Action  : summarize_paper
  Action Input: Attention Is All You Need
  Observation : Title: Attention Is All You Need …
  Final Answer: The paper "Attention Is All You Need" introduced …

HOW THE AGENT SEES THE TOOLS
------------------------------
The agent receives a text-formatted list of tool names and descriptions in its
system prompt.  It never sees function signatures or source code.  This is why
precise tool descriptions are critical: they are the agent's entire API docs.

WHY verbose=True IS IMPORTANT FOR LEARNING
-------------------------------------------
With verbose=True LangChain prints every Thought / Action / Observation to
stdout.  You can watch the agent's reasoning unfold in real time.  This is
invaluable for understanding why the agent chose a particular tool, and for
debugging when it makes the wrong choice.

THE DIFFERENCE BETWEEN AN AGENT AND A SIMPLE LLM CALL
--------------------------------------------------------
A simple LLM call is a single prompt → single response.  The LLM cannot fetch
new information mid-response.  An agent can:
  - Decide which tool to call based on intermediate results
  - Retry with a different query if the first search returns nothing
  - Chain multiple tool calls (search → summarize → compare)
  - Stop early if the first observation already answers the question

WHAT "ZERO SHOT" MEANS
------------------------
ZERO_SHOT_REACT_DESCRIPTION means the agent needs zero examples (shots) in its
prompt.  It figures out when and how to use each tool purely from the tool
description.  This keeps the prompt short and avoids the need to curate
few-shot examples for every new tool.
"""

from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain_community.vectorstores import FAISS

from src.tools.compare_tool import create_compare_tool
from src.tools.search_tool import create_search_tool
from src.tools.summary_tool import create_summary_tool

# System prompt injected as the agent's persona and behavioural guidelines.
# The prefix is prepended to the auto-generated ReAct prompt that lists tools.
_AGENT_PREFIX = """You are a rigorous AI research assistant.

The following research papers are available:

{paper_list}

Users may refer to expapers either by title or by number (e.g., "paper 1", "paper 2").

You MUST use tools to answer factual questions about the papers.
Never answer from your own knowledge.
If information is not found via tools, say so.

Workflow rules:
- If you do not know which paper contains the answer → use search_papers.
- If the user mentions a specific paper → use summarize_paper.
- If comparing papers → use compare_papers.
- Always cite paper titles in the final answer.
- Do not fabricate content.

Think step by step."""


def create_research_agent(
    vector_store: FAISS,
    paper_metadata: list,
    llm,
) -> AgentExecutor:
    """Build and return a fully configured ReAct research agent.

    Parameters
    ----------
    vector_store : FAISS
        Populated FAISS index (from paper_indexer.index_papers).
    paper_metadata : list[PaperMetadata]
        List of parsed paper metadata objects.
    llm :
        Any LangChain chat model (e.g., ChatOpenAI).

    Returns
    -------
    AgentExecutor
        The runnable agent.  Call agent.invoke({"input": query})["output"] to use it.
    """
    # Build a title → metadata dict for the summary and compare tools
    paper_metadata_dict = {pm.title: pm for pm in paper_metadata}

    # Build a numbered list of papers for the agent prompt
    paper_list = "\n".join(
    f"Paper {i+1}: {pm.title}" for i, pm in enumerate(paper_metadata)
    )

    # Instantiate each tool
    search_tool = create_search_tool(vector_store)
    summary_tool = create_summary_tool(paper_metadata_dict, llm)
    compare_tool = create_compare_tool(paper_metadata_dict, llm)

    tools = [search_tool, summary_tool, compare_tool]

    prefix = _AGENT_PREFIX.format(paper_list=paper_list)
    # initialize_agent wraps the LLM + tools in a ReAct prompt loop.
    # ZERO_SHOT_REACT_DESCRIPTION: no few-shot examples, tool selection driven
    # entirely by the description strings we provided above.
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,           # print Thought/Action/Observation to stdout
        handle_parsing_errors=True,  # recover gracefully from malformed tool calls
        agent_kwargs={"prefix": prefix},
        max_iterations=10,       # safety cap to prevent infinite loops
        early_stopping_method="generate"
    )

    return agent


def run_agent(query: str, agent: AgentExecutor) -> str:
    """Run a single query through the research agent.

    Parameters
    ----------
    query : str
        The user's question or instruction.
    agent : AgentExecutor
        The agent built by :func:`create_research_agent`.

    Returns
    -------
    str
        The agent's final answer.
    """
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    result = agent.invoke({"input": query})["output"]

    print(f"\n{'='*60}")
    print("Final Answer:")
    print(result)
    print(f"{'='*60}\n")

    return result
