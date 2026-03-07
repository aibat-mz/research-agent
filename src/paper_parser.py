"""
src/paper_parser.py
-------------------
Parses research PDFs into structured metadata using an LLM.

WHY STRUCTURED EXTRACTION INSTEAD OF RAW TEXT?
------------------------------------------------
Storing raw text is simple, but it makes downstream tasks hard:
  - Comparing papers requires knowing WHERE the methodology lives.
  - Gap analysis needs to see key_findings from every paper side-by-side.
  - Fuzzy title search works better when the title is its own field.

By asking the LLM to fill a fixed schema once (at index time), every later
operation (compare, summarise, gap-analyse) can just read Python attributes
instead of re-searching the raw text.
"""

import json
import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    """Structured representation of a research paper's key metadata.

    Fields are intentionally coarse-grained (e.g., 'methodology' is a
    1-2 sentence description) so the LLM can fill them reliably even when
    the PDF formatting is messy.
    """

    title: str = Field(description="Full title of the paper")
    authors: list[str] = Field(default_factory=list, description="List of author names")
    year: Optional[str] = Field(default=None, description="Publication year if found")
    abstract: Optional[str] = Field(default=None, description="Full abstract text")
    methodology: Optional[str] = Field(
        default=None,
        description="1-2 sentence description of the research methodology",
    )
    methodology_tags: list[str] = Field(
    default_factory=list,
    description="Keywords describing the methodology category"
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="3-5 main findings from the paper",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations acknowledged by the authors",
    )
    file_path: str = Field(description="Absolute or relative path to the source PDF")


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """Extract the following factual information only from the provided text. If a field is not explicitly stated, infer it from the context when possible.
If it truly cannot be inferred, return null.

Fields to extract:
- title: Full paper title
- authors: List of author names
- year: Publication year (if found)
- abstract: Full abstract text
- methodology: Brief description of research methodology (1-2 sentences)
- methodology_tags: List of 2-4 keywords describing the methodology
- key_findings: List of 3-5 main findings
- limitations: List of limitations mentioned by authors

Excerpt from the research paper:
{text}

Return valid JSON only."""


def parse_paper(file_path: str, llm) -> PaperMetadata:
    """Load a single PDF and extract structured metadata using an LLM.

    Steps
    -----
    1. Load all pages with PyPDFLoader.
    2. Extract text from the first 3 pages and from sections containing results, discussion or conclusion.
    3. Ask the LLM to fill the extraction schema (JSON response).
    4. Parse the JSON into a PaperMetadata object.
    5. On any failure, fall back to filename-derived title with empty fields.

    Parameters
    ----------
    file_path : str
        Path to the PDF file.
    llm :
        Any LangChain chat model (e.g., ChatOpenAI).

    Returns
    -------
    PaperMetadata
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # HEAD (начало статьи)
    head_text = "\n".join(p.page_content for p in pages[:3])

    # Keywords for sections likely containing findings/limitations
    keywords = ["results and discussion", "results", "discussion", "conclusion"]

    full_text = "\n".join(p.page_content for p in pages)
    lower_text = full_text.lower()
    tail_parts = []

    for kw in keywords:
         idx = lower_text.rfind(kw)
         if idx != -1:
             section = full_text[idx:idx + 1000]
             tail_parts.append(section)

    tail_parts = list(dict.fromkeys(tail_parts))      
    tail_text = "\n\n".join(tail_parts)

    excerpt = head_text + "\n\n" + tail_text
    excerpt = " ".join(excerpt.replace("\n", " ").split())
    prompt = _EXTRACTION_PROMPT.format(text=excerpt)

    try:
        response = llm.invoke(prompt)
        # Handle both string responses and AIMessage objects
        raw = response.content if hasattr(response, "content") else str(response)

        # Strip markdown code fences if the LLM wraps JSON in ```json ... ```
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()

        data = json.loads(raw)

        print("\n[DEBUG] LLM JSON:\n", raw, "\n")
        print(f"[paper_parser] Extracted: {data.get('title')}")

        return PaperMetadata(
            title=data.get("title") or Path(file_path).stem,
            authors=data.get("authors") or [],
            year=str(data.get("year")) if data.get("year") else None,
            abstract=data.get("abstract"),
            methodology=data.get("methodology"),
            methodology_tags=data.get("methodology_tags") or [],
            key_findings=data.get("key_findings") or [],
            limitations=data.get("limitations") or [],
            file_path=file_path,
        )

    except Exception as exc:
        # Graceful degradation: use the filename as title, leave everything else blank.
        # This means the paper can still be searched even if LLM extraction failed.
        print(f"[paper_parser] Warning: could not extract metadata from '{file_path}': {exc}")
        return PaperMetadata(
            title=Path(file_path).stem,
            file_path=file_path,
        )


def parse_all_papers(papers_dir: str, llm) -> list[PaperMetadata]:
    """Parse every PDF found in *papers_dir* and return a list of PaperMetadata.

    Parameters
    ----------
    papers_dir : str
        Directory that contains *.pdf files (non-recursive).
    llm :
        Any LangChain chat model.

    Returns
    -------
    list[PaperMetadata]
        One entry per successfully located PDF.  Empty list if no PDFs found.
    """
    papers_path = Path(papers_dir)
    pdf_files = sorted(papers_path.glob("*.pdf"))

    if not pdf_files:
        print(f"[paper_parser] No PDF files found in '{papers_dir}'.")
        return []

    results: list[PaperMetadata] = []
    for pdf in pdf_files:
        print(f"[paper_parser] Parsing: {pdf.name}")
        metadata = parse_paper(str(pdf), llm)
        results.append(metadata)

    print(f"[paper_parser] Parsed {len(results)} paper(s).")
    return results
