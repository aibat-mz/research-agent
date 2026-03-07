import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from src.agent import create_research_agent, run_agent
from src.paper_parser import parse_all_papers
from src.paper_indexer import index_papers
from src.gap_analyzer import analyze_gaps
from src.report_generator import generate_report

load_dotenv()

st.set_page_config(page_title="AI Research Agent", layout="wide")

st.title("📚 AI Research Agent")

papers_dir = "data/papers"

if "agent" not in st.session_state:

    st.info("Initializing research agent...")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )

    papers = parse_all_papers(papers_dir, llm)

    vector_store = index_papers(papers_dir)

    agent = create_research_agent(vector_store, papers, llm)

    st.session_state.agent = agent
    st.session_state.papers = papers
    st.session_state.llm = llm

    st.success("Agent ready.")

agent = st.session_state.agent

tab1, tab2 = st.tabs(["💬 Chat with Papers", "📄 Generate Report"])

# ---------------- CHAT ---------------- #

with tab1:

    st.subheader("Ask questions about the papers")

    user_input = st.chat_input("Ask something about the papers...")

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            response = run_agent(user_input, agent)
            st.write(response)

# ---------------- REPORT ---------------- #

with tab2:

    st.subheader("Generate Research Gap Report")

    if st.button("Generate Report"):

        with st.spinner("Analyzing papers..."):

            gaps = analyze_gaps(
                st.session_state.papers,
                st.session_state.llm
            )

            report = generate_report(
                paper_metadata_list=st.session_state.papers,
                gap_analysis=gaps,
                topic="Research Analysis",
            )

        st.markdown(report)

        st.download_button(
            label="Download Report",
            data=report,
            file_name="research_report.md"
        )