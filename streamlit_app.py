import os
import streamlit as st
import requests

from src.config import settings

st.set_page_config(
    page_title="Agentic AI RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .score-badge-high {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge-med {
        background-color: #FEF08A;
        color: #713F12;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .score-badge-low {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .chunk-box {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        font-size: 0.88rem;
    }
</style>
""", unsafe_allow_html=True)


SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup? (Refusal Test)"
]


# Initialize session state for messages and active query
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    api_mode = st.radio(
        "Execution Engine",
        ["Direct LangGraph Engine", "FastAPI Backend (localhost:8000)"],
        index=0
    )

    st.markdown("---")
    st.subheader("📚 Source Document")
    st.caption("Agentic AI eBook (`data/Ebook-Agentic-AI.pdf`)")
    st.caption(f"Pinecone Index: `{settings.pinecone_index_name}`")
    st.caption(f"Embedding: `{settings.openai_embedding_model}`")
    st.caption(f"LLM: `{settings.openai_model}`")

    st.markdown("---")
    st.subheader("💡 Benchmark Sample Queries")
    st.caption("Click any query to run an instant grounded test:")
    for sample in SAMPLE_QUERIES:
        clean_sample = sample.replace(" (Refusal Test)", "")
        if st.button(sample, key=f"btn_{sample}"):
            st.session_state.pending_query = clean_sample
            st.rerun()

    st.markdown("---")
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# Header
st.markdown('<div class="main-title">🤖 Agentic AI RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Strictly grounded question-answering powered by <b>LangGraph</b>, '
    '<b>Pinecone</b>, and <b>OpenAI</b>. Answers are derived strictly from the Agentic AI eBook.</div>',
    unsafe_allow_html=True
)


def query_rag(question: str, mode: str):
    """Query either directly through LangGraph or via FastAPI backend."""
    if "FastAPI" in mode:
        api_url = "http://localhost:8000/chat"
        resp = requests.post(api_url, json={"query": question}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer": data["final_answer"],
            "context": data["retrieved_context"],
            "score": data["confidence_score"],
            "sources": data.get("sources", [])
        }
    else:
        from src.graph import build_rag_graph
        graph = build_rag_graph()
        initial_state = {
            "question": question,
            "context": [],
            "answer": "",
            "score": 0.0,
            "sources": []
        }
        res = graph.invoke(initial_state)
        return {
            "answer": res.get("answer", ""),
            "context": res.get("context", []),
            "score": res.get("score", 0.0),
            "sources": res.get("sources", [])
        }


# Render conversation history
col_chat, col_inspect = st.columns([1.6, 1.0])

with col_chat:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "score" in msg:
                score = msg["score"]
                if score >= 0.8:
                    badge_class = "score-badge-high"
                elif score >= 0.5:
                    badge_class = "score-badge-med"
                else:
                    badge_class = "score-badge-low"

                st.markdown(
                    f'<span class="{badge_class}">Confidence Score: {score:.2f}</span>',
                    unsafe_allow_html=True
                )


# Side inspection panel for retrieved context
with col_inspect:
    st.subheader("🔍 Context Inspection")
    if st.session_state.messages:
        # Get the latest assistant message
        last_asst = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
        if last_asst and last_asst.get("context"):
            st.caption(f"Retrieved {len(last_asst['context'])} context chunks from Pinecone:")
            sources = last_asst.get("sources", [])
            for idx, chunk in enumerate(last_asst["context"]):
                src_info = sources[idx] if idx < len(sources) else {}
                page_label = f"Page {src_info.get('page')}" if src_info.get("page") else f"Chunk #{idx+1}"
                score_label = f" (Score: {src_info.get('score')})" if src_info.get("score") is not None else ""
                with st.expander(f"📄 {page_label}{score_label}", expanded=(idx == 0)):
                    st.write(chunk)
        elif last_asst:
            st.info("No context chunks retrieved for the latest query.")
    else:
        st.info("Submit a question to inspect retrieved context chunks and relevance scores here.")


# Handle pending query from sample buttons or chat input
prompt = st.chat_input("Ask a question about the Agentic AI eBook...")
active_query = prompt or st.session_state.pending_query

if active_query:
    st.session_state.pending_query = None

    # User message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with col_chat:
        with st.chat_message("user"):
            st.markdown(active_query)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Retrieving chunks and reasoning strictly with LangGraph..."):
                try:
                    result = query_rag(active_query, api_mode)
                    answer = result["answer"]
                    score = result["score"]
                    context = result["context"]
                    sources = result.get("sources", [])

                    st.markdown(answer)

                    if score >= 0.8:
                        badge_class = "score-badge-high"
                    elif score >= 0.5:
                        badge_class = "score-badge-med"
                    else:
                        badge_class = "score-badge-low"

                    st.markdown(
                        f'<span class="{badge_class}">Confidence Score: {score:.2f}</span>',
                        unsafe_allow_html=True
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "score": score,
                        "context": context,
                        "sources": sources
                    })
                    st.rerun()

                except Exception as ex:
                    st.error(f"Error executing query: {ex}")
