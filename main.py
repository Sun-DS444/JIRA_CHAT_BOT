import streamlit as st
from sentence_transformers import SentenceTransformer
from data_loader import process_uploaded_file
from search_engine import (
    hybrid_search,
    get_connection,
    get_total_ticket_count,
    build_context
)
from llm_client import call_llm
from ui_portal import (
    show_header,
    upload_sidebar,
    chat_input,
    show_searching_overlay
)
from llm_client import ensure_ollama_running

# 🔥 start AI engine once when app loads
if "ollama_ready" not in st.session_state:
    with st.spinner("Starting AI engine..."):
        ensure_ollama_running()
        st.session_state.ollama_ready = True

# ---------------- PAGE CONFIG(Like Tab name,default page slide bar etc) ---------------------
st.set_page_config(page_title="OpenJiraBot", page_icon="⚡", layout="wide")

# ---------------- HEADER ----------------
show_header()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- SIDEBAR INFO ----------------
total = get_total_ticket_count()
st.sidebar.metric("Total Jira Tickets", total)

# ---------------- SIDEBAR UPLOAD ----------------
uploaded_file = upload_sidebar()
if uploaded_file:
    with st.spinner("Processing file and storing in database..."):
        success = process_uploaded_file(uploaded_file, model)
        if success:
            st.sidebar.success("File processed and stored successfully!")
        else:
            st.sidebar.error("Error while processing file.")

# ---------------- CHAT INPUT ----------------
query, send = chat_input()

# ---------------- SEARCH + LLM ANSWER ----------------
if send and query:

    # Show loading overlay
    overlay = st.empty()
    with overlay:
        show_searching_overlay()

    # 1️⃣ Search Jira knowledge (RAG retrieval)
    conn = get_connection()
    results = hybrid_search(query, model, conn)
    conn.close()

    # 2️⃣ Build clean context (background only)
    context = build_context(results)

    # 3️⃣ Call LLM (ONLY ONCE)
    answer = call_llm(context, query)

    # Remove loader
    overlay.empty()

    # 4️⃣ Show final answer (ChatGPT-style)
    st.subheader("Answer")
    st.write(answer)
