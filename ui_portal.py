#------------------ui_portal.py
#--------------All UI Related Operations in One File

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import base64
from pathlib import Path
import streamlit as st


def load_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOADER_IMG = load_image_base64("assets/infinity_loader.png")


# ---------- HEADER -----------------
def show_header():
    st.markdown("<h1>💬 Open ChatBot Assistant</h1>", unsafe_allow_html=True)

# ---------- SIDEBAR UPLOAD ----------
def upload_sidebar():
    st.sidebar.header("Upload Jira Dataset")
    return st.sidebar.file_uploader(
        "Upload file here", 
        type=["csv", "xlsx","docx", "pdf", "txt"])

# ---------- CHAT INPUT ----------
def chat_input():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("chat_input", clear_on_submit=True):
        query = st.text_input("Ask a question about Jira tickets 🌞")
        send = st.form_submit_button("Search")

    return query, send

def show_results(results):
    if not results:
        st.warning("No matching tickets found.")
        return

    df = pd.DataFrame(results)

    # Normalize status (lowercase + trim)
    df["Status"] = df["Status"].str.strip().str.lower()

    # ---------------- STATUS GROUPS ----------------
    open_statuses = [
        "open",
        "in progress",
        "under investigation",
        "waiting for approval",
        "awaiting implementation",
        "pending",
        "under review"
    ]

    closed_statuses = [
        "closed",
        "canceled",
        "cancelled",
        "resolved",
        "completed"
    ]

    # ---------------- FILTER DATA ----------------
    open_df = df[df["Status"].isin(open_statuses)]
    closed_df = df[df["Status"].isin(closed_statuses)]

    # ---------------- OPEN TICKETS ----------------
    if not open_df.empty:
        st.subheader("🟢 OPEN TICKETS")
        st.dataframe(open_df, use_container_width=True)

    # ---------------- CLOSED TICKETS ----------------
    if not closed_df.empty:
        st.subheader("⚫ CLOSED TICKETS")
        st.dataframe(closed_df, use_container_width=True)

    # ---------------- FALLBACK ----------------
    if open_df.empty and closed_df.empty:
        st.info("Tickets found, but status does not match configured categories.")
 # ---------------- Beautify waiting time ----------------
def show_searching_overlay():
    html = f"""
    <style>
    .overlay {{
        position: fixed;
        inset: 0;
        background: #000;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .loader {{
         width: 550px; #####-------------------------To increase and descreas size of infinty  
        animation: pulse 1.8s ease-in-out infinite;
    }}
    @keyframes pulse {{
        0% {{ transform: scale(0.95); opacity: 0.7; }}
        50% {{ transform: scale(1.05); opacity: 1; }}
        100% {{ transform: scale(0.95); opacity: 0.7; }}
    }}
    </style>

    <div class="overlay">
        <img class="loader" src="data:image/png;base64,{LOADER_IMG}" />
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


