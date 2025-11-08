
import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(page_title="OpenJiraBot", page_icon="⚡", layout="wide")

# ------------------------------------------------
# SESSION DEFAULTS
# ------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark Blue"

# ------------------------------------------------
# DATABASE CONFIG
# ------------------------------------------------
DB_HOST = "ep-bitter-brook-adky0jvt-pooler.c-2.us-east-1.aws.neon.tech"
DB_PORT = "5432"
DB_NAME = "Data science"
DB_USER = "neondb_owner"
DB_PASSWORD = "npg_CfIcFp7dV6eb"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD, sslmode="require"
    )

# ------------------------------------------------
# THEMES
# ------------------------------------------------
THEME_COLORS = {
    "Dark Blue": {
        "bg": "linear-gradient(120deg,#001f4d,#002b5c,#001a33)",
        "fg": "#ffffff",
        "accent": "#007bff",
        "sidebar": "#000e1a"
    },
    "Dark Black": {
        "bg": "linear-gradient(120deg,#000000,#040b1a,#0b0b28)",
        "fg": "#ffffff",
        "accent": "#00aaff",
        "sidebar": "#000000"
    }
}

# ------------------------------------------------
# SIDEBAR: THEME SWITCH
# ------------------------------------------------
st.sidebar.markdown("### Theme")
col1, col2 = st.sidebar.columns(2)
if col1.button("🔵", help="Dark Blue Theme"):
    st.session_state.theme = "Dark Blue"
if col2.button("⚫", help="Dark Black Theme"):
    st.session_state.theme = "Dark Black"

st.sidebar.markdown("### 📂 Upload Jira CSV/Excel (Admin Only)")
uploaded_file = st.sidebar.file_uploader("", type=["csv", "xlsx"], label_visibility="collapsed")

# ------------------------------------------------
# BACKGROUND ANIMATION (REALISTIC STAR-GAZING EFFECT)
# ------------------------------------------------
theme = st.session_state.theme
colors = THEME_COLORS[theme]
bg = colors["bg"]
fg = colors["fg"]
accent = colors["accent"]
sidebar_bg = colors["sidebar"]

animated_css = f"""
<style>
/* Animated star-gazing background */
@keyframes skyFlow {{
  0% {{ background-position: 0% 50%; }}
  50% {{ background-position: 100% 50%; }}
  100% {{ background-position: 0% 50%; }}
}}

@keyframes twinkle {{
  0%, 100% {{ opacity: 0.8; }}
  50% {{ opacity: 1; }}
}}

[data-testid="stAppViewContainer"] {{
    background: radial-gradient(ellipse at bottom, #020111 0%, #000000 100%), {bg};
    background-size: 200% 200%;
    animation: skyFlow 60s ease-in-out infinite;
    color: {fg};
    position: relative;
}}

[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
      radial-gradient(1px 1px at 10% 20%, rgba(255,255,255,0.8) 50%, transparent 50%),
      radial-gradient(1px 1px at 30% 80%, rgba(255,255,255,0.7) 50%, transparent 50%),
      radial-gradient(1px 1px at 70% 50%, rgba(255,255,255,0.9) 50%, transparent 50%),
      radial-gradient(1px 1px at 90% 30%, rgba(255,255,255,0.85) 50%, transparent 50%);
    animation: twinkle 4s infinite ease-in-out alternate;
    z-index: 0;
}}

[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
    color: {fg} !important;
    z-index: 10;
}}

.model-status {{
    position: fixed;
    top: 15px;
    left: 15px;
    background: rgba(0, 123, 255, 0.85);
    padding: 8px 16px;
    border-radius: 10px;
    font-weight: 600;
    color: white;
    box-shadow: 0 0 15px rgba(0,0,0,0.4);
    z-index: 999;
}}

h1, p, label {{
    color: {fg} !important;
}}

.stButton>button {{
    background: linear-gradient(135deg,{accent},#00b4d8);
    color: white;
    font-weight: 600;
    border-radius: 8px;
}}

input {{
    background-color: rgba(255,255,255,0.1);
    color: white;
    border-radius: 6px;
}}
#MainMenu, footer, header {{visibility: hidden;}}
</style>
<div class="model-status">✅ AI model ready!</div>
"""

st.markdown(animated_css, unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------
with st.spinner("Loading AI model..."):
    model = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------------------------------------
# FILE UPLOAD PROCESSING
# ------------------------------------------------
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        if 'Summary' in df.columns:
            df.rename(columns={'Summary': 'Title'}, inplace=True)
        if 'Description' not in df.columns:
            df['Description'] = ''
        if 'TicketID' not in df.columns:
            df['TicketID'] = df.index.astype(str)
        if 'Status' not in df.columns:
            df['Status'] = 'Unknown'

        df = df.drop_duplicates(subset=['TicketID'])
        texts = (df['Title'].fillna('') + ' ' + df['Description'].fillna('')).tolist()
        embeddings = model.encode(texts, show_progress_bar=False)

        conn = get_connection()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS jira_tickets (
                ticketid TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                embedding VECTOR(384),
                status TEXT
            );
        """)
        conn.commit()

        records = [
            (str(df.iloc[i]['TicketID']), str(df.iloc[i]['Title']), str(df.iloc[i]['Description']),
             str(df.iloc[i]['Status']), embeddings[i].tolist())
            for i in range(len(df))
        ]

        execute_values(cur, """
            INSERT INTO jira_tickets (ticketid,title,description,status,embedding)
            VALUES %s
            ON CONFLICT (ticketid) DO UPDATE SET
                title=EXCLUDED.title,
                description=EXCLUDED.description,
                status=EXCLUDED.status,
                embedding=EXCLUDED.embedding;
        """, records)

        conn.commit()
        cur.close()
        conn.close()
        st.sidebar.success("✅ Data stored successfully!")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

# ------------------------------------------------
# MAIN INTERFACE
# ------------------------------------------------
st.markdown('<h1>💬 OpenJiraBot Assistant</h1>', unsafe_allow_html=True)
st.write("What’s on the radar today, Buddy?")
st.markdown("---")

# ------------------------------------------------
# CHAT FUNCTION
# ------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Ask me about Jira tickets or type a question...", max_chars=1000)
    send = st.form_submit_button("Send")

if send and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # st.experimental_return()
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]
    try:
        conn = get_connection()
        cur = conn.cursor()
        query_embedding = model.encode([prompt])[0]
        vector_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"
        cur.execute(f"""
            SELECT ticketid,title,description,status
            FROM jira_tickets
            ORDER BY embedding <-> '{vector_str}'::vector
            LIMIT 5;
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()

        if results:
            df_display = pd.DataFrame(results, columns=["TicketID", "Title", "Description", "Status"])
            df_display["Description"] = df_display["Description"].astype(str).str.slice(0, 180) + "..."
            st.session_state.messages.append({"role": "assistant", "content": "Here are similar Jira tickets found."})
            st.markdown("**Assistant:** Here are similar Jira tickets found.")
            st.dataframe(df_display, use_container_width=True)
        else:
            st.session_state.messages.append({"role": "assistant", "content": "No related Jira tickets found."})
            st.markdown("**Assistant:** No related Jira tickets found.")
    except Exception as e:
        st.error(e)

st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
