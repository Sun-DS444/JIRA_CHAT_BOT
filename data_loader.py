#                   data_loader.py
#  File Upload → Cleaning → Chunking → Embeddings → DB Insert
import pandas as pd
import json
import numpy as np
import psycopg2
import traceback
import streamlit as st
from docx import Document
from psycopg2.extras import execute_values

# ---------------- DOCX READER ----------------
def read_docx(uploaded_file):
    doc = Document(uploaded_file)
    return "\n".join(
        p.text for p in doc.paragraphs if p.text.strip()
    )

# ---------------- DATABASE CONFIG -----------------
DB_HOST = "127.0.0.1"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "Secure@123"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="disable"
    )

# ---------------- UNIVERSAL CLEANER ----------------
def clean_value(v):
    if isinstance(v, pd.Series):
        return clean_value(v.iloc[0])

    if isinstance(v, (list, dict)):
        try:
            return json.dumps(v)
        except Exception:
            return str(v)

    if isinstance(v, np.generic):
        return v.item()

    if pd.isna(v):
        return None

    return v

# ---------------- SAFE TEXT JOIN ----------------
def safe_text(*vals):
    return "\n".join(
        str(v).strip() for v in vals
        if v is not None and str(v).strip() and str(v).lower() != "nan"
    )
# ---------------- FIELD MAPPING ----------------
FIELD_MAP = {
    'Issue Type': 'issue_type',
    'Issue key': 'issue_key',
    'Summary': 'title',
    'Description': 'description',
    'Assignee': 'assignee',
    'Reporter': 'reporter',
    'Status': 'status',
    'Resolution': 'resolution',
    'Created': 'created_date',
    'Updated': 'updated_date',
    'Due date': 'resolved_date',
    'Custom field (Component)': 'affected_component',
    'Custom field (Type of Problems)': 'type_of_problems',
    'Custom field (Root causes)': 'root_causes',
    'Custom field (Immediate fix remark)': 'immediate_fix_remark',
    'Custom field (Immediate fix target date)': 'immediate_fix_target_date',
    'Custom field (Immediate fix closure date)': 'immediate_fix_closure_date',
    'Custom field (Permanent fix required)': 'permanent_fix_required',
    'Custom field (Permanent fix remark)': 'permanent_fix_remark',
    'Custom field (Issue Occurrence)': 'issue_occurrence',
    'Custom field (Issue Nature)': 'issue_nature',
    'Custom field (Issue origin)': 'issue_origin',
    'Custom field (Issue Classification)': 'issue_classification',
    'Custom field (Type of Resolution)': 'type_of_resolution',
    'Custom field (Root causes)':"root_cause",
    'Custom field (Request Source)' :"request_source"
}

# ---------------- DB COLUMNS ----------------
DB_COLUMNS = [
    "ticketid","issue_type","title","description","embedding","status","project_name",
    "issue_key","assignee","reporter","issue_id","created_date","updated_date",
    "priority","resolution","resolved_date","affected_area","root_causes",
    "immediate_fix_remark","immediate_fix_target_date",
    "immediate_fix_closure_date","permanent_fix_required","permanent_fix_remark",
    "issue_occurrence","issue_nature","issue_origin","issue_classification",
    "type_of_resolution","root_cause","permanent_fix_closure_date",
    "permanent_fix_target_date","request_source","type_of_problems",
    "chunk_type","chunk_text"
]

# ---------------- PROCESS FILE ----------------
def process_uploaded_file(uploaded_file, model):
    try:
        # -------- Read File (SAFE ROUTING) --------
        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file)

        elif filename.endswith(".docx"):
         text = read_docx(uploaded_file)

        doc_key = (
            "DOC_" +
            uploaded_file.name
            .replace(".docx", "")
            .replace(" ", "_")
            .upper()
        )

        df = pd.DataFrame([{
            "issue_key": doc_key,        #  REQUIRED
            "title": uploaded_file.name,
            "description": text,
            "status": "DOCUMENT"         # optional but useful
        }])


        # -------- Normalize Columns --------
        df.columns = df.columns.str.strip()
        rename_dict = {c: FIELD_MAP[c] for c in df.columns if c in FIELD_MAP}
        df.rename(columns=rename_dict, inplace=True)
        df.columns = df.columns.str.lower()

        # -------- Mandatory Columns --------
        for col in ["ticketid", "title", "description", "status", "issue_key"]:
            if col not in df.columns:
                df[col] = None

        # -------- DB Connection --------
        conn = get_connection()
        cur = conn.cursor()

        records = []

        # -------- CHUNKING + EMBEDDING --------
        for _, row in df.iterrows():
            chunk_map = {
                "summary": safe_text(row.get("title")),
                "description": safe_text(row.get("description"), row.get("type_of_problems")),
                "steps": safe_text(row.get("immediate_fix_remark")),
                "resolution": safe_text(row.get("resolution"), row.get("type_of_resolution")),
                "root_cause": safe_text(row.get("root_causes"), row.get("root_cause")),
                "dependency_reason": safe_text(
                    row.get("permanent_fix_required"),
                    row.get("permanent_fix_remark"),
                    row.get("permanent_fix_target_date"),
                    row.get("immediate_fix_target_date")
                )
            }

            for chunk_type, chunk_text in chunk_map.items():
                if not chunk_text:
                    continue

                embedding = model.encode(chunk_text).tolist()

                cleaned = {col: clean_value(row.get(col)) for col in df.columns}
                cleaned["embedding"] = json.dumps(embedding)
                cleaned["chunk_type"] = chunk_type
                cleaned["chunk_text"] = chunk_text

                record = tuple(cleaned.get(col) for col in DB_COLUMNS)
                records.append(record)

        # -------- INSERT INTO DB --------
        col_sql = ",".join(DB_COLUMNS)
        insert_sql = f"""
            INSERT INTO jira_tickets ({col_sql})
            VALUES %s
            ON CONFLICT (issue_key, chunk_type)
            DO UPDATE SET
                embedding = EXCLUDED.embedding,
                chunk_text = EXCLUDED.chunk_text;
        """

        execute_values(cur, insert_sql, records)
        conn.commit()

        cur.close()
        conn.close()

        return True

    except Exception as e:
        st.error(f"Upload error: {e}")
        st.text(traceback.format_exc())
        return False