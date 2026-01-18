#                   search_engine.py
#     BM25 + Dense Embedding + RRF Hybrid Search Engine
#        (CHUNK-BASED VERSION)
import json
import numpy as np
import psycopg2
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize


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

# ---------------- TEXT PREPROCESS ----------------
def preprocess(text):
    if not text:
        return []
    return word_tokenize(text.lower())

# ---------------- COSINE SIMILARITY ----------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom != 0 else 0.0

# ---------------- RRF FUSION ----------------
def rrf_fusion(bm25_results, dense_results, k=60):
    scores = {}

    for rank, r in enumerate(bm25_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (k + rank + 1)

    for rank, r in enumerate(dense_results):
        scores[r["id"]] = scores.get(r["id"], 0) + 1 / (k + rank + 1)

    return sorted(scores, key=scores.get, reverse=True)

# ---------------- FETCH CHUNKS ----------------
def fetch_chunks(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT issue_key, chunk_type, chunk_text, embedding
        FROM jira_tickets
        WHERE chunk_text IS NOT NULL
    """)
    rows = cur.fetchall()
    cur.close()
    return rows

# ---------------- BM25 SEARCH ----------------
def bm25_search(query, rows, top_k=20):
    documents = [r[2] for r in rows]
    tokenized_docs = [preprocess(d) for d in documents]

    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = preprocess(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    results = []
    for idx, score in ranked:
        issue_key, chunk_type, text, _ = rows[idx]
        results.append({
            "id": f"{issue_key}-{chunk_type}",
            "issue_key": issue_key,
            "chunk_type": chunk_type,
            "text": text,
            "score": score
        })

    return results
# New---------------- DENSE SEARCH ----------------
def dense_search(query, model, rows):
    query_emb = model.encode(query).tolist()
    results = []

    for row in rows:
        text = row[1]
        emb = row[2]

        # skip empty or null embeddings
        if emb is None or emb == "":
            continue

        # convert only if emb is string
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except json.JSONDecodeError:
                continue  # skip bad data

        score = cosine_similarity(query_emb, emb)

        results.append({
            "text": text,
            "score": score
        })

    return results

# ---------------- HYBRID SEARCH ----------------
def hybrid_search(query, model, conn, top_k=10):
    rows = fetch_chunks(conn)
    if not rows:
        return []

    bm25_results = bm25_search(query, rows)
    dense_results = dense_search(query, model, rows)

    fused_ids = rrf_fusion(bm25_results, dense_results)

    all_results = {r["id"]: r for r in bm25_results + dense_results}

    final_results = []
    for fid in fused_ids:
        if fid in all_results:
            final_results.append(all_results[fid])
        if len(final_results) >= top_k:
            break

    return final_results

# ---------------- COUNT ----------------
def get_total_ticket_count():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM jira_tickets;")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count
# ---Add context builder and it is also brudge to LLM because its clean data and take only 5 to 10 chunks
def build_context(results, max_chunks=3):
    priority = {
        "steps": 1,
        "dependency_reason": 2,
        "resolution": 3,
        "root_cause": 4,
        "description": 5,
        "summary": 6
    }

    # sort by chunk priority + score
    sorted_results = sorted(
        results,
        key=lambda r: (priority.get(r["chunk_type"], 99), -r["score"])
    )

    context_blocks = []
    used = set()

    for r in sorted_results:
        key = (r["issue_key"], r["chunk_type"])
        if key in used:
            continue

        block = f"""
Ticket: {r['issue_key']}
Section: {r['chunk_type']}
Content:
{r['text']}
""".strip()

        context_blocks.append(block)
        used.add(key)

        if len(context_blocks) >= max_chunks:
            break

    return "\n\n---\n\n".join(context_blocks)

