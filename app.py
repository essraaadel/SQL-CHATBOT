import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import re
import os
import json
from dotenv import load_dotenv

load_dotenv()

# ------------------- CONFIG -------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")
FEW_SHOTS_FILE = "fewshots.json"
TOP_K_EXAMPLES = 3  # How many similar examples to inject into the prompt

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Postgres DB :bar_chart:")

# ------------------- DATABASE -------------------
@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

def get_schema():
    engine = get_db_engine()
    inspector_query = text("""
        SELECT table_name, column_name 
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    schema_string = ""
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = None
            for row in result:
                table_name, column_name = row
                if table_name != current_table:
                    if current_table is not None:
                        schema_string += "\n"
                    schema_string += f"Table: {table_name}\n"
                    current_table = table_name
                schema_string += f"  - {column_name}\n"
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return ""
    return schema_string

# ------------------- LLM INITIALIZATION -------------------
@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

llm = get_llm()

# ------------------- FEW-SHOT FILE HELPERS -------------------
def load_few_shots() -> list[dict]:
    """Load all examples from fewshots.json."""
    if not os.path.exists(FEW_SHOTS_FILE):
        return []
    with open(FEW_SHOTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_few_shots(examples: list[dict]):
    """Save all examples to fewshots.json and clear the retriever cache."""
    with open(FEW_SHOTS_FILE, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    build_few_shot_retriever.clear()  # Force FAISS index to rebuild

def add_few_shot(question: str, sql: str):
    examples = load_few_shots()
    examples.append({"naturalQuestion": question, "sqlQuery": sql})
    save_few_shots(examples)

def delete_few_shot(index: int):
    examples = load_few_shots()
    if 0 <= index < len(examples):
        examples.pop(index)
        save_few_shots(examples)

# ------------------- SEMANTIC RETRIEVER -------------------
@st.cache_resource
def build_few_shot_retriever():
    """
    Build a FAISS vector store from fewshots.json.
    Cached so it only rebuilds when save_few_shots() calls .clear().
    Returns None if there are no examples.
    """
    examples = load_few_shots()
    if not examples:
        return None

    docs = [
        Document(
            page_content=ex["naturalQuestion"],
            metadata={"sql": ex["sqlQuery"]}
        )
        for ex in examples
        if ex.get("naturalQuestion") and ex.get("sqlQuery")
    ]
    if not docs:
        return None

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_EXAMPLES})

def get_dynamic_few_shots(user_question: str) -> str:
    """
    Retrieve the TOP_K_EXAMPLES most semantically similar Q→SQL pairs
    and format them for injection into the prompt.
    Falls back to static (all examples) if retriever unavailable.
    """
    retriever = build_few_shot_retriever()

    # Fallback: if no retriever (no examples or error), return empty string
    if retriever is None:
        return ""

    try:
        results = retriever.invoke(user_question)
    except Exception as e:
        st.warning(f"Retriever error (falling back to no examples): {e}")
        return ""

    if not results:
        return ""

    lines = []
    for i, doc in enumerate(results, 1):
        lines.append(f"  Example {i}:")
        lines.append(f"    Question : {doc.page_content}")
        lines.append(f"    SQL      : {doc.metadata['sql']}")

    return "\n".join(lines)

# ------------------- SQL HELPERS -------------------
def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()

# ------------------- SQL GENERATION (improved prompt) -------------------
def generate_sql_query(user_question: str, schema: str) -> str:
    few_shots_block = get_dynamic_few_shots(user_question)

    few_shots_section = ""
    if few_shots_block:
        few_shots_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIMILAR REFERENCE EXAMPLES
(Use these as style/pattern guidance only — adapt fully to the current question and schema)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{few_shots_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    prompt = f"""
You are an expert PostgreSQL Data Analyst. Your only output must be a valid SQL query — nothing else.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATABASE SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{schema}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES (never break these)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Tables were created via pandas — ALL table and column names are case-sensitive.
2. ALWAYS wrap every table name and column name in double quotes (e.g., "Customer", "CustomerId").
3. When joining tables that share column names, ALWAYS use table aliases to avoid ambiguity.
4. NEVER include Markdown fences, comments, or explanatory text — return only the raw SQL.
5. PostgreSQL type casting:
   - Dates   → "ColumnName"::timestamp
   - Numbers → "ColumnName"::numeric
{few_shots_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a PostgreSQL query that answers this question:
Question: {user_question}
SQL:
"""
    try:
        response = llm.invoke(prompt)
        return clean_sql(response)
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return ""

# ------------------- NATURAL LANGUAGE RESPONSE -------------------
def get_natural_language_response(question: str, data_df: pd.DataFrame) -> str:
    try:
        limited_data = data_df.head(15).to_markdown(index=False)
    except Exception:
        limited_data = data_df.head(15).to_string(index=False)

    prompt = f"""
You are a professional Data Analyst. Answer the user's question directly and concisely based solely on the data below.
Include specific numbers from the data. Keep a professional tone.

User Question: {question}

Query Results (up to 15 rows):
{limited_data}
"""
    try:
        return llm.invoke(prompt)
    except Exception as e:
        st.error(f"Error generating natural language response: {e}")
        return "Error generating response."

# ------------------- SIDEBAR: FEW-SHOT MANAGER UI -------------------
def render_few_shot_manager():
    st.sidebar.title("📚 Few-Shot Example Manager")
    st.sidebar.caption(
        f"Top **{TOP_K_EXAMPLES}** semantically similar examples are auto-selected per question."
    )

    examples = load_few_shots()

    # ---- ADD NEW EXAMPLE ----
    with st.sidebar.expander("➕ Add New Example", expanded=False):
        new_q = st.text_area("Natural Language Question", key="new_q", height=80)
        new_sql = st.text_area("SQL Query", key="new_sql", height=120)
        if st.button("Save Example", key="save_btn"):
            if new_q.strip() and new_sql.strip():
                add_few_shot(new_q.strip(), new_sql.strip())
                st.success("Example saved and index updated!")
                st.rerun()
            else:
                st.warning("Both fields are required.")

    st.sidebar.divider()

    # ---- VIEW / DELETE EXISTING EXAMPLES ----
    if not examples:
        st.sidebar.info("No examples yet. Add one above.")
        return

    st.sidebar.markdown(f"**{len(examples)} example(s) stored:**")

    for i, ex in enumerate(examples):
        with st.sidebar.expander(f"#{i+1}  {ex.get('naturalQuestion', '')[:55]}…", expanded=False):
            st.markdown("**Question:**")
            st.write(ex.get("naturalQuestion", ""))
            st.markdown("**SQL:**")
            st.code(ex.get("sqlQuery", ""), language="sql")
            if st.button(f"🗑️ Delete", key=f"del_{i}"):
                delete_few_shot(i)
                st.success("Deleted and index updated.")
                st.rerun()

# ------------------- MAIN APP -------------------
if __name__ == "__main__":
    render_few_shot_manager()

    schema = get_schema()
    if not schema:
        st.stop()

    user_question = st.text_input("Ask a question about the database:")

    if st.button("Get Answer") and user_question:
        with st.spinner("Retrieving similar examples & generating SQL…"):
            sql_query = generate_sql_query(user_question, schema)

        st.subheader("Generated SQL")
        st.code(sql_query, language="sql")

        clean_query_check = sql_query.lower().strip()
        allowed_starts = ("select", "with")

        result_df = pd.DataFrame()
        if not clean_query_check.startswith(allowed_starts):
            st.warning("LLM did not generate a valid SELECT or WITH query.")
        else:
            try:
                engine = get_db_engine()
                with engine.connect() as conn:
                    result_df = pd.read_sql(text(sql_query), conn)
                st.subheader("Query Results")
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error executing SQL: {e}")

        if not result_df.empty:
            with st.spinner("Generating natural language answer…"):
                answer = get_natural_language_response(user_question, result_df)
            st.subheader("Answer")
            st.markdown(answer)

            st.divider()
            st.markdown("**Was this query correct?**")
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("✅ Save as Example"):
                    add_few_shot(user_question, sql_query)
                    st.success("Saved! Future similar questions will use this as a reference.")
                    st.rerun()
