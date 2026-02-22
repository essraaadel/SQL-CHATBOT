<div align="center">

# 🗄️ SQL Chatbot
### Chat with your PostgreSQL database using natural language

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![Google Gemini](https://img.shields.io/badge/Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)

<br/>

![Demo](https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/assets/demo.gif)

> **Ask questions in plain English. Get SQL queries and clear answers — instantly.**

</div>

---

## ✨ Features

- 🔍 **Natural Language to SQL** — Type any question, get a precise PostgreSQL query powered by Gemini 2.5 Flash
- 🧠 **Semantic Few-Shot Retrieval** — Uses FAISS vector search to find the most relevant example Q→SQL pairs for each question, improving accuracy over time
- 📚 **Few-Shot Manager UI** — Add, view, and delete example pairs directly from the sidebar — no file editing needed
- 💾 **Self-Learning Feedback Loop** — Confirm a good query with one click and it's saved as a future reference example
- 📊 **Instant Results** — Query results displayed as an interactive table with a natural language summary
- 🔒 **Read-Only Safety** — Only `SELECT` and `WITH` queries are allowed to execute

---

## 🖥️ Demo

| Ask a question | See the SQL | Get the answer |
|---|---|---|
| *"Who are the top 5 customers by total spend?"* | Auto-generated PostgreSQL query | Natural language summary + data table |

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│  FAISS Vector Store  │  ◄── fewshots.json (your Q→SQL examples)
│  (Semantic Search)   │
└─────────┬───────────┘
          │  Top-K similar examples
          ▼
┌─────────────────────┐
│   Gemini 2.5 Flash  │  ◄── Schema + Rules + Examples + Question
│   (SQL Generation)  │
└─────────┬───────────┘
          │  Raw SQL
          ▼
┌─────────────────────┐
│    PostgreSQL DB     │
└─────────┬───────────┘
          │  Results DataFrame
          ▼
┌─────────────────────┐
│   Gemini 2.5 Flash  │  (Natural Language Summary)
└─────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A running PostgreSQL database
- A Google AI API key ([get one here](https://ai.google.dev))

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/essraaadel/SQL-CHATBOT.git
cd SQL-CHATBOT
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set up environment variables**

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
DB_URL=postgresql://username:password@host:port/database_name
```

**4. Run the app**
```bash
streamlit run app.py
# or on Windows if streamlit isn't in PATH:
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📦 Dependencies

```txt
streamlit
pandas
sqlalchemy
psycopg2-binary
langchain-google-genai
langchain-community
faiss-cpu
python-dotenv
tabulate
```

> Install all at once: `pip install -r requirements.txt`

---

## 📁 Project Structure

```
📦 sql-chatbot/
├── app.py              # Main Streamlit application
├── fewshots.json       # Q→SQL example pairs (auto-managed via UI)
├── requirements.txt    # Python dependencies
├── .env                # API keys & DB URL (never commit this!)
├── .gitignore          # Excludes .env and other sensitive files
└── README.md           # You are here
```

---

## 🧪 How Few-Shot Learning Works

The chatbot gets smarter the more you use it:

1. **Ask** a question → SQL is generated
2. **If correct**, click ✅ Save as Example
3. The Q→SQL pair is saved to `fewshots.json`
4. Next time a **similar** question is asked, the FAISS retriever finds it and uses it as a reference in the prompt
5. The LLM produces more accurate SQL by following proven patterns

You can also **manually add examples** from the sidebar without running a query first.

---

## ⚙️ Configuration

| Variable | Location | Description |
|---|---|---|
| `GOOGLE_API_KEY` | `.env` | Your Google AI Studio API key |
| `DB_URL` | `.env` | SQLAlchemy connection string for PostgreSQL |
| `TOP_K_EXAMPLES` | `app.py` | Number of similar examples to retrieve (default: `3`) |
| `FEW_SHOTS_FILE` | `app.py` | Path to the few-shots JSON file (default: `fewshots.json`) |

---

## 🔐 Security Notes

- **Never commit your `.env` file** — add it to `.gitignore`
- The app enforces **read-only queries** (only `SELECT` / `WITH` statements execute)
- All credentials stay local — nothing is sent except prompts to the Google AI API

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

Made with ❤️ using Streamlit, LangChain & Google Gemini

</div>
