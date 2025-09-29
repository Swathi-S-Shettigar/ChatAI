# import requests
# from flask import Flask, request, jsonify
# from sqlalchemy import create_engine, text
# import pandas as pd
# import os
# from dotenv import load_dotenv

# app = Flask(__name__)
# load_dotenv()

# # Hugging Face
# HF_TOKEN = os.getenv("HF_TOKEN")
# HF_MODEL = "tiiuae/falcon-7b-instruct"
# HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

# # MySQL Connection
# DB_USER = "root"
# DB_PASS = "Root123."
# DB_HOST = "localhost"
# DB_NAME = "ai_data_agent"

# engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# # -------------------- Upload File -------------------- #
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     filename = file.filename
#     ext = os.path.splitext(filename)[1].lower()

#     try:
#         if ext == ".csv":
#             df = pd.read_csv(file)
#         elif ext == ".xls":
#             df = pd.read_excel(file, engine="xlrd")
#         elif ext == ".xlsx":
#             df = pd.read_excel(file, engine="openpyxl")
#         else:
#             return jsonify({"error": "Unsupported file type. Use .csv, .xls, or .xlsx"}), 400

#         # Handle unnamed columns
#         df.columns = [
#             f"col_{i}" if "Unnamed" in str(c) else str(c).strip().replace(" ", "_").lower()
#             for i, c in enumerate(df.columns)
#         ]

#         df.to_sql("uploaded_data", con=engine, if_exists="replace", index=False)

#         return jsonify({"message": "File uploaded and saved to DB", "columns": df.columns.tolist()})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # -------------------- Ask Question -------------------- #
# @app.route("/ask", methods=["POST"])
# def ask_question():
#     data = request.get_json()
#     question = data.get("question")
#     if not question:
#         return jsonify({"error": "No question provided"}), 400

#     try:
#         # Get table schema from MySQL
#         table_name = "uploaded_data"
#         with engine.connect() as conn:
#             result = conn.execute(text(f"DESCRIBE {table_name}"))
#             schema_info = result.fetchall()

#         schema_str = "\n".join([f"{col[0]} ({col[1]})" for col in schema_info])

#         # Build prompt for HF model
#         prompt = f"""
# You are an assistant that converts natural language questions to SQL queries.
# Table schema:
# {schema_str}

# Question: {question}

# Return only a SQL query for MySQL.
# """

#         headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#         payload = {"inputs": prompt, "parameters": {"temperature": 0}}

#         response = requests.post(HF_API_URL, headers=headers, json=payload)
#         print("HF raw response:", response.text)  # Debugging

#         result = response.json()
#         if isinstance(result, dict) and "error" in result:
#             return jsonify({"error": result["error"]}), 500

#         # Extract generated SQL
#         sql_query = result[0]["generated_text"].strip()

#         # Execute SQL
#         with engine.connect() as conn:
#             df = pd.read_sql(sql_query, conn)

#         return jsonify({
#             "sql": sql_query,
#             "table": df.to_dict(orient="records")
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)



# import requests
# from flask import Flask, request, jsonify
# from sqlalchemy import create_engine, text
# import pandas as pd
# import os
# from dotenv import load_dotenv

# app = Flask(__name__)
# load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS  # add this
import requests
import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)  # 👈 allow all origins (React frontend can connect)

load_dotenv()

# -------------------- OpenRouter Setup -------------------- #
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o"  # you can change model if needed

# -------------------- MySQL Setup -------------------- #
DB_USER = "root"
DB_PASS = "Root123."
DB_HOST = "localhost"
DB_NAME = "ai_data_agent"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")


# -------------------- OpenRouter Query -------------------- #
def query_openrouter(prompt: str):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }

    response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


# -------------------- Upload File -------------------- #
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    try:
        # Read file
        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext == ".xls":
            df = pd.read_excel(file, engine="xlrd")
        elif ext == ".xlsx":
            df = pd.read_excel(file, engine="openpyxl")
        else:
            return jsonify({"error": "Unsupported file type. Use .csv, .xls, or .xlsx"}), 400

        # Clean column names
        df.columns = [
            f"col_{i}" if "Unnamed" in str(c) else str(c).strip().replace(" ", "_").lower()
            for i, c in enumerate(df.columns)
        ]

        # Save to MySQL
        df.to_sql("uploaded_data", con=engine, if_exists="replace", index=False)

        return jsonify({"message": "File uploaded and saved to DB", "columns": df.columns.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------- Ask Question -------------------- #
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        table_name = "uploaded_data"

        # Get table schema
        with engine.connect() as conn:
            result = conn.execute(text(f"DESCRIBE {table_name}"))
            schema_info = result.fetchall()

        schema_str = "\n".join([f"{col[0]} ({col[1]})" for col in schema_info])

        # Build prompt
        prompt = f"""
You are an assistant that converts natural language questions into MySQL queries.
Table schema:
{schema_str}

Question: {question}

Return ONLY the SQL query (no explanation) and use the table name '{table_name}' directly.
"""

        sql_query = query_openrouter(prompt)

        # Clean query: remove ```sql / ``` and placeholders
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = sql_query.replace("<table_name>", table_name)

        # Execute SQL
        with engine.connect() as conn:
            df = pd.read_sql(sql_query, conn)

        return jsonify({
            "sql": sql_query,
            "results": df.to_dict(orient="records")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)



# without api key

# from flask import Flask, request, jsonify
# import sqlite3

# app = Flask(__name__)

# # -----------------------------
# # Setup SQLite database
# # -----------------------------
# def init_db():
#     conn = sqlite3.connect("business.db")
#     cur = conn.cursor()

#     # Example Sales table
#     cur.execute("""
#     CREATE TABLE IF NOT EXISTS Sales (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         region TEXT,
#         product TEXT,
#         revenue REAL
#     )
#     """)

#     # Insert sample data (only if empty)
#     cur.execute("SELECT COUNT(*) FROM Sales")
#     if cur.fetchone()[0] == 0:
#         sample_data = [
#             ("North", "Laptop", 1200),
#             ("South", "Laptop", 1000),
#             ("North", "Phone", 800),
#             ("West", "Tablet", 600),
#             ("South", "Tablet", 700),
#             ("West", "Laptop", 1100),
#         ]
#         cur.executemany("INSERT INTO Sales (region, product, revenue) VALUES (?, ?, ?)", sample_data)

#     conn.commit()
#     conn.close()

# # -----------------------------
# # Helper: Convert natural language → SQL (rule-based)
# # -----------------------------
# def nl_to_sql(question: str):
#     q = question.lower()

#     # 1. Revenue summaries
#     if "total revenue per region" in q:
#         return "SELECT region, SUM(revenue) as total_revenue FROM Sales GROUP BY region"

#     if "total revenue per product" in q:
#         return "SELECT product, SUM(revenue) as total_revenue FROM Sales GROUP BY product"

#     if "total revenue" in q:
#         return "SELECT SUM(revenue) as total_revenue FROM Sales"

#     # 2. Averages
#     if "average revenue per region" in q:
#         return "SELECT region, AVG(revenue) as avg_revenue FROM Sales GROUP BY region"

#     if "average revenue per product" in q:
#         return "SELECT product, AVG(revenue) as avg_revenue FROM Sales GROUP BY product"

#     if "average revenue" in q:
#         return "SELECT AVG(revenue) as avg_revenue FROM Sales"

#     # 3. Min/Max
#     if "highest revenue region" in q:
#         return "SELECT region, SUM(revenue) as total_revenue FROM Sales GROUP BY region ORDER BY total_revenue DESC LIMIT 1"

#     if "lowest revenue region" in q:
#         return "SELECT region, SUM(revenue) as total_revenue FROM Sales GROUP BY region ORDER BY total_revenue ASC LIMIT 1"

#     if "highest selling product" in q:
#         return "SELECT product, SUM(revenue) as total_revenue FROM Sales GROUP BY product ORDER BY total_revenue DESC LIMIT 1"

#     if "lowest selling product" in q:
#         return "SELECT product, SUM(revenue) as total_revenue FROM Sales GROUP BY product ORDER BY total_revenue ASC LIMIT 1"

#     # 4. Raw data
#     if "all sales" in q or "list all sales" in q:
#         return "SELECT * FROM Sales"

#     # No match
#     return None


# # -----------------------------
# # API route
# # -----------------------------
# @app.route("/ask", methods=["POST"])
# def ask():
#     try:
#         data = request.get_json()
#         question = data.get("question")

#         sql_query = nl_to_sql(question)
#         if not sql_query:
#             return jsonify({"error": "Sorry, I cannot understand this question yet."})

#         conn = sqlite3.connect("business.db")
#         cur = conn.cursor()
#         cur.execute(sql_query)
#         rows = cur.fetchall()

#         # Column names
#         col_names = [desc[0] for desc in cur.description]

#         # Convert rows → list of dicts
#         results = [dict(zip(col_names, row)) for row in rows]

#         conn.close()
#         return jsonify({
#             "question": question,
#             "sql_query": sql_query,
#             "results": results
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)})

# # -----------------------------
# # Start app
# # -----------------------------
# if __name__ == "__main__":
#     init_db()
#     app.run(debug=True, port=5000)
