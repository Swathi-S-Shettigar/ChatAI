
# from flask import Flask, request, jsonify
# from flask_cors import CORS  # add this
# import requests
# import pandas as pd
# import os
# from sqlalchemy import create_engine, text
# from dotenv import load_dotenv


# app = Flask(__name__)
# CORS(app)  # 👈 allow all origins (React frontend can connect)

# load_dotenv()

# # -------------------- OpenRouter Setup -------------------- #
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# MODEL_NAME = "openai/gpt-4o"  # you can change model if needed

# # -------------------- MySQL Setup -------------------- #
# # DB_USER = "root"
# # DB_PASS = "Root123."
# # DB_HOST = "localhost"
# # DB_NAME = "ai_data_agent"
# DB_USER = os.getenv("DB_USER", "root")
# DB_PASS = os.getenv("DB_PASS", "Root123.")
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_NAME = os.getenv("DB_NAME", "ai_data_agent")


# engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")


# @app.route("/")
# def home():
#     return {"message": "Backend is running ✅"}


# # -------------------- OpenRouter Query -------------------- #
# def query_openrouter(prompt: str):
#     headers = {
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     payload = {
#         "model": MODEL_NAME,
#         "messages": [{"role": "user", "content": prompt}],
#         "max_tokens": 1000
#     }

#     response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    
#     if response.status_code != 200:
#         raise Exception(f"OpenRouter API error {response.status_code}: {response.text}")

#     result = response.json()
#     return result["choices"][0]["message"]["content"].strip()


# # -------------------- Upload File -------------------- #
# @app.route("/upload", methods=["POST"])
# def upload_file():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     filename = file.filename
#     ext = os.path.splitext(filename)[1].lower()

#     try:
#         # Read file
#         if ext == ".csv":
#             df = pd.read_csv(file)
#         elif ext == ".xls":
#             df = pd.read_excel(file, engine="xlrd")
#         elif ext == ".xlsx":
#             df = pd.read_excel(file, engine="openpyxl")
#         else:
#             return jsonify({"error": "Unsupported file type. Use .csv, .xls, or .xlsx"}), 400

#         # Clean column names
#         df.columns = [
#             f"col_{i}" if "Unnamed" in str(c) else str(c).strip().replace(" ", "_").lower()
#             for i, c in enumerate(df.columns)
#         ]

#         # Save to MySQL
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
#         table_name = "uploaded_data"

#         # Get table schema
#         with engine.connect() as conn:
#             result = conn.execute(text(f"DESCRIBE {table_name}"))
#             schema_info = result.fetchall()

#         schema_str = "\n".join([f"{col[0]} ({col[1]})" for col in schema_info])

#         # Build prompt
#         prompt = f"""
# You are an assistant that converts natural language questions into MySQL queries.
# Table schema:
# {schema_str}

# Question: {question}

# Return ONLY the SQL query (no explanation) and use the table name '{table_name}' directly.
# """

#         sql_query = query_openrouter(prompt)

#         # Clean query: remove ```sql / ``` and placeholders
#         sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
#         sql_query = sql_query.replace("<table_name>", table_name)

#         # Execute SQL
#         with engine.connect() as conn:
#             df = pd.read_sql(sql_query, conn)

#         return jsonify({
#             "sql": sql_query,
#             "results": df.to_dict(orient="records")
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500




# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

# -------------------- API Keys -------------------- #
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o"

# -------------------- Database -------------------- #
# DB_USER = os.getenv("DB_USER", "root")
# DB_PASS = os.getenv("DB_PASS", "Root123.")
# DB_HOST = os.getenv("DB_HOST", "localhost")
# DB_PORT = os.getenv("DB_PORT", "3306")
# DB_NAME = os.getenv("DB_NAME", "ai_data_agent")

# engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

@app.route("/")
def home():
    return {"message": "✅ Backend running successfully"}

# -------------------- Query OpenRouter -------------------- #
def query_openrouter(prompt: str):
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}
    res = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
    if res.status_code != 200:
        raise Exception(f"OpenRouter error: {res.text}")
    return res.json()["choices"][0]["message"]["content"].strip()

# -------------------- Upload File -------------------- #
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()

    try:
        if ext == ".csv":
            df = pd.read_csv(file)
        elif ext in [".xls", ".xlsx"]:
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        df.columns = [
            f"col_{i}" if "Unnamed" in str(c) else str(c).strip().replace(" ", "_").lower()
            for i, c in enumerate(df.columns)
        ]

        df.to_sql("uploaded_data", con=engine, if_exists="replace", index=False)

        return jsonify({"message": "✅ File uploaded and stored in database", "columns": df.columns.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Ask Question -------------------- #
@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question missing"}), 400

    try:
        with engine.connect() as conn:
            schema = conn.execute(text("DESCRIBE uploaded_data")).fetchall()
        schema_text = "\n".join([f"{col[0]} ({col[1]})" for col in schema])

        prompt = f"""
You are an AI that converts questions into MySQL queries.
Schema:
{schema_text}

Question: {question}

Return only SQL query using table 'uploaded_data'.
"""
        sql_query = query_openrouter(prompt)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        with engine.connect() as conn:
            df = pd.read_sql(sql_query, conn)

        return jsonify({"sql": sql_query, "results": df.to_dict(orient="records")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
