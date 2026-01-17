from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
from dotenv import load_dotenv
import psycopg2
import json
import resend
import requests
from rag import rag_chain, retriever

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
# OLD (Broken in v0.3)
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# NEW (Correct Explicit Paths)


from langchain_core.prompts import ChatPromptTemplate




# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# --- Configuration ---
DATABASE_URL = os.environ.get("DATABASE_URL")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
titanic_model = joblib.load("model/Titanic_model.pkl")


# --- Database Connection ---
def get_conn():
    return psycopg2.connect(DATABASE_URL, sslmode="require")


@app.route("/predict-titanic", methods=["POST"])
def predict_titanic():
    data = request.get_json()
    
    X = np.array([[
        int(data["pclass"]),
        float(data["age"]),
        int(data["sibsp"]),
        int(data["parch"]),
        int(data["adult_male"]),
        int(data["alone"]),
        int(data["male"])
    ]])

    prediction = titanic_model.predict(X)[0]
    probability = titanic_model.predict_proba(X).max()

    
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO visits (page_path, model_name, prediction, probability, payload)
        VALUES (%s,%s,%s,%s,%s)
        """,
        ("/titanic-demo", 
         "titanic-rf", 
         int(prediction), 
         float(probability), 
         json.dumps(data))
    )
    conn.commit()
    cur.close()
    conn.close()
    

    return jsonify({
        "survived": bool(prediction),
        "probability": round(float(probability), 2)
    })






# --- Contact Form ---
@app.route("/contact", methods=["POST"])
def contact():
    data = request.get_json()
    name = data["name"]
    email = data["email"]
    message = data["message"]
    source_page = data.get("source_page", "")

    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO contact_messages (name, email, message, source_page)
            VALUES (%s, %s, %s, %s)
            """,
            (name, email, message, source_page)
        )
        conn.commit()
        cur.close()
        conn.close()

        send_email(name, email, message, source_page)
        send_user_copy(name, email, message)
    except Exception as e:
        print(f"Contact Error: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({"status": "success"})

def send_email(name, email, message, source_page):
    if not RESEND_API_KEY: return
    try:
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from": "HalfDigit Contact Form <no-reply@halfdigit.com>",
            "to": ["sahuabhayaa333@gmail.com", "abhayas@zohomail.in"],
            "subject": f"new contact message from {name}",
            "html": f"<p><strong>Name:</strong>{name}</p><p><strong>Email:</strong>{email}</p><p><strong>Message:</strong>{message}</p><p><strong>Source:</strong>{source_page}</p>"
        })
    except Exception as e:
        print(f"Email Error: {e}")

def send_user_copy(name, email, message):
    if not RESEND_API_KEY: return
    try:
        resend.api_key = RESEND_API_KEY
        resend.Emails.send({
            "from": "Abhaya Prasad Sahu <no-reply@halfdigit.com>",
            "to": [email],
            "subject": "Thanks for reaching out – HalfDigit.com",
            "html": f"<p>Hi {name},</p><p>Thanks for reaching out via <a href='https://halfdigit.com'>halfdigit.com</a>.</p><p>I will get back to you soon.</p><hr><blockquote>{message}</blockquote>"
        })
    except Exception as e:
        print(f"User Copy Email Error: {e}")



def transcribe_with_hf(audio_bytes, content_type="audio/wav"):

    print(content_type)
    
    api_url = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": content_type
    }

    print(f"Attempting to send {len(audio_bytes)} bytes to {api_url}...")

    response = requests.post(
        api_url,
        headers=headers,
        data=audio_bytes,
        timeout=100
    )

    if response.status_code != 200:
        print(f"HF Error {response.status_code}: {response.text}")
        raise Exception(f"HF Error: {response.text}")

    result = response.json()
    
    return result.get("text", str(result))


@app.route("/speech-to-text", methods=["POST"])
def speech_to_text():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_bytes = audio_file.read()
    contenttype= audio_file.content_type
    if not contenttype or contenttype == 'application/octet-stream':
       
          
        if audio_file.filename.lower().endswith(".mp3"):
            contenttype = "audio/mpeg"
        elif audio_file.filename.lower().endswith(".wav"):
            contenttype = "audio/wav"
        else:
            contenttype = "audio/wav"


    try:
        transcript = transcribe_with_hf(audio_bytes,contenttype)
        return jsonify({"transcript": transcript})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route('/chat-about-me', methods=['POST'])
def chat_about_me():
    # ... existing setup ...
    data = request.json
    user_query = data.get('query')

    # --- DEBUGGING START ---
    # This will print found documents to your VS Code terminal
    print(f"\nUser asked: {user_query}")
    docs = retriever.invoke(user_query)
    print(f"Retrieved {len(docs)} relevant chunks.")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:200]) # Print first 200 chars of each chunk
        print("----------------")
    # --- DEBUGGING END ---

    response = rag_chain.invoke(user_query)
    return jsonify({"answer": response})

from flask import Flask, request, jsonify
import psycopg2  # or sqlite3
from datetime import datetime

app = Flask(__name__)

# Database connection function (simplified)
def get_db_connection():
    return psycopg2.connect(
        host="your-db-host",
        database="your-db-name",
        user="your-db-user",
        password="your-db-password"
    )



@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('message')
    session_id = data.get('session_id', 'anonymous') # generated by frontend

    # 1. Generate your AI response (Your existing RAG logic)
    # response_text = your_rag_function(user_question)
    response_text = "This is a dummy AI response for " + user_question

    # 2. LOGGING: Save to Database
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_logs (user_session_id, user_question, ai_response) VALUES (%s, %s, %s)",
            (session_id, user_question, response_text)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Logging failed: {e}") # Don't crash the chat if logging fails

    # 3. Return response to user
    return jsonify({"response": response_text})


if __name__ == "__main__":
    app.run(debug=True, port=8000)