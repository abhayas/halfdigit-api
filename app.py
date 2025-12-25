from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sklearn
import os
from dotenv import load_dotenv
import psycopg2
import json
import resend





app = Flask(__name__)
CORS(app)

titanic_model = joblib.load("model/Titanic_model.pkl")

TITANIC_FEATURES = [
    "pclass", "age", "sibsp", "parch", "adult_male", "alone", "male"
]



DATABASE_URL = os.environ.get("DATABASE_URL")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY")




@app.route("/")
def home():
    return{"status": "ok"}

@app.route("/health")
def health():
    return '<h1> Hello how are you? </h1>'

def get_conn():
    return psycopg2.connect(DATABASE_URL,sslmode="require")


@app.route("/predict-titanic", methods=["POST"])
def predict_titanic():
    data = request.get_json()

    # Build input vector in correct order
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

    # Log to DB
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO visits (page_path, model_name, prediction, probability, payload)
        VALUES (%s,%s,%s,%s,%s)
        """,
        (
            "/titanic-demo",
            "titanic-rf",
            int(prediction),
            float(probability),
            json.dumps(data)
        )
    )
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({
        "survived": bool(prediction),
        "probability": round(float(probability), 2)
    })



@app.route("/predict")
def predict():
    loadmodel = joblib.load("model/kyphosis-model_new.pkl")
    xmanual = np.array([[158,3,14]])
    predimanual = loadmodel.predict(xmanual)
    conn = get_conn()
    cur = conn.cursor()
    payload = json.dumps(predimanual[0].tolist())
   

    cur.execute(

        """
        INSERT INTO  visits ( page_path, model_name, prediction, probability, payload)
        VALUES (%s, %s, %s, %s, %s)""",

        ("/predict", "kyphosis-model_new","jsonify(predimanual[0].tolist())", 1,payload )
    )

    conn.commit()
    cur.close()
    conn.close()



    return jsonify(predimanual[0].tolist())



@app.route("/log-visit", methods=["POST"])
def log_visit():
    data = request.get_json()

    page_path = data.get("page_path", "/")
    referrer = data.get("referrer", "")
    user_agent = data.get("user_agent", "")

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO visits (page_path, model_name, prediction, probability, payload)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (
            page_path,
            "page-visit",
            None,
            None,
            json.dumps({
                "referrer": referrer,
                "user_agent": user_agent
            })
        )
    )

    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"status": "logged"})




@app.route("/contact", methods=["POST"])
def contact():
    data = request.get_json()

    name = data["name"]
    email = data["email"]
    message = data["message"]
    source_page = data.get("source_page", "")

    # Save to DB
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

    # Send email (Resend / SMTP)
    send_email(name, email, message, source_page)
    send_user_copy(name, email, message)

    return jsonify({"status": "success"})

def send_email(name, email, message, source_page):
    resend.Emails.send({ 

        "from" : "HalfDigit Contact Form <no-reply@halfdigit.com>",
        "to" : [ "sahuabhayaa333@gmail.com", "abhayas@zohomail.in" ],
        "subject" : f"new contact message from {name}",
        "html" : f"""
                    <p><strong>Name :</strong>{name}</p>
                    <p><strong>Email :</strong>{email}</p>
                    <p><strong>Message :</strong>{message}</p>
                    <p><strong>Source Page :</strong>{source_page}</p>
                    <p>{message}</p>      

                """

        
       })
    



def send_user_copy(name, email, message):
    resend.Emails.send({
        "from": "Abhaya Prasad Sahu <no-reply@halfdigit.com>",
        "to": [email],
        "subject": "Thanks for reaching out – HalfDigit.com",
        "html": f"""
            <p>Hi {name},</p>

            <p>Thanks for reaching out via my portfolio website <a href="https://halfdigit.com">halfdigit.com</a>.</p>

            <p>I have received your message and will personally get back to you soon.</p>

            <hr>

            <p><b>Your message:</b></p>
            <blockquote>{message}</blockquote>

            <p>
                Best regards,<br>
                <b>Abhaya Prasad Sahu</b><br>
                <span style="color:#555;">
                    AI/ML | Data Science | SharePoint | <br>
                    <a href="https://halfdigit.com">halfdigit.com</a>
                </span>
            </p>
        """
    })



if __name__ == "__main__":
    app.run(debug=True, port=8000)