from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import sklearn
import os
from dotenv import load_dotenv
import psycopg2
import json



app = Flask(__name__)
CORS(app)

titanic_model = joblib.load("model/Titanic_model.pkl")

TITANIC_FEATURES = [
    "pclass", "age", "sibsp", "parch", "adult_male", "alone", "male"
]



DATABASE_URL = os.environ.get("DATABASE_URL")

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








if __name__ == "__main__":
    app.run(debug=True, port=8000)