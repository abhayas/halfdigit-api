# HalfDigit API ✅

A small Flask API that exposes a Titanics prediction demo, a contact form handler (stores messages and sends emails via Resend), and a speech-to-text endpoint using Hugging Face's Whisper model.

---

## Table of Contents

- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment variables](#environment-variables)
- [Install](#install)
- [Database schema (example)](#database-schema-example)
- [API Endpoints](#api-endpoints)
- [Run locally](#run-locally)
- [Production tips](#production-tips)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Project structure 🔧

Only non-ignored files are shown (files matched by `.gitignore` such as `venv/`, `*.ipynb`, `*.csv`, and `.env` are intentionally excluded):

- `app.py` – main Flask application and API endpoints
- `requirements.txt` – Python dependencies
- `model/` – directory that contains `Titanic_model.pkl` (trained model artifact)

---

## Prerequisites 🧰

- Python 3.8+
- PostgreSQL (or a hosted PostgreSQL compatible DB)
- An active Hugging Face token (for speech-to-text requests)
- Optional: Resend API key for sending emails

---

## Environment variables 🔒

Set these in your environment (or in a local `.env` — note `.env` is ignored by git):

- `DATABASE_URL` — PostgreSQL connection string 
- `RESEND_API_KEY` — (optional) API key for Resend to send emails
- `HF_TOKEN` — Hugging Face token used for the speech-to-text call

---

## Install ⚙️

1. Create and activate a virtual environment (recommended):

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the model file `model/Titanic_model.pkl` exists in the `model/` folder.

---

## Database schema (example) 🗄️

Application expects a Postgres database and inserts into two tables. Example SQL to create them:

```sql
CREATE TABLE visits (
  id serial PRIMARY KEY,
  page_path text,
  model_name text,
  prediction integer,
  probability real,
  payload jsonb,
  created_at timestamptz DEFAULT now()
);

CREATE TABLE contact_messages (
  id serial PRIMARY KEY,
  name text,
  email text,
  message text,
  source_page text,
  created_at timestamptz DEFAULT now()
);
```

---

## API Endpoints 📡

1) POST /predict-titanic

- Description: Predicts survival using the loaded Titanic model and logs the visit.
- Body (JSON):

```json
{
  "pclass": 1,
  "age": 29.0,
  "sibsp": 0,
  "parch": 0,
  "adult_male": 1,
  "alone": 0,
  "male": 1
}
```

- Response example:

```json
{ "survived": true, "probability": 0.87 }
```

2) POST /contact

- Description: Stores a contact message in the database and (optionally) sends emails (requires `RESEND_API_KEY`).
- Body (JSON):

```json
{
  "name": "Jane Doe",
  "email": "jane@example.com",
  "message": "Hello!",
  "source_page": "/some-page"   // optional
}
```

- Response: `{ "status": "success" }` on success.

3) POST /speech-to-text

- Description: Accepts an audio file (multipart form field named `audio`) and returns a transcription using Hugging Face.
- Example curl:

```bash
curl.exe -F "audio=@./sample.wav" http://localhost:8000/speech-to-text
```

- Response example: `{ "transcript": "Transcribed text here" }`

---

## Run locally ▶️

Set the required environment variables, then run:

```bash
python app.py
```

The server listens on port `8000` by default (see `app.py`). For production, use Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

---

## Production tips 🔐

- Keep secrets (DB credentials, API keys) out of the repo — store them in environment variables or a secret store.
- Use HTTPS in front of the app (proxy/nginx or a managed host).
- Tune Gunicorn worker counts according to available CPU/memory.

---

## Troubleshooting ⚠️

- If the Hugging Face call fails, check `HF_TOKEN` and network connectivity. API errors are printed to the server logs.
- Database connection errors typically mean `DATABASE_URL` is missing or incorrect.
- If email sending silently fails, confirm `RESEND_API_KEY` is set and valid.

---

## Contributing ✨

Contributions are welcome — open an issue or create a PR. Please follow standard best practices for Python projects and include tests where appropriate.

---

## License

This repository does not currently include a license file. Add one if you wish to specify licensing terms.

---

If you'd like, I can also add a short `docker-compose` example, a health-check endpoint, or CI config — tell me which you prefer. 💡