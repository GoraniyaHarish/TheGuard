"""TheGuard - Flask application.

Run:  python3 app.py     then open http://127.0.0.1:5000
Endpoints:
  GET  /            the web interface
  POST /analyze     {"message": "..."} -> verdict + red flags + advice
  GET  /health      model + corpus info
"""
import os
import joblib
from flask import Flask, request, jsonify, render_template
from agent import explain, RED_FLAG_RULES
import features

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = joblib.load(os.path.join(BASE, "model.joblib"))

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object with a 'message' field."}), 400
    raw = data.get("message")
    message = raw.strip() if isinstance(raw, str) else ""
    lang = data.get("lang", "en")
    if not message:
        return jsonify({"error": "Please paste a message to check."}), 400
    score = float(MODEL.predict_proba([message])[0][1])
    result = explain(message, score, lang)
    result["message_length"] = len(message)
    result["signals"] = features.explain(message)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "Hybrid: TF-IDF (word+char) + engineered scam signals (LogReg)",
        "agent_rules": len(RED_FLAG_RULES),
        "engineered_signals": features.N_ENGINEERED,
        "classes": ["legitimate", "fraudulent"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
