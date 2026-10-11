<div align="center">

# TheGuard

### Pause before you click. Check a message for scam signals.

TheGuard is an AI-powered web app that helps people review suspicious SMS messages. Paste a message to get a fraud risk assessment, a breakdown of warning signs, and practical safety guidance.

[**Open TheGuard**](https://theguard-two.vercel.app/)

</div>

---

## What TheGuard does

- Analyzes a message for signs commonly associated with scams and fraud.
- Shows a risk assessment and highlights signals that influenced it.
- Provides clear next steps to help you stay safe.
- Offers a simple web interface for checking messages.

TheGuard is a decision-support tool. Treat its result as guidance, and verify unexpected requests through an official channel before taking action.

## Try the website

1. Visit [theguard-two.vercel.app](https://theguard-two.vercel.app/).
2. Paste the SMS or message you want to check.
3. Review the assessment, warning signs, and safety advice.

## How it works

The web app is built with Flask. Its trained text classifier combines word and character patterns with engineered scam signals. The `/analyze` endpoint returns the assessment and explanations; `/health` reports whether the app is responding.

Messages are processed by the deployed application to produce a result. The app code does not save submitted messages to a database.

## Run locally

You’ll need Python and pip.

```bash
git clone https://github.com/GoraniyaHarish/TheGuard.git
cd TheGuard
python -m venv .venv
```

Activate the virtual environment, then install the runtime dependencies and start the app:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux: use this activation command instead
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## API

Check a message with a JSON `POST` request:

```bash
curl -X POST http://127.0.0.1:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"message":"Your package is held. Pay a fee now to release it."}'
```

The endpoint returns a JSON assessment with warning signals and advice. The optional `lang` field selects the advice language.

Check the app status at `GET /health`.

## Deployments

- **Live site:** [theguard-two.vercel.app](https://theguard-two.vercel.app/)
- **Source code:** [github.com/GoraniyaHarish/TheGuard](https://github.com/GoraniyaHarish/TheGuard)

The project is configured for Vercel’s Python/Flask deployment. Pushes to the connected GitHub branch can trigger a new deployment.

## Project structure

| File or folder | Purpose |
| --- | --- |
| `app.py` | Flask web app and API routes |
| `templates/index.html` | Web interface |
| `model.joblib` | Trained hybrid text classifier |
| `features.py` | Engineered message signals |
| `agent.py` | Explanations and safety guidance |
| `requirements.txt` | Runtime Python dependencies |
| `train.py`, `models.py`, `dataset.py`, `data.py` | Model training and data preparation |

## Help improve TheGuard

Bug reports and suggestions are welcome. Open an issue in the [GitHub repository](https://github.com/GoraniyaHarish/TheGuard/issues).
