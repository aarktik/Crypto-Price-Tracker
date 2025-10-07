# Crypto Price Tracker — Themed + AI Summary

Dark/Blue theme + AI Market Summarizer (OpenAI-compatible). **Data analysis only**.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Keys (optional)
# PowerShell:
# $env:COINGECKO_API_KEY="YOUR_CG_KEY"
# $env:OPENAI_API_KEY="YOUR_OPENAI_KEY"

streamlit run app.py
```

## AI Summarizer
- Reads `OPENAI_API_KEY` and uses `/v1/chat/completions` (model default: `gpt-4o-mini`).
- If no key is set, falls back to a rule-based summary.
- Adjust model via `OPENAI_MODEL` env or in-app text box.

## Theme
- See `.streamlit/config.toml` (Dark/Blue).

