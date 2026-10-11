# OpenAI Chatbot

A simple English AI chatbot built with [Streamlit](https://streamlit.io) and the [OpenAI API](https://platform.openai.com).

## Features
- Chat interface with streaming replies
- Remembers the conversation during a session
- Always responds in clear, simple English
- API key kept out of the code (loaded from `.env` or entered in the sidebar)

## Run locally
```bash
git clone https://github.com/saicharanllm/openai-chatbot.git
cd openai-chatbot
pip install -r requirements.txt
cp .env.example .env      # then put your OpenAI key in .env
streamlit run app.py
```
Open http://localhost:8501 in your browser.

## Change the model or personality
- Model: set `OPENAI_MODEL` in `.env` (default `gpt-4o-mini`).
- Personality: edit `SYSTEM_PROMPT` in `app.py`.

## Security
Never commit your `.env` file or paste your API key into the code. `.env` is already in `.gitignore`.
