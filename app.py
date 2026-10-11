"""Simple English AI chatbot using Google Gemini (OpenAI-compatible API) and Streamlit."""
import os

import streamlit as st
from dotenv import load_dotenv
from openai import APIError, APIStatusError, AuthenticationError, OpenAI, RateLimitError

load_dotenv()  # reads GEMINI_API_KEY from a local .env file (never commit it)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK_MODELS = [MODEL, "gemini-3.5-flash", "gemini-flash-latest"]
SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant. "
    "Always respond in clear, simple English. "
    "Keep answers concise and accurate, and ask a follow-up question when the request is unclear."
)


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:")
st.title("AI Chatbot")
st.caption(f"Powered by Google Gemini - model: {MODEL}")

api_key = get_secret("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API key", type="password")
if not api_key:
    st.info("Add your Gemini API key to Streamlit secrets, a .env file, or the sidebar to start chatting.")
    st.stop()

client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def ask(model):
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages,
        stream=True,
    )
    return st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)


if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        reply, error = None, None
        for model in dict.fromkeys(FALLBACK_MODELS):
            try:
                reply = ask(model)
                break
            except AuthenticationError:
                error = "Invalid Gemini API key. Check the key and try again."
                break
            except RateLimitError:
                error = "Free Gemini quota reached for now. Please wait a minute and try again."
            except APIStatusError as e:
                error = f"Gemini is busy or unavailable right now ({e.status_code}). Please try again."
            except APIError as e:
                error = f"Gemini error: {e}"
        if reply is None:
            st.error(error)
            st.session_state.messages.pop()
            st.stop()
    st.session_state.messages.append({"role": "assistant", "content": reply})
