"""Simple English AI chatbot using the OpenAI API and Streamlit."""
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENAI_API_KEY from a local .env file (never commit it)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a friendly, helpful AI assistant. "
    "Always respond in clear, simple English. "
    "Keep answers concise and accurate, and ask a follow-up question when the request is unclear."
)

st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:")
st.title("AI Chatbot")
st.caption(f"Powered by OpenAI - model: {MODEL}")

api_key = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API key", type="password")
if not api_key:
    st.info("Add your OpenAI API key to a .env file or the sidebar to start chatting.")
    st.stop()

client = OpenAI(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages,
            stream=True,
        )
        reply = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": reply})
