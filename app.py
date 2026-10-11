"""Simple English AI chatbot using the OpenAI API and Streamlit."""
import os

import streamlit as st
from dotenv import load_dotenv
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

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
        try:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state.messages,
                stream=True,
            )
            reply = st.write_stream(stream)
        except AuthenticationError:
            st.error("Invalid OpenAI API key. Check the key and try again.")
            st.session_state.messages.pop()
            st.stop()
        except RateLimitError as e:
            if "insufficient_quota" in str(e):
                st.error(
                    "Your OpenAI account has no credits left. Add credits at "
                    "https://platform.openai.com/settings/organization/billing/overview and try again."
                )
            else:
                st.warning("Too many requests right now. Please wait a few seconds and try again.")
            st.session_state.messages.pop()
            st.stop()
        except APIError as e:
            st.error(f"OpenAI error: {e}")
            st.session_state.messages.pop()
            st.stop()
    st.session_state.messages.append({"role": "assistant", "content": reply})
