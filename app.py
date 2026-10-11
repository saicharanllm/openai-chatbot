"""Simple English AI chatbot using Google Gemini (OpenAI-compatible API) and Streamlit."""
import os
import uuid

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

FEATURES = [
    "Web Search",
    "File Analysis",
    "Document Creation",
    "Image Generation",
    "Coding & Debugging",
    "Data Analysis",
    "Memory",
    "Scheduled Tasks",
    "Voice Mode",
    "Computer Assistance",
    "File Conversion",
    "PDF Analysis",
    "Image Analysis",
    "Data Visualization",
    "Website Development",
    "Code Execution",
    "Personalized Learning",
    "Project Assistance",
    "Canvas & Writing",
]


def get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)


st.set_page_config(page_title="AI Chatbot", page_icon=":robot_face:", layout="wide")

# ---------- Sidebar styling (ChatGPT-like list) ----------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] { background-color: #f9f9f9; }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        justify-content: flex-start;
        text-align: left;
        border: none;
        background: transparent;
        padding: 6px 10px;
        border-radius: 10px;
        font-size: 15px;
        color: #222;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { background: #ececec; }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #ececec; color: #222; font-weight: 500;
    }
    section[data-testid="stSidebar"] .stButton > button p {
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .side-heading {
        color: #8e8e8e; font-size: 14px; font-weight: 600;
        margin: 18px 0 4px 10px;
    }
    .side-empty { color: #b0b0b0; font-size: 14px; margin: 0 0 6px 10px; }
    section[data-testid="stSidebar"] [data-testid="stPopover"] button {
        border: none; background: transparent; padding: 4px 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Session state ----------
if "chats" not in st.session_state:
    st.session_state.chats = {}  # id -> {"title", "messages", "pinned"}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "feature" not in st.session_state:
    st.session_state.feature = None


def new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {"title": "New chat", "messages": [], "pinned": False}
    st.session_state.current_chat = chat_id
    st.session_state.feature = None


def delete_chat(chat_id):
    st.session_state.chats.pop(chat_id, None)
    if st.session_state.current_chat == chat_id:
        st.session_state.current_chat = None


if st.session_state.current_chat not in st.session_state.chats:
    new_chat()

current = st.session_state.chats[st.session_state.current_chat]

# ---------- Sidebar ----------
with st.sidebar:
    st.button(":material/edit_square: New chat", on_click=new_chat, key="new_chat_btn")

    st.markdown('<div class="side-heading">Projects</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-empty">No projects</div>', unsafe_allow_html=True)

    with st.expander("Features", expanded=False):
        for name in FEATURES:
            is_active = st.session_state.feature == name
            if st.button(name, key=f"feat_{name}", type="primary" if is_active else "secondary"):
                st.session_state.feature = None if is_active else name
                st.rerun()

    with st.expander("Recents", expanded=True):
        chats = [
            (cid, c) for cid, c in st.session_state.chats.items()
            if c["messages"] or cid == st.session_state.current_chat
        ]
        # pinned first, then newest first
        chats = sorted(reversed(chats), key=lambda x: not x[1]["pinned"])
        if not chats:
            st.markdown('<div class="side-empty">No chats yet</div>', unsafe_allow_html=True)
        for cid, chat in chats:
            col_title, col_menu = st.columns([0.82, 0.18], vertical_alignment="center")
            label = ("📌 " if chat["pinned"] else "") + chat["title"]
            with col_title:
                if st.button(
                    label,
                    key=f"chat_{cid}",
                    type="primary" if cid == st.session_state.current_chat else "secondary",
                ):
                    st.session_state.current_chat = cid
                    st.rerun()
            with col_menu:
                with st.popover("⋯"):
                    new_title = st.text_input("Rename", value=chat["title"], key=f"rename_{cid}")
                    if new_title and new_title != chat["title"]:
                        chat["title"] = new_title
                        st.rerun()
                    if st.button("Unpin" if chat["pinned"] else "Pin", key=f"pin_{cid}"):
                        chat["pinned"] = not chat["pinned"]
                        st.rerun()
                    if st.button("Delete", key=f"del_{cid}"):
                        delete_chat(cid)
                        st.rerun()

# ---------- Main area ----------
st.title("AI Chatbot")
st.caption(f"Powered by Google Gemini - model: {MODEL}")
if st.session_state.feature:
    st.info(f"Mode: {st.session_state.feature}")

api_key = get_secret("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API key", type="password")
if not api_key:
    st.info("Add your Gemini API key to Streamlit secrets, a .env file, or the sidebar to start chatting.")
    st.stop()

client = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

for msg in current["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def ask(model):
    system = SYSTEM_PROMPT
    if st.session_state.feature:
        system += f" The user has selected the '{st.session_state.feature}' mode; focus your help on that."
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}] + current["messages"],
        stream=True,
    )
    return st.write_stream(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices)


if prompt := st.chat_input("Type your message..."):
    if not current["messages"]:
        current["title"] = prompt[:40] + ("..." if len(prompt) > 40 else "")
    current["messages"].append({"role": "user", "content": prompt})
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
            current["messages"].pop()
            st.stop()
    current["messages"].append({"role": "assistant", "content": reply})
    st.rerun()  # refresh sidebar so the new chat title shows in Recents
