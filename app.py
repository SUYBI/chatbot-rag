import streamlit as st
import re

from load_chatbot import init_llm_and_index

st.set_page_config(page_title="Assistant Kaydan 🤖", page_icon="🤖", layout="centered")

st.title("Assistant Kaydan 🤖💬")
st.write("Bienvenue ! Je suis là pour vous accompagner dans vos démarches. 😊")


def remove_think(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    return cleaned.strip()


# INITIALISATION DES MODELS ET DE L'INDEX
@st.cache_resource
def get_chat_engine():
    return init_llm_and_index()

chat_engine = get_chat_engine()

# 💬 SESSION STATE POUR L'HISTORIQUE

if "messages" not in st.session_state:
    st.session_state.messages = []


# 🧠 AFFICHAGE HISTORIQUE DU CHAT

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 📝 SAISIE UTILISATEUR

prompt = st.chat_input("Écrivez votre message ici...")

if prompt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ----- Réponse IA -----
    with st.chat_message("assistant"):
        with st.spinner("Kaydan rédige une réponse... 😊"):
            response = chat_engine.chat(prompt)
            clean_text = remove_think(response.response)

            st.markdown(clean_text)

    st.session_state.messages.append({"role": "assistant", "content": clean_text})