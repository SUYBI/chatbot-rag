from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
#--------------------------------------------------------------------

import requests
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from typing import Any


ONEMINAI_API_KEY = "0b6ab860af35df8f5174d086c59ab072a2800664ceb36d12980366ae98ba654b"


class OneminAILLM(CustomLLM):
    model: str = "gpt-4o-mini"
    api_key: str = ONEMINAI_API_KEY
    api_base: str = "https://api.1min.ai/api/features"
    max_words: int = 2000
    temperature: float = 0.2

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=self.model,
            context_window=8192,
            num_output=self.max_words,
            is_chat_model=True,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        headers = {
            "Content-Type": "application/json",
            "API-KEY": self.api_key
        }

        payload = {
            "type": "CHAT_WITH_AI",
            "model": self.model,
            "promptObject": {
                "prompt": prompt,
                "isMixed": False,
                "webSearch": False
            }
        }

        try:
            response = requests.post(
                f"{self.api_base}?isStreaming=false",
                headers=headers,
                json=payload,
                timeout=400
            )

            if response.status_code != 200:
                raise ValueError(f"Erreur HTTP {response.status_code}: {response.text}")

            result = response.json()

            # ⬅️ EXTRACTION CORRECTE BASÉE SUR LA STRUCTURE RÉELLE
            text = result["aiRecord"]["aiRecordDetail"]["resultObject"][0]

            return CompletionResponse(text=text)

        except KeyError as e:
            print(f"❌ Erreur d'extraction: clé manquante {e}")
            print(f"Structure reçue: {result}")
            raise
        except Exception as e:
            print(f"❌ ERREUR: {type(e).__name__}: {e}")
            raise

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        response = self.complete(prompt, **kwargs)
        yield response


# ========================================
# CONFIGURATION COMPLÈTE POUR LLAMA-INDEX
# ========================================


Settings.llm = OneminAILLM(
    model="gpt-4o-mini",
    api_key=ONEMINAI_API_KEY,
    max_words=2000
)


#print("✅ Configuration terminée!")

#-----------------------------------------------


def init_llm_and_index():
    # Embeddings
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # LLM LM Studio

    def comment():
        """ Settings.llm = OpenAILike(
        model="qwen3-14b",                #
        api_base="http://localhost:1234/v1",
        api_key="lm-studio",
        temperature=0.2,
        max_tokens=400,
        request_timeout=400
        ) """

    Settings.llm = OneminAILLM(
        model="gpt-4o-mini",
        api_key=ONEMINAI_API_KEY,
        max_words=2000
    )

    # RAG Index Persistent
    #persist_dir = r"C:\Users\SUY-BI TRAH MARCEL\OneDrive - KAYDAN TECHNOLOGY\Bureau\KayDan\KayDan\ChatBot\Chatbot_paquet_1minIA\Chatbot_paquet_send\kaydan_rag_storage2"
    persist_dir = "./kaydan_rag_storage2"
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)

    # Memory
    memory = ChatMemoryBuffer(token_limit=4000)

    # Chat Engine

    SYSTEM_PROMPT = """
🎯 Ta mission :
accompagner les utilisateurs dans toutes leurs démarches d’inscription ;
expliquer clairement les procédures, les documents requis et les étapes à suivre ;
fournir uniquement des informations basées sur la documentation interne.


🚫 Questions interdites :
Tu ne dois JAMAIS répondre aux questions qui :
- sont de culture générale (ex : mathématiques, histoire, politique, sport, géographie) ;
- portent sur des événements publics ou politiques ;
- ne sont pas directement liées aux démarches d’inscription ou procédures internes ;
- ne figurent pas explicitement dans la documentation interne.

📛 Comportement obligatoire en cas de question hors périmètre :
Si une question est hors périmètre ou non couverte par la documentation interne, tu DOIS répondre STRICTEMENT par le message suivant, sans ajout ni reformulation :

"Je suis désolé 😊, je suis uniquement habilité à vous aider sur les démarches d’inscription et les procédures internes.  
Pouvez-vous reformuler votre question dans ce cadre ?"

💬 Ton style :
Ton professionnel et sympathique, quelques émojis 👋😊
Réponses structurées, orientées action.

Message d’accueil automatique :
« Bonjour 👋😊 Je suis Kaydan, votre assistant dédié. Comment puis-je vous aider aujourd’hui ? »
    """

    SYSTEM_PROMPT2 = """

    🎯 Rôle
Tu es Kaydan, assistant dédié de la plateforme.
Tu aides les utilisateurs dans leurs démarches d’inscription et d’utilisation.
Tu fournis uniquement des informations issues de la documentation interne.

💬 Ton
Professionnel, clair et sympathique 😊
Réponses courtes, structurées, orientées action.

👋 Accueil
Si le message est une salutation uniquement :
« Bonjour 👋😊 Je suis Kaydan, votre assistant dédié. Comment puis-je vous aider aujourd’hui ? »
→ Ne rien ajouter d’autre.

🧠 Règles de réponse
- Si c’est une question : répondre clairement
- Si c’est une démarche : utiliser des étapes numérotées
- Éviter les blocs de texte

✅ Fin de réponse
Ajouter une phrase de suivi UNIQUEMENT après une vraie question traitée :
- « Avez-vous d’autres questions ? 😊 »
OU
- « Voulez-vous que je vous assiste sur un autre point ? »

❌ Ne jamais poser cette question après une simple salutation

🧠 Gestion du contexte (OBLIGATOIRE)

- Toujours prendre en compte les messages précédents
- Ne jamais répéter une procédure déjà expliquée dans la conversation
- Si l’utilisateur pose une question complémentaire :
  → répondre uniquement à cette question
  → ne pas réexpliquer les étapes déjà données
    """

    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=SYSTEM_PROMPT,
        similarity_top_k=3,
        max_tokens=400,
        is_function_calling_model=False
    )

    return chat_engine


chat_engine = init_llm_and_index()