from langchain_core import messages
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

# Vector Embedding Models:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

# LLM Chat Abstraction:
from langchain_ollama import ChatOllama
from langchain_huggingface import ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Vector Databases:
from pinecone import Pinecone
from langchain_chroma import Chroma

## ------ OLLAMA -------
# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text",
# )
load_dotenv(override=True)
groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key= os.getenv("GROQ_API_KEY"))
groq_model = "openai/gpt-oss-120b"

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key = "ollama")
ollama_chat_model = "llama3.1:8b"
ollama_embed_model = "nomic-embed-text"

load_dotenv(override=True)

DB_NAME = str(Path(".") / "vector_db")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

RETRIEVAL_K = 10

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.

Context:
{context}
"""

vector_store = Chroma(embedding_function=embeddings, persist_directory=DB_NAME)
retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_K})

# ChatGroq needs the model name without "openai/" prefix
llm = ChatGroq(
    temperature=0,
    model= groq_model,  # or your preferred Groq model
    groq_api_key=os.getenv("GROQ_API_KEY")
)


def fetch_context(question: str) -> list[Document]:
    return retriever.invoke(question)

def combined_question(question: str, history: list[dict] | None = None) -> str:
    if history is None:
        history = []
    question = str(question)
    prior = "\n".join(
        str(m["content"])
        for m in history
        if isinstance(m, dict) and "content" in m
    )
    if prior:
        return prior + "\n" + question
    return question


def answer_question(question: str, history: list[dict] | None = None) -> tuple[str, list[Document]]:
    if history is None:
        history = []
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)

    messages = [SystemMessage(content=system_prompt)]
    # filter out any SystemMessages from history to avoid overwriting context
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))

    response = llm.invoke(messages)
    return response.content, docs
