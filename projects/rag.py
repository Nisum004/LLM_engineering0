import os
import glob
import gradio as gr
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key= os.getenv("GROQ_API_KEY"))
groq_model = "openai/gpt-oss-120b"

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key = "ollama")
ollama_model = "llama3.1:8b"

knowledge = {}

filenames = glob.glob("/Users/nisumlimbu/Desktop/LLM_Enigneering/projects/knowledge-base/employees/*")

for filename in filenames:
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

filenames = glob.glob("/Users/nisumlimbu/Desktop/LLM_Enigneering/projects/knowledge-base/products/*")

for filename in filenames:
    name = Path(filename).stem
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

def get_relevant_context(message):
    text = ""

    for ch in message:
        if ch.isalpha() or ch.isspace():
            text += ch
    words = text.lower().split()

    relevant_context = []
    for word in words:
        if word in knowledge:
            relevant_context.append(knowledge[word])
    return relevant_context

def additional_context(message):
    relevant_context = get_relevant_context(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result


def chat(message, history):
    add_context = additional_context(message)
    system_message = SYSTEM_PREFIX + add_context
    print(system_message)
     # Clean gradio history for OPENAI USE GROQ. NO NEED FOR OLLAMA
    cleaned_history = []

    for msg in history:
        cleaned_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    messages = [{'role':'system','content':system_message}]+ cleaned_history + [{'role':'user','content':message}]
    response = groq.chat.completions.create(
        model = groq_model,
        messages = messages
    )
    return response.choices[0].message.content

def main():
    view = gr.ChatInterface(chat).launch(inbrowser=True)

if __name__ == "__main__":
    main()