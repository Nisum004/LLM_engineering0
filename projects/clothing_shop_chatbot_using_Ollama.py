# imports

import os
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr

load_dotenv(override=True)
openrouter_api = os.getenv("OPENROUTER_API_KEY")
groq_api = os.getenv("GROQ_API_KEY")

if openrouter_api:
    print("Openrouter api found")
else:
    print("OPENROUTER NOT FOUND")
if groq_api:
    print("groq api found")
else:
    print("Groq api not found")

groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key= groq_api)

openrouter = OpenAI(base_url="https://openrouter.ai/api/v1" , api_key= openrouter_api)

ollama = OpenAI(base_url="http://localhost:11434/v1", api_key= "ollama")

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."

system_message += "\nIf the customer asks for shoes, you should respond that shoes are not on sale today, \
but remind the customer to look at hats!"


def chatbott(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    relevant_system_message = system_message
    if 'belt' in message.lower():
        relevant_system_message += " The store does not sell belts; if you are asked for belts, be sure to point out other items on sale."
    
    messages = [{"role": "system", "content": relevant_system_message}] + history + [{"role": "user", "content": message}]

    stream = ollama.chat.completions.create(model="llama3.1:8b", messages=messages, stream=True)

    response = ""
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

def main():
    gr.ChatInterface(fn = chatbott).launch(share=True)

if __name__ == "__main__":
    main()