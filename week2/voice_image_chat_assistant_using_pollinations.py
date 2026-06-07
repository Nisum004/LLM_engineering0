import json
import gradio as gr
from groq import Groq
from PIL import Image
from io import BytesIO
import requests
import os
from dotenv import load_dotenv
# ── Clients ────────────────────────────────────────────────
load_dotenv()

groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_model = "llama-3.3-70b-versatile"

# ── System prompt & tools ──────────────────────────────────
system_message = """You are a helpful travel assistant. 
When users ask about travel destinations, use get_ticket_price 
to provide pricing information."""

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "Get ticket price for a destination city",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The destination city name"
                    }
                },
                "required": ["destination_city"]
            }
        }
    }
]

def get_ticket_price(city):
    prices = {
        "paris": "$800", "tokyo": "$1200",
        "kathmandu": "$600", "london": "$700"
    }
    return prices.get(city.lower(), "$500")

# ── Artist using Pollinations (no API key needed!) ─────────
def artist(city):
    prompt = f"A vacation in {city}, tourist spots and landmarks, cartoon style"
    encoded_prompt = requests.utils.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true"
    
    print(f"Generating image for {city}...")
    response = requests.get(url, timeout=60)
    
    image = Image.open(BytesIO(response.content))
    image.save(f"{city}_vacation.png")
    print(f"Saved: {city}_vacation.png")
    return image

# ── Talker using Groq Orpheus TTS ──────────────────────────
def talker(message):
    response = groq.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="autumn",
        input=message,
        response_format="wav"
    )
    audio_path = "response.wav"
    with open(audio_path, "wb") as f:
        f.write(response.read())
    return audio_path

# ── Tool call handler ──────────────────────────────────────
def handle_tool_calls_and_return_cities(message):
    responses = []
    cities = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get("destination_city")
            cities.append(city)
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses, cities

# ── Main chat function ─────────────────────────────────────
def chat(history):
    messages = [{"role": "system", "content": system_message}]
    messages += [{"role": h["role"], "content": h["content"]} for h in history]

    response = groq.chat.completions.create(
        model=groq_model,
        messages=messages,
        tools=tools
    )

    cities = []
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        responses, cities = handle_tool_calls_and_return_cities(message)
        messages.append(message)
        messages.extend(responses)
        response = groq.chat.completions.create(
            model=groq_model,
            messages=messages,
            tools=tools
        )

    reply = response.choices[0].message.content
    history = history + [{"role": "assistant", "content": reply}]

    audio_path = talker(reply)
    image = artist(cities[0]) if cities else None

    return history, audio_path, image

def put_message_in_chatbot(message, history):
    return "", history + [{"role": "user", "content": message}]

# ── Gradio UI ──────────────────────────────────────────────
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500)
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot]
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_output, image_output]
    )

def main():
    ui.launch(inbrowser=True)

if __name__ == "__main__":
    main()