import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sqlite3



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

groq_model ="llama-3.3-70b-versatile"
openrouter_model = "openai/gpt-oss-120b:free"
ollama_model = "llama3.1:8b"

system_message = """You are a helpful airline ticket booking assistant. 
You help customers find ticket prices to different cities.

RULES:
- You have access to a tool called get_ticket_price
- ALWAYS use the tool to get prices, never guess or make up prices
- NEVER show raw JSON or tool call syntax to the user
- Respond naturally and conversationally
- If asked about other cities, say they are not available
"""
price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of an airlines ticket to the desination city",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city":{
                "type":"string",
                "description":"The city that the customer wants to travel to",
            },
        },
        "required": ['destination_city'],
        'additionalProperties': False
    }
}

tools = [{'type':'function', "function": price_function}]

DB = "prices.db"

with sqlite3.connect(DB) as conn:
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS prices (city TEXT PRIMARY KEY, price REAL)')
    conn.commit()

def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT price FROM prices WHERE city = ?', (city.lower(),))
        result = cursor.fetchone()
        return f"Ticket price to {city} is ${result[0]}" if result else "No price data available for this city"

def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT INTO prices (city, price) VALUES (?, ?) ON CONFLICT(city) DO UPDATE SET price = ?', (city.lower(), price, price))
        conn.commit()

ticket_prices = {"london":799, "paris": 899, "tokyo": 1420, "sydney": 2999}
for city, price in ticket_prices.items():
    set_ticket_price(city, price)

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        if tool_call.function.name == "get_ticket_price":
            arguments = json.loads(tool_call.function.arguments)
            city = arguments.get('destination_city')
            price_details = get_ticket_price(city)
            responses.append({
                "role": "tool",
                "content": price_details,
                "tool_call_id": tool_call.id
            })
    return responses

# FIXED FUNCTION

def chatt(message, history):
    history = [{"role": h["role"], "content": h["content"]} for h in history]
    messages = (
        [{"role": "system", "content": system_message}]
        + history
        + [{"role": "user", "content": message}]
    )

    response = groq.chat.completions.create(
        model=groq_model,
        messages=messages,
        tools=tools
    )

    while response.choices[0].finish_reason == "tool_calls":
        assistant_message = response.choices[0].message
        tool_responses = handle_tool_calls(assistant_message)

        # ✅ Convert tool_calls Pydantic objects → plain dicts
        messages.append({
            "role": "assistant",
            "content": assistant_message.content or "",   # ✅ None → ""
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in assistant_message.tool_calls    # ✅ serialized properly
            ]
        })

        messages.extend(tool_responses)                   # ✅ extend not append

        response = groq.chat.completions.create(
            model=groq_model,
            messages=messages,
            tools=tools
        )

    return response.choices[0].message.content

def main():
    gr.ChatInterface(fn = chatt).launch()

if __name__ == "__main__":
    main()
