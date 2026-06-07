import os
from dotenv import load_dotenv
from openai import OpenAI

import gradio as gr

load_dotenv(override=True)
groq_api = os.getenv("GROQ_API_KEY")

if groq_api:
    print("groq api found")
else:
    print("Groq api not found")

groq = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api)

system_message = "You are a helpful assistant"

def message_groq(prompt):
    messages = [{"role":"system","content": system_message},
    {"role":"user", "content": prompt}]

    response = groq.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages= messages
    )
    return response.choices[0].message.content

message_input = gr.Textbox(label = "Your message:", info="Enter a message for AI MODEL")
message_output = gr.Textbox(label="Response:")

view = gr.Interface(
    fn = message_groq,
    inputs=[message_input],
    outputs=[message_output],
    examples=["What is a Blackhole?", "How many bananas in a dozan ?"],
    flagging_mode="never"
)
def main():
    view.launch()

if __name__ == "__main__":
    res = input("Enter 'start':")
    if res == 'start':
        main()
