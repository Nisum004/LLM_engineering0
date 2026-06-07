import os
import ollama
import gradio as gr
from deep_translator import GoogleTranslator
from openai import OpenAI

# --- Clients ---
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1" , api_key= os.getenv("GROQ_API_KEY"))
# --- Processing function ---
def process_news(audio_file):
    # Step 1: Transcribe Nepali audio with Groq Whisper
    with open(audio_file, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            language="ne",
            response_format="text"
        )

    # Step 2: Translate Nepali → English
    english_text = GoogleTranslator(source="ne", target="en").translate(transcription)

    # Step 3: Generate news report using local Ollama llama3.1:8b
    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a professional news reporter. Write a short, clear English news report from the given translated content. Include a headline, a lead paragraph, and 2-3 key points."
            },
            {
                "role": "user",
                "content": f"Here is the translated Nepali news content:\n\n{english_text}\n\nWrite a short news report in English."
            }
        ]
    )
    report = response["message"]["content"]

    return transcription, english_text, report

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## Nepali News Reporter\nUpload a Nepali news MP3 → get an English news report.")

    audio_in = gr.Audio(type="filepath", label="Upload Nepali News MP3")
    btn = gr.Button("Generate Report")

    gr.Markdown("### Nepali Transcript")
    nepali_out = gr.Textbox(label="Nepali Transcript", lines=5)

    gr.Markdown("### English Translation")
    english_out = gr.Textbox(label="English Translation", lines=5)

    gr.Markdown("### News Report")
    report_out = gr.Markdown()

    btn.click(fn=process_news, inputs=audio_in, outputs=[nepali_out, english_out, report_out], show_progress="minimal")

demo.launch(inbrowser=True)