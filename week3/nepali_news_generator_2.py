import os
import requests
import ollama
import gradio as gr
from io import BytesIO
from PIL import Image
from deep_translator import GoogleTranslator
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
groq_api = os.getenv("GROQ_API_KEY")
groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api)


def generate_news_image(english_text):
    prompt_response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You create short, visual image prompts for news illustrations. Reply with ONLY the image prompt, no explanations. Max 25 words. Describe a realistic news scene."
            },
            {
                "role": "user",
                "content": f"Create an image prompt for this news:\n\n{english_text}"
            }
        ]
    )
    visual_prompt = prompt_response["message"]["content"].strip()

    encoded_prompt = requests.utils.quote(visual_prompt + ", news illustration, photorealistic, high quality")
    url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=576&nologo=true"
    response = requests.get(url, timeout=120)
    image = Image.open(BytesIO(response.content))
    return image


def process_news(audio_file):
    with open(audio_file, "rb") as f:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="text",
        )

    english_text = GoogleTranslator(source="ne", target="en").translate(transcription)

    response = ollama.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a professional news reporter. Write a short, clear English news report from the given translated content that includes custom duty of rupees 100. Include a headline (BOLD AND LARGE FONT), a lead sentence, and key points. do not make up anything not included in the translated content."
            },
            {
                "role": "user",
                "content": f"Here is the translated Nepali news content:\n\n{english_text}\n\nWrite a short news report in English."
            }
        ]
    )
    report = response["message"]["content"]

    news_image = generate_news_image(english_text)

    return transcription, english_text, report, news_image


with gr.Blocks() as demo:
    gr.Markdown("## Nepali News Reporter\nUpload a Nepali news MP3 → get an English news report with illustration.")

    audio_in = gr.Audio(type="filepath", label="Upload Nepali News MP3")
    btn = gr.Button("Generate Report")

    gr.Markdown("### Nepali Transcript")
    nepali_out = gr.Textbox(label="Nepali Transcript", lines=5)

    gr.Markdown("### English Translation")
    english_out = gr.Textbox(label="English Translation", lines=5)

    gr.Markdown("### News Report")
    report_out = gr.Markdown()

    gr.Markdown("### News Illustration")
    image_out = gr.Image(label="Generated News Image", type="pil")

    btn.click(
        fn=process_news,
        inputs=audio_in,
        outputs=[nepali_out, english_out, report_out, image_out],
        show_progress="minimal"
    )

demo.launch(inbrowser=True)