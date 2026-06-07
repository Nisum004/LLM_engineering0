import gradio as gr
from dotenv import load_dotenv

from answer import answer_question

load_dotenv(override=True)

def format_context(context):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        source = doc.metadata.get("source", "Unknown")
        doc_type = doc.metadata.get("doc_type", "")
        result += f"<span style='color: #ff7800;'>Source: {source}"
        if doc_type:
            result += f" | Type: {doc_type}"
        result += "</span><br><br>"
        result += doc.page_content.replace("\n", "<br>") + "<br><br>---<br><br>"
    return result


def chat(history):
    last_message = history[-1]["content"]
    prior = history[:-1]
    answer, context = answer_question(last_message, prior)
    history.append({"role": "assistant", "content": answer})
    return history, format_context(context)


def main():
    def put_message_in_chatbot(message, history):
        return "", history + [{"role": "user", "content": message}]

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="Insurellm Expert Assistant") as ui:   # ← theme moved out
        gr.Markdown("# Insurellm Expert Assistant\nAsk me anything about Insurellm!")

        with gr.Row():
            with gr.Column(scale=1):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,            
                )
                message = gr.Textbox(
                    placeholder="Ask anything about Insurellm...",
                    show_label=False,
                )

            with gr.Column(scale=1):
                context_display = gr.HTML(
                    value="<i>Retrieved context will appear here</i>",
                )

        message.submit(
            put_message_in_chatbot,
            inputs=[message, chatbot],
            outputs=[message, chatbot]
        ).then(
            chat,
            inputs=chatbot,
            outputs=[chatbot, context_display]
        )

    ui.launch(inbrowser=True, theme=theme)  


if __name__ == "__main__":
    main()