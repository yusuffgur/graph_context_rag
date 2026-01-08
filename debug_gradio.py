import gradio as gr
import inspect

print(f"Gradio Version: {gr.__version__}")
print(f"Chatbot Init Signature: {inspect.signature(gr.Chatbot.__init__)}")
