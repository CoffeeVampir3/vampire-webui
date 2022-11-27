import gradio as gr
import sys
import random

#TODO:: Conversion script seems to have issues. Ommited for now.
def do_convert(file):
    print(file)

def create(css):
    block = gr.Blocks(css=css)
    with gr.Row():
        convert_model_btn = gr.Button(value="Convert.")
        convert_model_btn.click(do_convert, inputs=gr.File(label="File"), outputs=None, preprocess=False)
    return block