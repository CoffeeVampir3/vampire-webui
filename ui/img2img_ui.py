import gradio as gr
import ui.ui_config as ui_config
from ui.ui_segments import *
from functools import partial

def create(css):
    block = gr.Blocks(css=css)

    app_inputs = []
    with gr.Row():
        with gr.Column():
            textseg = create_textbox_segment()
            setseg = create_settings_segment()

        with gr.Box(elem_id="gallerybox"):
            with gr.Row():
                model_dropdown, sampler_dropdown = create_model_settings_segment()
                launch_btn = gr.Button(value="Generate")

            app_inputs = [model_dropdown, sampler_dropdown, *textseg, *setseg]
            output_gallery = gr.Gallery()
            launch_btn.click(dp.run_pipeline, inputs=app_inputs, outputs=output_gallery)

        load_config = partial(ui_config.load_ui_config, model_dropdown)
        block.load(load_config, inputs=None, outputs=app_inputs)
    return block